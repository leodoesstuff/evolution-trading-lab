"""Resumable experiment state machine. Only the worker mutates checkpoints."""
import copy
import json
import random
import time
import uuid
import numpy as np
import torch
from .config import Config
from .data import load_dataset
from .model import Bot, episode
from .store import Store
from .sim import benchmark
from .day_rules import enabled, eligible, breeding_eligible
from .training_windows import practice_plan, learning_window


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def new_run(home, dataset, config):
    config.validate()
    torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    market = load_dataset(dataset, config.symbols, config.train_end)
    practice_plan(market,config)
    market.bounds(config.train_end, config.validation_end)
    market.bounds(config.validation_end, config.test_end)
    run = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    bots = [Bot.fresh(f'g000-b{i:03}', market.features.shape[1] + len(config.symbols) + 2 + (2 if enabled(config) else 0),
                      config).pack() for i in range(config.population)]
    state = dict(run=run, config=config.to_dict(), dataset=str(dataset),
                 fingerprint=market.meta['fingerprint'], meta=market.meta,
                 generation=0, cursor=0, phase='training', status='ready',
                 bots=bots, results=[], hall=[], validation=[], test=None,
                 champion=None, paper=None, rng=rng_state(), elapsed=0.0,
                 validation_cursor=0, last_error=None)
    state['baselines'] = {}
    store = Store(home)
    store.save(run, state)
    store.event(run, 'created', config=config.to_dict(), feed=market.meta['feed'])
    return run


def aggregate(items):
    result=dict(score=float(np.median([r['score'] for r in items])),
                net_return=float(np.mean([r['net_return'] for r in items])),
                drawdown=max(r['drawdown'] for r in items),alive=all(r['alive'] for r in items),windows=items)
    if items and items[0].get('rules_version',1)>=2:
        result.update(rules_version=2,selection_eligible=all(r['selection_eligible'] for r in items),
                      activity_eligible=all(r['activity_eligible'] for r in items),
                      flat_at_close=all(r['flat_at_close'] for r in items),
                      roundtrips=sum(r['roundtrips'] for r in items),days=sum(r['days'] for r in items),
                      active_days=sum(r['active_days'] for r in items),
                      eligibility_reason='All practice windows active and within risk limits' if all(r['selection_eligible'] for r in items) else 'One or more practice windows failed activity or risk checks')
        for key in ('mean_exposure','active_bar_fraction','turnover','fees','slippage_cost','profit_dollars'):
            result[key]=float(np.mean([r[key] for r in items]))
        result['score_parts']={key:float(np.mean([r['score_parts'][key] for r in items])) for key in items[0]['score_parts']}
        result['score']=result['score_parts']['profit']-result['score_parts']['drawdown_penalty']-result['score_parts']['consistency_penalty']
        if not result['alive']:result['score']-=1000
        result['worst_day']=min(r['worst_day'] for r in items)
        result['profitable_days']=sum(r['profitable_days'] for r in items)
        result['daily_loss_stops']=sum(r['daily_loss_stops'] for r in items)
    return result


def qualifies(result, limit, config=None):
    return bool(result and result['alive'] and result['net_return']>0 and result['drawdown']<limit and
                (not config or not enabled(config) or eligible(result,True)))


def coverage_qualifies(market, config):
    from .data import session_grid
    import pandas as pd
    periods = [('2020-01-01', config.train_end), (config.train_end, config.validation_end),
               (config.validation_end, config.test_end)]
    for start, end in periods:
        expected = session_grid(start, str((pd.Timestamp(end) - pd.Timedelta(days=1)).date()))
        mask = (market.times >= pd.Timestamp(start, tz='UTC')) & (market.times < pd.Timestamp(end, tz='UTC'))
        valid = np.isfinite(market.opens[mask]) & np.isfinite(market.closes[mask]) & (market.volumes[mask] > 0)
        if not len(expected) or (valid.sum(axis=0) / len(expected) < .70).any():
            return False
    return True


def breed(state, inputs, config):
    generation = state['generation'] + 1
    ranked = sorted([(r, p) for r, p in zip(state['results'], state['bots']) if breeding_eligible(r,config,state['generation'])],
                    key=lambda pair: pair[0]['score'], reverse=True)
    count = config.population
    elites = min(max(1, count // 5), len(ranked))
    descendants = int(count * .6) if ranked else 0
    next_bots = []
    for _, packed in ranked[:elites]:
        next_bots.append(copy.deepcopy(packed))
    for i in range(descendants):
        parent = Bot.unpack(random.choice(ranked[:max(elites, 1)])[1], inputs, config)
        next_bots.append(parent.child(f'g{generation:03}-b{len(next_bots):03}', generation, config).pack())
    while len(next_bots) < count:
        next_bots.append(Bot.fresh(f'g{generation:03}-b{len(next_bots):03}', inputs, config, generation).pack())
    return next_bots


def write_report(store, state):
    summary = {k: state.get(k) for k in ('run', 'config', 'generation', 'status', 'phase',
                                        'fingerprint', 'validation', 'test', 'elapsed', 'promotion_reason', 'baselines', 'continuation', 'seed_source')}
    summary['feed'] = state['meta']['feed']
    summary['synthetic'] = state['meta'].get('synthetic', False)
    summary['champion'] = state['champion']['id'] if state.get('champion') else None
    (store.home / state['run'] / 'report.json').write_text(json.dumps(summary, indent=2))


def run_experiment(home, run, minutes=None):
    torch.set_num_threads(1)
    store = Store(home)
    state = store.load(run)
    config = Config(**state['config'])
    if state['status'] in ('stopped', 'complete', 'paper_eliminated'):
        return state
    market = load_dataset(state['dataset'], config.symbols, config.train_end)
    if market.meta['fingerprint'] != state['fingerprint']:
        raise ValueError('Underlying dataset changed. Create a new experiment.')
    plan=practice_plan(market,config)
    inputs = market.features.shape[1] + len(config.symbols) + 2 + (2 if enabled(config) else 0)
    restore_rng(state['rng'])
    begin = time.monotonic()
    deadline = begin + 60 * (minutes if minutes is not None else config.minutes)
    state['status'], state['last_error'] = 'running', None

    def interrupted():
        return time.monotonic() >= deadline or store.control(run) != 'run'

    def save():
        nonlocal begin
        now = time.monotonic()
        state['elapsed'] += now - begin
        begin = now
        state['rng'] = rng_state()
        store.save(run, state)
        write_report(store, state)

    save()
    try:
        while not interrupted():
            phase = state['phase']
            if phase == 'training':
                if state['cursor'] >= config.population:
                    if config.stop_on_extinction and not any(r['alive'] for r in state['results']) and not state['hall']:
                        state.update(phase='finished',status='complete',
                                     promotion_reason='All bots reached risk limits during training. No surviving candidate; stopped instead of repeatedly replacing the entire population.')
                        store.event(run,'population_extinct',generation=state['generation'])
                        save()
                        return state
                    if state['generation'] + 1 >= config.generations:
                        state['phase'] = 'validation'
                        state['validation_cursor'] = 0
                        save()
                        continue
                    next_bots = breed(state, inputs, config)
                    store.event(run, 'generation_finished', generation=state['generation'],
                                survivors=sum(r['alive'] for r in state['results']),
                                breeding_eligible=sum(breeding_eligible(r,config,state['generation']) for r in state['results']),
                                retained=sum(b['id'] in {p['id'] for p in state['bots']} for b in next_bots),
                                descendants=sum(bool(b['parent']) and b['birth']==state['generation']+1 for b in next_bots),
                                training_selection=config.training_selection)
                    state.update(bots=next_bots, generation=state['generation'] + 1,
                                 cursor=0, results=[])
                    save()
                    continue
                i = state['cursor']
                old_rng = rng_state()
                bot = Bot.unpack(state['bots'][i], inputs, config)
                start,end=learning_window(plan)
                learned, training_book = episode(bot, market, start, end,
                                                 config, True, interrupted)
                if learned is None:
                    restore_rng(old_rng)
                    break
                evaluations, books = [], []
                if learned['alive']:
                    windows = plan['evaluation']
                    for start,end in windows:
                        result, book = episode(bot, market, start, end,
                                               config, False, interrupted)
                        if result is None:
                            break
                        evaluations.append(result)
                        books.append(book)
                    if len(evaluations) != len(windows):
                        restore_rng(old_rng)
                        break
                    result = aggregate(evaluations)
                else:
                    result = learned
                bot.alive = result['alive']
                packed = bot.pack()
                state['bots'][i] = packed
                state['results'].append(result)
                state['cursor'] += 1
                state['simulated_bars'] = state.get('simulated_bars', 0) + len(training_book.curve)//2 + sum(len(b.curve)//2 for b in books)
                if result['alive'] and (not enabled(config) or eligible(result, True)):
                    # Keep only the best snapshot of each lineage ID in the hall of fame.
                    hall = [h for h in state['hall'] if h['bot']['id'] != bot.id]
                    previous = next((h for h in state['hall'] if h['bot']['id'] == bot.id), None)
                    candidate = {'bot': copy.deepcopy(packed), 'result': result}
                    hall.append(previous if previous and previous['result']['score'] > result['score'] else candidate)
                    state['hall'] = sorted(hall, key=lambda h: h['result']['score'], reverse=True)[:5]
                store.metric(run, state['generation'], bot, result)
                # Full trades for all learned lives; evaluation traces for the first fixed window.
                store.log_trace(run, bot.id, f'training:g{state["generation"]}', bot.version,
                                training_book.trades, training_book.curve, book=training_book)
                if books:
                    store.log_trace(run, bot.id, f'evaluation:g{state["generation"]}', bot.version,
                                    books[0].trades, books[0].curve, book=books[0])
                if not result['alive']:
                    store.event(run, 'bot_eliminated', bot=bot.id, generation=state['generation'])
                save()
            elif phase == 'validation':
                if not state['hall']:
                    state.update(phase='finished', status='complete', promotion_reason='No surviving candidate.')
                    save()
                    return state
                if 'validation' not in state.get('baselines', {}):
                    a, b = market.bounds(config.train_end, config.validation_end)
                    result, book = benchmark(market, a, b, config, interrupted)
                    if result is None:
                        break
                    state.setdefault('baselines', {})['validation'] = {'buy_and_hold': result, 'cash_return': 0.0}
                    store.log_trace(run, 'buy-and-hold', 'validation', 0, book.trades, book.curve, book=book)
                    save()
                i = state['validation_cursor']
                if i >= len(state['hall']):
                    best = max(state['validation'], key=lambda r: r['result']['score'])
                    state['champion'] = state['hall'][best['index']]['bot']
                    state['champion_validation'] = best['result']
                    if not qualifies(best['result'], config.death_drawdown, config):
                        state.update(phase='finished', status='complete', promotion_reason='Champion failed validation.')
                        save()
                        return state
                    state['phase'] = 'test'
                    save()
                    continue
                old_rng = rng_state()
                bot = Bot.unpack(state['hall'][i]['bot'], inputs, config)
                a, b = market.bounds(config.train_end, config.validation_end)
                result, book = episode(bot, market, a, b, config, False, interrupted)
                if result is None:
                    restore_rng(old_rng)
                    break
                state['validation'].append({'index': i, 'bot': bot.id, 'result': result})
                state['validation_cursor'] += 1
                store.log_trace(run, bot.id, 'validation', bot.version, book.trades, book.curve, book=book)
                save()
            elif phase == 'test':
                # Claim BEFORE looking at any test result. Reused holdouts cannot auto-promote.
                scope = f'{"synthetic" if market.meta.get("synthetic") else "US-equities"}:{config.validation_end}:{config.test_end}'
                fresh_holdout = store.claim_holdout(scope, run)
                state['fresh_holdout'] = fresh_holdout
                if 'test' not in state.get('baselines', {}):
                    a, b = market.bounds(config.validation_end, config.test_end)
                    result, book = benchmark(market, a, b, config, interrupted)
                    if result is None:
                        break
                    state.setdefault('baselines', {})['test'] = {'buy_and_hold': result, 'cash_return': 0.0}
                    store.log_trace(run, 'buy-and-hold', 'final_test', 0, book.trades, book.curve, book=book)
                    save()
                old_rng = rng_state()
                bot = Bot.unpack(state['champion'], inputs, config)
                a, b = market.bounds(config.validation_end, config.test_end)
                result, book = episode(bot, market, a, b, config, False, interrupted)
                if result is None:
                    restore_rng(old_rng)
                    break
                state['test'] = result
                store.log_trace(run, bot.id, 'final_test', bot.version, book.trades, book.curve, book=book)
                promotion_ok = (fresh_holdout and not market.meta.get('synthetic') and
                            coverage_qualifies(market, config) and
                            qualifies(result, config.death_drawdown, config))
                if promotion_ok:
                    state.update(phase='paper', status='paper_ready',
                                 promotion_reason='Passed validation and untouched final test.')
                    store.event(run, 'promoted', bot=bot.id)
                else:
                    reason = ('Synthetic data cannot promote.' if market.meta.get('synthetic') else
                              'Final test already used by another experiment.' if not fresh_holdout else
                              'Champion failed final test or data coverage.')
                    state.update(phase='finished', status='complete', promotion_reason=reason)
                save()
                return state
            elif phase == 'paper':
                from .paper import run_paper
                save()
                return run_paper(store, state, config, deadline)
            else:
                return state
        control = store.control(run)
        state['status'] = 'stopped' if control in ('stop', 'emergency') else 'paused'
        if control == 'emergency':
            state['phase'] = 'finished'
        save()
        store.event(run, state['status'], phase=state['phase'])
        return state
    except Exception as exc:
        state['status'] = 'error'
        # Errors contain only our sanitized messages; never HTTP credential headers.
        state['last_error'] = str(exc)
        save()
        store.event(run, 'error', error=type(exc).__name__, message=str(exc))
        raise
