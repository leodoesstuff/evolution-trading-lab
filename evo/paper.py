"""Local paper portfolios. Only fresh present-time quotes can execute an order."""
import time
import os
import numpy as np
import pandas as pd
import requests
from .data import Alpaca, prepare, session_grid
from .model import Bot
from .sim import Book, allocation, observation
from .engine import rng_state, write_report
from .day_rules import live_exit_required, flag_unclosed
from .cadence import decision_due, transition_bars


def utcnow():
    return pd.Timestamp.now(tz='UTC')


def fresh_quotes(payload, symbols, now, max_age=60):
    quotes = payload.get('quotes') or {}
    prices, bid, ask, capacity = [], [], [], []
    for symbol in symbols:
        q = quotes.get(symbol)
        if not q:
            raise ValueError(f'Waiting: no live IEX quote for {symbol}.')
        age = (now - pd.Timestamp(q['t'])).total_seconds()
        b, a = float(q.get('bp', 0)), float(q.get('ap', 0))
        if not -2 <= age <= max_age or b <= 0 or a < b:
            raise ValueError(f'Waiting: stale or invalid live IEX quote for {symbol}.')
        bid.append(b)
        ask.append(a)
        prices.append((a + b) / 2)
        # Conservative: treat reported quote size as shares, without assuming round lots.
        capacity.append(max(0, min(float(q.get('bs', 0)), float(q.get('as', 0)))))
    return tuple(np.asarray(v, dtype=float) for v in (prices, bid, ask, capacity))


def pending_is_fresh(pending, now):
    return bool(pending and 0 < (now - pd.Timestamp(pending['created'])).total_seconds() <= 60)


def run_paper(store, state, config, deadline):
    if state['meta'].get('synthetic'):
        raise ValueError('Synthetic models cannot enter live paper trading.')
    client = Alpaca(cancelled=lambda: store.control(state['run']) != 'run')
    inputs = len(state['meta']['scaler_mean']) + len(config.symbols) * 2 + 2 + (2 if getattr(config,'rules_version',1)>=2 else 0)
    bot = Bot.unpack(state.get('paper', {}).get('bot', state['champion']) if state.get('paper') else
                     state['champion'], inputs, config)
    paper = state.get('paper') or dict(book=Book.fresh(config), bot=bot.pack(),
                                     last_bar=None, pending=None, transition=None, message='Connecting to IEX…')
    book = paper['book']
    book.version = bot.version
    # Pending orders never survive a worker restart; do not backfill missed executions.
    paper['pending'] = None
    paper['transition'] = None
    state['paper'] = paper
    store.save_model(state['run'], bot)
    started = time.monotonic()
    next_fetch = 0
    market = None

    def save():
        nonlocal started
        state['elapsed'] += time.monotonic() - started
        started = time.monotonic()
        paper['bot'] = bot.pack()
        state['rng'] = rng_state()
        store.save(state['run'], state)
        write_report(store, state)

    def flush():
        store.log_trace(state['run'], bot.id, 'paper', bot.version, book.trades, book.curve, book=book)
        book.trades.clear()
        # Keep one recent valuation in the checkpoint; all history is in SQLite.
        if book.curve:
            book.curve[:] = book.curve[-1:]

    save()
    while time.monotonic() < deadline and store.control(state['run']) == 'run':
        now = utcnow()
        day = str(now.tz_convert('America/New_York').date())
        grid = session_grid(day, day)
        is_open = bool(len(grid) and grid[0] <= now < grid[-1] + pd.Timedelta(minutes=5))
        try:
            if not is_open:
                if flag_unclosed(book,config,now):
                    state.update(status='paper_eliminated',phase='finished')
                    paper['message']='Could not exit before the close; no overnight fills invented.'
                    paper['pending']=None;save();return state
                paper['pending'] = None
                paper['message'] = 'Market closed. Holding positions; waiting for the next session.'
                state['status'] = 'paper_waiting'
                save()
                time.sleep(5)
                continue
            if time.monotonic() >= next_fetch:
                start = (now - pd.Timedelta(days=14)).isoformat()
                bars = client.bars(config.symbols, start, now.isoformat())
                # Do not observe an incomplete five-minute bar.
                bars = bars[pd.to_datetime(bars.time, utc=True) + pd.Timedelta(minutes=5) <= now]
                action_start = min(str(start)[:10], paper.get('actions_through', day))
                actions = client.actions(config.symbols, action_start, str((now + pd.Timedelta(days=45)).date()))
                if bars.empty:
                    raise ValueError('Waiting for completed IEX bars.')
                meta = dict(state['meta'], synthetic=False)
                market = prepare(bars, actions, config.symbols, meta, config.train_end)
                completed = np.flatnonzero(market.times + pd.Timedelta(minutes=5) <= now)
                if len(completed) < 79:
                    raise ValueError('Not enough recent completed bars for live features.')
                market_index = int(completed[-1])
                if not (np.isfinite(market.closes[market_index]) & (market.volumes[market_index] > 0)).all():
                    raise ValueError('Waiting: latest completed bar missing for a stock; no orders placed.')
                next_fetch = time.monotonic() + 15
                if paper.get('chart_bar') != str(market.times[market_index]):
                    folder = store.home / state['run']
                    bars.to_parquet(folder / 'live-bars.tmp.parquet', index=False)
                    os.replace(folder / 'live-bars.tmp.parquet', folder / 'live-bars.parquet')
                    paper['chart_bar'] = str(market.times[market_index])
                # Apply all intervening corporate actions, including across sleeps and restarts.
                # No historical trades are replayed.
                previous_day = paper.get('actions_through', day)
                due = actions[(actions['date'] >= previous_day) & (actions['date'] <= day)]
                for action_day, group in due.groupby('date', sort=True):
                    splits = np.ones(len(config.symbols))
                    removed = np.zeros(len(config.symbols), bool)
                    dividends = []
                    # Stable IDs prevent repeated actions after restarts; process splits before dividends.
                    fresh = group[~group['id'].isin(book.applied)]
                    for a in fresh.to_dict('records'):
                        if a['symbol'] not in config.symbols:
                            continue
                        j = config.symbols.index(a['symbol'])
                        if a['kind'] in ('forward_split', 'reverse_split'):
                            splits[j] *= a['ratio']
                        elif a['kind'] == 'cash_dividend':
                            dividends.append((j, a['cash'], a['payable']))
                        elif a['kind'] == 'worthless_removal':
                            removed[j] = True
                    if len(fresh):
                        key = 'events:' + ':'.join(sorted(fresh['id'].astype(str)))
                        book.actions(splits, dividends, removed, action_day, key)
                        book.applied.update(fresh['id'].tolist())
                book.actions(np.ones(len(config.symbols)), [], np.zeros(len(config.symbols), bool), day, f'pay:{day}')
                paper['actions_through'] = day
            # Fresh quotes are mandatory both for marking and order execution.
            payload = client.get('/v2/stocks/quotes/latest', {'symbols': ','.join(config.symbols), 'feed': 'iex'})
            quote_now = utcnow()
            mids, bids, asks, capacities = fresh_quotes(payload, config.symbols, quote_now)
            if store.control(state['run']) != 'run':
                break
            book.version = bot.version
            equity = book.observe(mids, quote_now, config)
            if not book.alive:
                state['status'], state['phase'] = 'paper_eliminated', 'finished'
                paper['pending'] = None
                paper['message'] = 'Eliminated: portfolio reached its 15% drawdown limit.'
                store.event(state['run'], 'paper_eliminated', bot=bot.id, equity=equity)
                flush()
                save()
                return state
            exiting=live_exit_required(book,config,mids,quote_now)
            if exiting and paper['pending'] and not paper['pending'].get('liquidate'):paper['pending']=paper['transition']=None
            if exiting and paper['pending'] is None and np.any(book.shares>1e-8):
                paper['pending']=dict(created=quote_now.isoformat(),targets=[0.]*len(config.symbols),version=bot.version,liquidate=True)
            pending = paper['pending']
            if pending and pending['created']==quote_now.isoformat():pending=None
            if pending:
                if pending_is_fresh(pending, quote_now):
                    targets = np.array(pending['targets'])
                    weights = book.shares * mids / max(equity, 1e-8)
                    execution = np.where(targets > weights, asks, bids)
                    book.version = pending['version']
                    # Quote-size capacity, not historical bar volume, limits live fills.
                    book.execute(targets, execution, capacities / config.participation, quote_now, config,liquidate=pending.get('liquidate',False))
                    book.observe(mids, quote_now, config)
                    paper['transition'] = None if pending.get('liquidate') else dict(obs=pending['obs'], action=pending['action'],
                                               equity=pending['equity'], dd=pending['dd'], bar=pending.get('bar'))
                    store.event(state['run'], 'paper_order_processed', bot=bot.id, version=pending['version'])
                else:
                    store.event(state['run'], 'paper_order_expired', bot=bot.id)
                paper['pending'] = None
            latest = market.times[market_index]
            recent = 0 <= (quote_now - (latest + pd.Timedelta(minutes=5))).total_seconds() <= 120
            if not exiting and recent and decision_due(paper['last_bar'], latest, config):
                # Portfolio inputs use present-time marks; market inputs use completed bars.
                obs = observation(market, market_index, book)
                obs[-(len(config.symbols) + 2):] = np.r_[book.shares * mids / max(equity, 1e-8),
                                                        book.cash / max(equity, 1e-8), book.drawdown(mids)]
                transition = paper['transition']
                if transition:
                    reward = 100 * np.log(max(equity, 1e-8) / max(transition['equity'], 1e-8))
                    reward -= 10 * max(0, book.max_drawdown - transition['dd'])
                    bot.learn(np.array(transition['obs'], dtype=np.float32), np.array(transition['action']),
                              float(reward), obs, False,
                              steps=transition_bars(transition.get('bar') or paper['last_bar'], latest, config))
                    store.save_model(state['run'], bot)
                    store.event(state['run'], 'model_updated', bot=bot.id, version=bot.version)
                    paper['transition'] = None
                action = bot.act(obs, True)
                targets = allocation(action, book, mids, config)
                paper['pending'] = dict(created=quote_now.isoformat(), targets=targets.tolist(),
                                        obs=obs.tolist(), action=action.tolist(), version=bot.version,
                                        equity=equity, dd=book.max_drawdown, bar=str(latest))
                paper['last_bar'] = str(latest)
            paper['prices'] = mids.tolist()
            paper['quote_time'] = quote_now.isoformat()
            paper['message'] = f'Live IEX simulation • {config.decision_bars*5}-minute strategy decisions • continuous risk checks'
            state['status'] = 'paper_running'
            flush()
            save()
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            # Drop queued orders after any data interruption. Resume only on fresh data.
            paper['pending'], paper['transition'] = None, None
            paper['message'] = ('Network unavailable. Waiting for fresh data.' if isinstance(exc, requests.RequestException)
                                else str(exc))
            state['status'] = 'paper_waiting'
            next_fetch = 0
            save()
        time.sleep(5)
    action = store.control(state['run'])
    paper['pending'] = None
    if action in ('emergency', 'stop'):
        state['status'], state['phase'] = 'stopped', 'finished'
        paper['message'] = 'Stopped. No pending orders; holdings remain recorded.'
    else:
        state['status'] = 'paused'
    save()
    return state
