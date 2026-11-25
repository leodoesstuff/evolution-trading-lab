import copy
import numpy as np
import pandas as pd
import torch

from evo.cadence import decision_due
from evo.config import Config
from evo.data import demo_dataset
from evo.engine import new_run, run_experiment
from evo.model import Bot, episode
from evo.sim import Book, step
from evo.store import Store
from test_day_rules import day_market


def test_strategy_holds_between_decisions_and_learns_the_whole_period():
    torch.set_num_threads(1)
    market = day_market('2023-01-03', '2023-01-03')
    outcomes = {}
    for cadence in (1, 6):
        cfg = Config(rules_version=2, decision_bars=cadence, learn_every=8)
        bot = Bot.fresh('alternating', 44, cfg)
        calls, transitions = [], []
        def act(obs, learning):
            action = 2 if len(calls)%2 == 0 else 0
            calls.append(action)
            return np.full(5, action)
        bot.act = act
        bot.learn_batch = lambda batch: transitions.extend(copy.deepcopy(batch))
        result, book = episode(bot, market, 0, len(market.times)-1, cfg, True)
        assert len(book.curve) == 1+2*(len(market.times)-1)
        assert sum(t[5] for t in transitions) == len(market.times)-1
        assert len(transitions) == len(calls)
        assert all(np.all(t[1] != 3) for t in transitions)  # Credit real decisions, not automatic holds.
        assert transitions[-1][4] and not any(t[4] for t in transitions[:-1])
        assert result['flat_at_close'] and np.allclose(book.shares, 0)
        outcomes[cadence] = result
    assert outcomes[6]['fees'] < outcomes[1]['fees']/3
    assert outcomes[6]['slippage_cost'] < outcomes[1]['slippage_cost']/3
    assert outcomes[6]['roundtrips'] > 0 and outcomes[6]['activity_eligible']


def test_holding_skips_rebalancing_but_never_skips_loss_stops():
    cfg = Config(rules_version=2, decision_bars=6, fee_bps=0, slippage_bps=0)
    market = day_market('2023-01-03', '2023-01-03')
    book = Book.fresh(cfg)
    step(market, 1, book, np.full(5, .2), cfg)
    original = book.shares.copy()
    market.opens[2] = market.marks[2] = 102
    step(market, 2, book, np.zeros(5), cfg, hold=True)
    np.testing.assert_array_equal(book.shares, original)
    market.opens[3] = market.marks[3] = 96
    step(market, 3, book, np.full(5, .2), cfg, hold=True)
    assert book.analytics['loss_stops'] == 1 and not book.shares.any()


def test_cadence_respects_session_boundaries_and_legacy_default():
    cfg = Config(decision_bars=6)
    stamp = pd.Timestamp('2026-11-25T15:00:00Z')
    assert decision_due(None, stamp, cfg)
    assert not decision_due(stamp, stamp, cfg)
    assert not decision_due(stamp, stamp+pd.Timedelta(minutes=25), cfg)
    assert decision_due(stamp, stamp+pd.Timedelta(minutes=30), cfg)
    assert decision_due(stamp, stamp+pd.Timedelta(days=2), cfg)
    assert Config(**{'seed': 7}).decision_bars == 1


def test_extinction_stops_repeated_population_replacement(tmp_path):
    dataset = demo_dataset(tmp_path/'data', Config().symbols)
    cfg = Config(population=5, generations=100, train_steps=8, eval_steps=8,
                 eval_windows=1, stop_on_extinction=True)
    run = new_run(tmp_path/'runs', dataset, cfg)
    store = Store(tmp_path/'runs')
    state = store.load(run)
    state.update(cursor=5, results=[{'alive': False}]*5)
    store.save(run, state)
    ended = run_experiment(store.home, run, minutes=1)
    assert ended['phase'] == 'finished' and ended['status'] == 'complete'
    assert 'All bots reached risk limits' in ended['promotion_reason']
    assert ended['generation'] == 0 and ended['champion'] is None
    assert [b['id'] for b in ended['bots']] == [b['id'] for b in state['bots']]


def test_live_practice_uses_same_cadence_and_exits_between_decisions(tmp_path):
    from evo.arena import create_practice, process_quote
    from test_arena import source_state
    source = source_state(2)
    source['config']['decision_bars'] = 6
    session = create_practice(tmp_path, source, ['population:buyer'], learning=False)
    store = Store(tmp_path)
    state = store.load(session)
    cfg = Config(**state['config'])
    bot = Bot.unpack(state['participants'][0]['bot'], 44, cfg)
    full = day_market('2025-11-26', '2025-11-26')
    def quote(stamp):
        now = pd.Timestamp(stamp)
        market = copy.copy(full)
        count = int((full.times+pd.Timedelta(minutes=5) <= now).sum())
        for field in ('times','opens','closes','volumes','marks','features','splits','dividends','removed'):
            setattr(market,field,getattr(full,field)[:count])
        prices = np.full(5,100.)
        process_quote(state,[bot],[market],prices,prices,prices,np.full(5,10000.),now,store)
        return state['participants'][0]
    participant = quote('2025-11-26T15:00:00Z')
    assert participant['pending']
    quote('2025-11-26T15:00:05Z')
    assert participant['book'].shares.any() and participant['pending'] is None
    quote('2025-11-26T15:05:00Z')
    assert participant['pending'] is None
    quote('2025-11-26T15:30:00Z')
    assert participant['pending'] and not participant['pending'].get('liquidate')
    quote('2025-11-26T20:50:00Z')
    assert participant['pending']['liquidate']
    quote('2025-11-26T20:50:05Z')
    assert not participant['book'].shares.any()
