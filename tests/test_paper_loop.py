"""End-to-end paper loop tests use a deterministic fake clock and read-only feed."""
import copy
import numpy as np
import pandas as pd
import requests
import torch
from evo.config import Config
from evo.data import ACTION_COLUMNS, prepare, session_grid
from evo.model import Bot
from evo.paper import run_paper
from evo.store import Store


def scenario(tmp_path, monkeypatch, behavior='normal', duration=325):
    import evo.paper as paper
    torch.set_num_threads(1)
    cfg = Config()
    clock = [0.0]
    epoch = pd.Timestamp('2026-11-25T15:00:00Z')
    now = lambda: epoch + pd.Timedelta(seconds=clock[0])
    monkeypatch.setattr(paper, 'utcnow', now)
    monkeypatch.setattr(paper.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(paper.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    grid = session_grid('2026-11-11', '2026-11-25')
    rows = []
    for symbol in cfg.symbols:
        for t in grid:
            rows.append((t,symbol,100.,100.1,99.9,100.,100000.))
    bars = pd.DataFrame(rows,columns=['time','symbol','open','high','low','close','volume'])
    actions = pd.DataFrame(columns=ACTION_COLUMNS)
    meta = dict(synthetic=False,feed='IEX',scaler_mean=[0.]*30,scaler_std=[1.]*30,fingerprint='test-only')
    market = prepare(bars,actions,cfg.symbols,meta)
    bot = Bot.fresh('champion',42,cfg)
    with torch.no_grad():
        bot.policy.actor.weight.zero_()
        bot.policy.actor.bias.copy_(torch.tensor([0.,0.,20.,0.]*5))
    store = Store(tmp_path)
    run = 'paper-test'
    store.request(run,'run')
    calls = [0]

    class Feed:
        def __init__(self, **kwargs):
            pass

        def bars(self,*args):
            return bars[bars.time <= now() - pd.Timedelta(minutes=5)].copy()

        def actions(self,*args):
            return actions.copy()

        def get(self,*args):
            calls[0] += 1
            if calls[0] == 2 and behavior == 'outage':
                raise requests.ConnectionError('fake network interruption')
            if calls[0] == 2 and behavior == 'emergency':
                store.request(run,'emergency')
            mid = 100 * (1 + clock[0] * .00001)
            q = dict(t=now().isoformat(),bp=mid-.01,ap=mid+.01,bs=100,**{'as':100})
            return {'quotes':{s:q for s in cfg.symbols}}

    monkeypatch.setattr(paper,'Alpaca',Feed)
    state = dict(run=run,champion=bot.pack(),paper=None,meta=market.meta,
                 elapsed=0.,phase='paper',status='paper_ready',config=cfg.to_dict(),
                 generation=0,validation=[],test={},fingerprint='test-only')
    frozen = copy.deepcopy(state['champion']['weights'])
    result = run_paper(store,state,cfg,duration)
    trades = store.rows('SELECT * FROM trades ORDER BY time,id')
    return result,trades,frozen,store,epoch


def test_paper_fills_fresh_quotes_and_keeps_learning(tmp_path,monkeypatch):
    state,trades,frozen,store,epoch = scenario(tmp_path,monkeypatch)
    assert len(trades) > 0
    assert pd.to_datetime(trades.time,utc=True).min() > epoch
    assert state['paper']['bot']['version'] >= 1
    assert all(torch.equal(v,state['champion']['weights'][k]) for k,v in frozen.items())
    assert any(not torch.equal(v,state['paper']['bot']['weights'][k]) for k,v in frozen.items())
    assert state['status'] == 'paused'
    updates = store.rows("SELECT * FROM events WHERE kind='model_updated'")
    assert len(updates) >= 1


def test_outage_discards_pending_order_instead_of_replaying_it(tmp_path,monkeypatch):
    state,trades,_,_,epoch = scenario(tmp_path,monkeypatch,behavior='outage')
    assert len(trades) > 0
    assert pd.to_datetime(trades.time,utc=True).min() >= epoch + pd.Timedelta(minutes=5)
    assert state['paper']['pending'] is None


def test_emergency_during_quote_request_prevents_execution(tmp_path,monkeypatch):
    state,trades,_,_,_ = scenario(tmp_path,monkeypatch,behavior='emergency')
    assert trades.empty
    assert state['status'] == 'stopped'
    assert state['paper']['pending'] is None
