"""Competition acceptance tests with a shared feed and fake market clock."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from evo.arena import create_practice,run_practice,saved_candidates,apply_actions
from evo.config import Config
from evo.data import ACTION_COLUMNS,session_grid
from evo.model import Bot
from evo.sim import Book
from evo.store import Store


def source_state(rules_version=1):
    cfg=Config(rules_version=rules_version);bots=[]
    for name,action in [('buyer',2),('cashbot',0)]:
        b=Bot.fresh(name,44 if rules_version>=2 else 42,cfg)
        with torch.no_grad():
            b.policy.actor.weight.zero_();b.policy.actor.bias.fill_(-20)
            b.policy.actor.bias.reshape(5,4)[:,action]=20
        bots.append(b.pack())
    return dict(run='source',config=cfg.to_dict(),bots=bots,hall=[],champion=None,paper=None,
                fingerprint='source-fingerprint',meta=dict(synthetic=False,feed='IEX',scaler_mean=[0.]*30,scaler_std=[1.]*30))


def run_scenario(tmp_path,monkeypatch,behavior='normal',seconds=325,learning=False,rules_version=1):
    import evo.arena as arena
    torch.set_num_threads(1)
    clock=[0.];epoch=pd.Timestamp('2026-11-25T15:00:00Z')
    def now():return epoch+pd.Timedelta(seconds=clock[0])
    monkeypatch.setattr(arena,'utcnow',now)
    monkeypatch.setattr(arena.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(arena.time,'sleep',lambda s:clock.__setitem__(0,clock[0]+s))
    source=source_state(rules_version);store=Store(tmp_path)
    store.save('source',source);original=(tmp_path/'source'/'checkpoint.pt').read_bytes()
    session=create_practice(tmp_path/'practice',source,list(saved_candidates(source)),learning)
    grid=session_grid('2026-11-11','2026-11-25')
    bars=pd.DataFrame([(t,s,100.,100.1,99.9,100.,100000.) for s in Config().symbols for t in grid],columns=['time','symbol','open','high','low','close','volume'])
    class Feed:
        def __init__(self):self.calls=0;self.bar_calls=0
        def bars(self,*a):self.bar_calls+=1;return bars[bars.time<=now()-pd.Timedelta(minutes=5)].copy()
        def actions(self,*a):return pd.DataFrame(columns=ACTION_COLUMNS)
        def get(self,path,params):
            self.calls+=1
            assert path=='/v2/stocks/quotes/latest'
            assert params['feed']=='iex'
            stamp=now()
            if self.calls==2 and behavior=='outage':stamp-=pd.Timedelta(minutes=2)
            if self.calls==2 and behavior=='emergency':Store(tmp_path/'practice').request(session,'emergency')
            mid=100*(1+clock[0]*.00001)
            quote={'t':str(stamp),'bp':mid-.01,'ap':mid+.01,'bs':100,'as':100}
            return {'quotes':{s:quote for s in Config().symbols}}
    feed=Feed();result=run_practice(tmp_path/'practice',session,seconds/60,feed)
    assert (tmp_path/'source'/'checkpoint.pt').read_bytes()==original
    return result,Store(tmp_path/'practice'),feed,epoch,source


def test_competition_learning_and_resume_preserve_portfolios(tmp_path,monkeypatch):
    state,store,feed,epoch,source=run_scenario(tmp_path,monkeypatch,learning=True)
    assert all(p['bot']['version']>=1 for p in state['participants'])
    assert list((store.home/state['run']/'models'/'entrant-1').glob('*.pt'))
    assert list((store.home/state['run']/'models'/'entrant-2').glob('*.pt'))
    cash=[p['book'].cash for p in state['participants']]
    shares=[p['book'].shares.copy() for p in state['participants']]
    resumed=run_practice(store.home,state['run'],.1,feed)
    assert resumed['started']==state['started']
    assert [p['book'].cash for p in resumed['participants']]==cash
    for expected,p in zip(shares,resumed['participants']):np.testing.assert_array_equal(expected,p['book'].shares)


def test_shared_feed_two_bots_baselines_and_isolation(tmp_path,monkeypatch):
    state,store,feed,epoch,source=run_scenario(tmp_path,monkeypatch)
    trades=store.rows('SELECT * FROM trades ORDER BY time,id')
    assert state['status']=='paused'
    assert set(trades.bot)=={'entrant-1','buy-and-hold'}
    assert feed.calls>50 and feed.bar_calls==2  # Not multiplied by number of bots.
    assert state['participants'][1]['book'].cash==10000
    assert state['participants'][0]['book'].cash<10000
    assert state['participants'][0]['bot']['version']==source['bots'][0]['version']
    times=store.rows('SELECT DISTINCT time,bot FROM equity')
    for bot in ['entrant-1','entrant-2','buy-and-hold','cash']:
        assert len(times[times.bot==bot])>0
    assert store.rows('SELECT * FROM holdouts').empty
    assert state['source_run']=='source'
    assert pd.to_datetime(trades.time,utc=True).min()>epoch
    restored=store.load(state['run'])
    assert restored['participants'][0]['book'].cash==state['participants'][0]['book'].cash


def test_stale_data_discards_all_pending_orders_and_recovers(tmp_path,monkeypatch):
    state,store,feed,epoch,_=run_scenario(tmp_path,monkeypatch,'outage')
    trades=store.rows("SELECT * FROM trades WHERE bot LIKE 'entrant-%'")
    assert not trades.empty
    assert pd.to_datetime(trades.time,utc=True).min()>=epoch+pd.Timedelta(minutes=5)
    assert state['connection']=='connected'


def test_competition_emergency_has_no_orders(tmp_path,monkeypatch):
    state,store,_,_,_=run_scenario(tmp_path,monkeypatch,'emergency')
    assert state['status']=='stopped'
    assert store.rows('SELECT * FROM trades').empty
    assert all(p['pending'] is None for p in state['participants'])


def test_market_closed_waits_without_network_calls(tmp_path,monkeypatch):
    import evo.arena as arena
    clock=[0.]
    monkeypatch.setattr(arena,'utcnow',lambda:pd.Timestamp('2026-11-26T15:00:00Z')+pd.Timedelta(seconds=clock[0]))
    monkeypatch.setattr(arena.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(arena.time,'sleep',lambda s:clock.__setitem__(0,clock[0]+s))
    source=source_state();session=create_practice(tmp_path,source,['population:buyer'])
    class NoNetwork:
        def get(self,*a):raise AssertionError('No network needed on a holiday')
    result=run_practice(tmp_path,session,.2,NoNetwork())
    assert result['status']=='paused' and result['connection']=='market closed'
    assert result['started'] is None
    assert result['participants'][0]['book'].cash==10000


def test_manual_practice_is_not_qualification_or_reset(tmp_path):
    source=source_state();source['meta']['synthetic']=True
    old=copy.deepcopy(source['bots'][0]['weights'])
    first=create_practice(tmp_path,source,['population:buyer'])
    second=create_practice(tmp_path,source,['population:buyer'])
    assert first!=second
    s=Store(tmp_path).load(first)
    assert s['source_synthetic'] and s['kind']=='practice'
    assert s['participants'][0]['book'].cash==10000
    assert all(torch.equal(old[k],source['bots'][0]['weights'][k]) for k in old)
    with pytest.raises(ValueError):create_practice(tmp_path,source,[])
    with pytest.raises(ValueError):create_practice(tmp_path,source,['unknown'])


def test_practice_actions_idempotent_for_bots_and_benchmark():
    actions=pd.DataFrame([['split','AAPL','forward_split','2026-11-25',2.,0.,''],['div','AAPL','cash_dividend','2026-11-25',1.,1.,'2026-11-27']],columns=ACTION_COLUMNS)
    b=Book(0,np.array([10.,0,0,0,0]),1000)
    for _ in range(2):apply_actions(b,actions,'2026-11-25','2026-11-25',Config().symbols)
    assert b.shares[0]==20 and sum(r['amount'] for r in b.receivables)==20
    apply_actions(b,actions,'2026-11-25','2026-11-27',Config().symbols)
    assert b.cash==20 and not b.receivables


def test_live_room_ui_can_create_session_without_touching_source(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    source=source_state();store=Store(tmp_path/'training');store.save('source',source)
    fingerprint=hashlib.sha256((tmp_path/'training'/'source'/'checkpoint.pt').read_bytes()).hexdigest()
    monkeypatch.setenv('APCA_API_KEY_ID','test-key');monkeypatch.setenv('APCA_API_SECRET_KEY','test-secret')
    monkeypatch.setattr('evo.live_ui.launch_practice',lambda *a:None)
    script=tmp_path/'room_test.py'
    script.write_text(f"from evo.store import Store\nfrom evo.live_ui import render_live_room\nrender_live_room(Store({str(tmp_path/'training')!r}).load('source'),{str(tmp_path/'practice')!r})\n")
    app=AppTest.from_file(str(script)).run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label=='Start live practice').click().run(timeout=30)
    assert not app.exception
    assert any('Live competition' in m.value for m in app.markdown)
    assert hashlib.sha256((tmp_path/'training'/'source'/'checkpoint.pt').read_bytes()).hexdigest()==fingerprint
    practice=Store(tmp_path/'practice').load(app.session_state['practice_session'])
    assert b'test-secret' not in (tmp_path/'practice'/practice['run']/'checkpoint.pt').read_bytes()


def test_v2_practice_uses_clock_inputs_and_exposure_metrics(tmp_path,monkeypatch):
    state,store,feed,epoch,source=run_scenario(tmp_path,monkeypatch,rules_version=2,learning=True)
    p=state['participants'][0]
    assert p['bot']['weights']['body.0.weight'].shape[1]==44
    assert p['bot']['version']>=1
    assert p['book'].analytics['bars']>0
    assert p['book'].analytics['exposure_sum']>0


def test_v2_live_exit_is_queued_then_filled_without_reentry(tmp_path):
    from evo.arena import process_quote
    from evo.model import Bot
    torch.set_num_threads(1)
    source=source_state(2);cfg=Config(**source['config'])
    session=create_practice(tmp_path,source,['population:buyer'],False)
    store=Store(tmp_path);state=store.load(session)
    p=state['participants'][0];p['book'].cash=0.;p['book'].shares[:]=20
    bot=Bot.unpack(p['bot'],44,cfg)
    now=pd.Timestamp('2026-11-25T20:50:00Z');prices=np.full(5,100.)
    process_quote(state,[bot],[None],prices,prices-.01,prices+.01,np.full(5,100.),now,store)
    assert p['pending']['liquidate'] and np.all(p['book'].shares==20)
    process_quote(state,[bot],[None],prices,prices-.01,prices+.01,np.full(5,100.),now+pd.Timedelta(seconds=5),store)
    assert np.allclose(p['book'].shares,0)
    trades=store.rows("SELECT * FROM trades WHERE bot='entrant-1'")
    assert len(trades)==5 and (trades.shares<0).all()
    assert p['pending'] is None
