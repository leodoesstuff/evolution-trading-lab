import copy
import pickle
import numpy as np
import pandas as pd
import pytest
import torch
from evo.config import Config
from evo.data import prepare,session_grid,ACTION_COLUMNS
from evo.day_rules import live_exit_required,flag_unclosed
from evo.engine import aggregate,breed,qualifies,new_run,run_experiment
from evo.model import Bot,episode
from evo.sim import Book,results,step,observation
from evo.store import Store


def day_market(start='2023-01-03',end='2023-01-04',gap=False):
    grid=session_grid(start,end)
    price=np.where(grid.tz_convert('America/New_York').date==grid[0].tz_convert('America/New_York').date(),100.,50.) if gap else np.full(len(grid),100.)
    rows=[]
    for s in Config().symbols:
        rows.extend((t,s,float(p),float(p),float(p),float(p),100000.) for t,p in zip(grid,price))
    bars=pd.DataFrame(rows,columns=['time','symbol','open','high','low','close','volume'])
    return prepare(bars,pd.DataFrame(columns=ACTION_COLUMNS),Config().symbols,{},train_end='2026-01-01')


def test_daily_exits_prevent_overnight_gap_losses():
    c=Config(rules_version=2,fee_bps=0,slippage_bps=0)
    m=day_market(gap=True);b=Bot.fresh('buyer',44,c)
    b.act=lambda obs,learning:np.full(5,2)
    r,book=episode(b,m,0,len(m.times)-1,c,False)
    assert r['net_return']==pytest.approx(0)
    assert np.allclose(book.shares,0)
    assert r['flat_at_close'] and r['roundtrips']==10
    assert r['mean_exposure']>.05 and r['active_days']==2
    assert not qualifies(r,.15,c)  # Flat profit is not qualification.
    for t in book.trades:
        stamp=pd.Timestamp(t[0]).tz_convert('America/New_York')
        if t[2]>0:assert stamp.hour<15 or stamp.minute<50
    assert observation(m,80,Book.fresh(c)).shape==(44,)


def test_early_close_uses_actual_exchange_calendar():
    c=Config(rules_version=2)
    m=day_market('2025-11-28','2025-11-28')
    b=Bot.fresh('buyer',44,c);b.act=lambda *a:np.full(5,2)
    r,book=episode(b,m,0,len(m.times)-1,c)
    sales=[pd.Timestamp(t[0]).tz_convert('America/New_York') for t in book.trades if t[2]<0]
    assert sales and all(t.hour==12 and t.minute>=50 for t in sales)
    assert np.allclose(book.shares,0)


def test_cash_only_and_tiny_exposure_cannot_breed_or_win():
    c=Config(rules_version=2,warmup_generations=0,population=5)
    m=day_market()
    b=Bot.fresh('cash',44,c);b.act=lambda *a:np.zeros(5,dtype=int)
    r,book=episode(b,m,0,len(m.times)-1,c)
    assert r['alive'] and not r['selection_eligible'] and r['roundtrips']==0
    assert r['score']==0 and not qualifies(r,.15,c)
    state=dict(generation=4,results=[r]*5,bots=[b.pack()]*5)
    children=breed(state,44,c)
    assert all(p['parent']=='' for p in children)
    book=Book.fresh(c)
    book.analytics=dict(bars=100,exposure_sum=.1,roundtrips=100,days={'d':dict(start=10000,end=10001,active=True)})
    book.observe(np.full(5,100.),'2023-01-03T20:00:00Z',c)
    assert not results(book,c)['activity_eligible']


def test_unfilled_exit_fails_without_fabricating_a_trade():
    c=Config(rules_version=2)
    m=day_market('2023-01-03','2023-01-03')
    book=Book(0,np.full(5,20.),10000,rules_version=2)
    m.opens[-1]=np.nan;m.volumes[-1]=0
    step(m,len(m.times)-1,book,np.zeros(5),c)
    assert book.analytics['flat_failed'] and not book.alive
    assert not book.trades and np.all(book.shares==20)


def test_daily_loss_stop_flattens_and_blocks_reentry():
    c=Config(rules_version=2,fee_bps=0,slippage_bps=0)
    m=day_market('2023-01-03','2023-01-03')
    book=Book.fresh(c)
    step(m,1,book,np.full(5,.2),c)
    m.opens[2]=96;m.closes[2]=96;m.marks[2]=96
    step(m,2,book,np.full(5,.2),c)
    assert book.analytics['loss_stops']==1 and np.allclose(book.shares,0)
    step(m,3,book,np.full(5,.2),c)
    assert np.allclose(book.shares,0)


def test_legacy_checkpoint_rules_and_book_load_unchanged():
    c=Config(**{'seed':7})
    assert c.rules_version==1
    book=Book.fresh(c)
    del book.analytics;del book.rules_version
    restored=pickle.loads(pickle.dumps(book))
    assert restored.rules_version==1 and restored.analytics=={}
    m=day_market(gap=True);b=Bot.fresh('legacy',42,c);b.act=lambda *a:np.full(5,2)
    r,book=episode(b,m,0,len(m.times)-1,c)
    assert 'rules_version' not in r and not book.alive


def test_activity_aggregation_and_postwarmup_profit_gate():
    c=Config(rules_version=2,population=5)
    m=day_market();b=Bot.fresh('buyer',44,c);b.act=lambda *a:np.full(5,2)
    r,book=episode(b,m,0,len(m.times)-1,c)
    assert r['selection_eligible'] and r['net_return']<0
    result=aggregate([r,copy.deepcopy(r)])
    assert result['roundtrips']==20
    assert result['score']==pytest.approx(result['score_parts']['profit']-result['score_parts']['drawdown_penalty']-result['score_parts']['consistency_penalty'])
    state=dict(generation=4,results=[result]*5,bots=[b.pack()]*5)
    assert all(not p['parent'] for p in breed(state,44,c))
    assert not qualifies(result,.15,c)
    bad=copy.deepcopy(r);bad['selection_eligible']=bad['activity_eligible']=False
    assert not aggregate([r,bad])['selection_eligible']


def test_live_exit_and_missed_close_have_same_constraints():
    c=Config(rules_version=2)
    book=Book(0,np.full(5,20.),10000,rules_version=2)
    assert live_exit_required(book,c,np.full(5,100.),pd.Timestamp('2026-11-27T17:50:00Z'))  # 12:50 early close.
    assert flag_unclosed(book,c,pd.Timestamp('2026-11-27T18:00:00Z'))
    assert not book.alive and book.analytics['flat_failed']
