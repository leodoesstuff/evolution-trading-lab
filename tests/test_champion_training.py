import copy
import hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from evo.champion_seed import latest_champion,seed_from_champion,transfer_champion
from evo.config import Config
from evo.data import demo_dataset
from evo.day_rules import breeding_eligible
from evo.engine import breed,new_run,qualifies,run_experiment,write_report
from evo.model import Bot
from evo.names import bot_name,GIVEN,MIDDLE,FAMILY
from evo.sim import Book,results
from evo.store import Store
from evo.training_windows import practice_plan
from test_day_rules import day_market


def source_champion(tmp_path):
    cfg=Config(population=5,generations=2,train_steps=8,eval_steps=8,eval_windows=1)
    dataset=demo_dataset(tmp_path/'data'/'demo',cfg.symbols)
    store=Store(tmp_path/'runs')
    source=new_run(store.home,dataset,cfg)
    state=store.load(source)
    bot=Bot.fresh('legacy-winner',42,cfg)
    bot.learn(np.zeros(42,dtype=np.float32),np.ones(5,dtype=int),.2,
              np.ones(42,dtype=np.float32),False)
    state.update(champion=bot.pack(),promotion_reason='Passed validation and untouched final test.',
                 status='complete',phase='finished')
    store.save(source,state);write_report(store,state)
    return store,source,state,bot


def modern_config():
    return Config(rules_version=2,population=5,generations=2,train_steps=78,eval_steps=78,
                  eval_windows=1,training_selection='active_survivors',session_aligned=True,
                  decision_bars=12,learn_every=8)


def test_transfer_keeps_decisions_and_adam_state_when_clock_inputs_are_inserted(tmp_path):
    torch.set_num_threads(1)
    store,source,state,old=source_champion(tmp_path)
    before=copy.deepcopy(state['champion'])
    cfg=modern_config()
    new=transfer_champion(state['champion'],Config(**state['config']),cfg,35)
    obs=torch.randn(12,42)
    expanded=torch.cat((obs[:,:35],torch.randn(12,2),obs[:,35:]),dim=1)
    with torch.no_grad():
        old_logits,old_value=old.policy(obs)
        new_logits,new_value=new.policy(expanded)
    torch.testing.assert_close(old_logits,new_logits)
    torch.testing.assert_close(old_value,new_value)
    assert not new.policy.body[0].weight[:,35:37].any()
    adam=new.optimizer.state_dict()['state'][0]
    assert adam['exp_avg'].shape==(32,44)
    assert not adam['exp_avg'][:,35:37].any()
    torch.testing.assert_close(adam['exp_avg'][:,:35],before['optimizer']['state'][0]['exp_avg'][:,:35])
    torch.testing.assert_close(adam['exp_avg'][:,37:],before['optimizer']['state'][0]['exp_avg'][:,35:])
    new.learn(expanded[0].numpy(),np.ones(5,dtype=int),.1,expanded[1].numpy(),False)
    assert new.version==old.version+1
    assert all(torch.isfinite(p).all() for p in new.policy.parameters())
    assert all(torch.equal(before['weights'][k],state['champion']['weights'][k]) for k in before['weights'])


def test_seeded_run_preserves_source_and_starts_new_comparisons(tmp_path):
    store,source,state,old=source_champion(tmp_path)
    original=(store.home/source/'checkpoint.pt').read_bytes()
    cfg=modern_config()
    run=seed_from_champion(store.home,source,cfg)
    seeded=store.load(run)
    assert run!=source and (store.home/source/'checkpoint.pt').read_bytes()==original
    assert seeded['generation']==seeded['cursor']==0
    assert seeded['hall']==seeded['validation']==seeded['results']==[]
    assert seeded['champion'] is seeded['test'] is seeded['paper'] is None
    assert seeded['bots'][0]['parent']=='legacy-winner'
    assert seeded['bots'][0]['version']==old.version
    assert sum(b['parent']=='g000-b000' for b in seeded['bots'])==3
    assert sum(not b['parent'] for b in seeded['bots'])==1
    assert seeded['seed_source']['mutated_descendants']==3
    assert 'reused' in seeded['seed_source']['holdout_note']
    assert latest_champion(store.home)['run']==source
    # The old holdout cannot silently become fresh because a new run uses the champion.
    assert store.claim_holdout('same-period',source)
    assert not store.claim_holdout('same-period',run)
    assert store.rows('SELECT * FROM metrics WHERE run=?',(run,)).empty


def test_active_losing_survivors_retain_learning_but_cannot_qualify():
    cfg=modern_config()
    bots=[Bot.fresh(f'bot-{i}',44,cfg) for i in range(5)]
    bots[0].learn(np.zeros(44,dtype=np.float32),np.ones(5,dtype=int),-.1,np.zeros(44,dtype=np.float32),False)
    result=dict(alive=True,rules_version=2,selection_eligible=True,net_return=-.01,score=-.02,drawdown=.02)
    records=[result,dict(result,score=-.03),dict(result,selection_eligible=False,score=0),
             dict(result,alive=False,score=1),dict(result,selection_eligible=False,score=2)]
    state=dict(generation=50,bots=[b.pack() for b in bots],results=records)
    next_bots=breed(state,44,cfg)
    assert next_bots[0]['id']==bots[0].id and next_bots[0]['version']==1
    assert all(torch.equal(next_bots[0]['weights'][k],state['bots'][0]['weights'][k]) for k in state['bots'][0]['weights'])
    assert next_bots[0]['optimizer']['state'][0]['step'].item()==1
    assert sum(b['parent']==bots[0].id for b in next_bots)==3
    assert breeding_eligible(result,cfg,50)
    assert not qualifies(result,.15,cfg)
    assert not breeding_eligible(records[2],cfg,50) and not breeding_eligible(records[3],cfg,50)
    assert not breeding_eligible(result,replace(cfg,training_selection='profit_after_warmup'),50)


def test_worker_accumulates_learning_across_losing_generations(tmp_path,monkeypatch):
    import evo.engine as engine
    torch.set_num_threads(1)
    cfg=modern_config()
    dataset=demo_dataset(tmp_path/'data',cfg.symbols)
    run=new_run(tmp_path/'runs',dataset,cfg)
    store=Store(tmp_path/'runs')
    first=store.load(run)['bots'][0]
    def negative_practice(bot,market,start,end,config,learning=False,interrupted=None):
        if learning:
            bot.learn(np.zeros(44,dtype=np.float32),np.full(5,2),-.1,np.ones(44,dtype=np.float32),True)
        book=Book.fresh(config);book.version=bot.version
        prices=np.full(5,100.);liquidity=np.full(5,100000.)
        stamp=market.times[start]
        book.observe(prices,stamp,config)
        book.execute(np.full(5,.2),prices,liquidity,stamp,config)
        book.execute(np.zeros(5),prices,liquidity,stamp+pd.Timedelta(minutes=5),config,liquidate=True)
        book.observe(prices,stamp+pd.Timedelta(minutes=10),config)
        book.analytics.update(bars=1,exposure_sum=.5,active_bars=1,
                              days={str(stamp.date()):dict(start=config.initial_cash,end=book.equity(prices),active=True)})
        return results(book,config),book
    monkeypatch.setattr(engine,'episode',negative_practice)
    ended=run_experiment(store.home,run,minutes=2)
    retained=next(b for b in ended['bots'] if b['id']==first['id'])
    assert retained['version']==2
    assert retained['optimizer']['state'][0]['step'].item()==2
    assert any(not torch.equal(retained['weights'][k],first['weights'][k]) for k in first['weights'])
    event=store.rows("SELECT payload FROM events WHERE run=? AND kind='generation_finished'",(run,)).iloc[0].payload
    import json
    event=json.loads(event)
    assert event['breeding_eligible']==5 and event['retained']==1 and event['descendants']==3
    assert ended['champion'] is None and ended['hall']==[]  # Losses never become winners.


def test_practice_uses_complete_sessions_and_stays_before_validation():
    market=day_market('2023-11-20','2023-12-01')
    cfg=replace(modern_config(),train_steps=3*78,eval_steps=2*78,eval_windows=4)
    plan=practice_plan(market,cfg)
    for windows,count in ((plan['training'],3),(plan['evaluation'],2)):
        for start,end in windows:
            times=market.times[start:end+1].tz_convert('America/New_York')
            assert times[0].hour==9 and times[0].minute==30
            assert times[-1].minute==55 and times[-1].hour in (12,15)
            assert len(set(times.date))==count
            assert market.times[end]<pd.Timestamp(cfg.train_end,tz='UTC')
    assert any(market.times[end].tz_convert('America/New_York').hour==12 for _,end in plan['training'])


def test_friendly_names_are_stable_and_ids_remain_searchable():
    from evo.dashboard_queries import bot_directory
    assert (len(GIVEN),len(MIDDLE),len(FAMILY))==(64,64,128)
    names=[bot_name(f'g{g:03}-b{i:03}') for g in range(100) for i in range(100)]
    assert len(set(names))==len(names)
    assert bot_name('g033-b079')=='Vega Juniper Stone'
    assert 'g033' not in bot_name('g033-b079')
    state={'bots':[dict(id='g033-b079',parent='',birth=33,alive=True)]}
    assert bot_directory(state,pd.DataFrame(),query='g033-b079')[1]==1
    assert bot_directory(state,pd.DataFrame(),query='Vega Juniper')[1]==1


def test_ui_can_start_from_last_champion_without_changing_its_data(tmp_path,monkeypatch):
    import evo.config as settings
    from streamlit.testing.v1 import AppTest
    store,source,state,bot=source_champion(tmp_path)
    before=hashlib.sha256((store.home/source/'checkpoint.pt').read_bytes()).hexdigest()
    monkeypatch.setattr(settings,'DATA',tmp_path)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**kw:None)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    next(s for s in app.selectbox if s.label=='Starting bots').select('Last champion')
    app.select_slider[0].set_value(5)
    app.number_input[0].set_value(1)
    next(b for b in app.button if b.label=='Start experiment').click().run(timeout=30)
    assert not app.exception
    saved=store.load(app.session_state['selected_run'])
    assert saved['seed_source']['bot']=='legacy-winner'
    assert saved['config']['training_selection']=='active_survivors'
    assert saved['config']['session_aligned']
    assert any('Started from champion' in x.value for x in app.info)
    assert hashlib.sha256((store.home/source/'checkpoint.pt').read_bytes()).hexdigest()==before
