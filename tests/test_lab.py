import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from evo.config import Config
from evo.data import ACTION_COLUMNS, Alpaca, demo_dataset, load_dataset, prepare, session_grid
from evo.engine import breed, coverage_qualifies, new_run, qualifies, restore_rng, rng_state, run_experiment
from evo.model import Bot, episode
from evo.paper import fresh_quotes, pending_is_fresh
from evo.sim import Book, allocation, observation, step
from evo.store import Store


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp('dataset')
    demo_dataset(folder, Config().symbols)
    return folder


@pytest.fixture(scope='module')
def market(dataset):
    return load_dataset(dataset, Config().symbols)


def config(**kwargs):
    return Config(population=5, generations=2, train_steps=8, eval_steps=8,
                  eval_windows=1, **kwargs)


def test_cash_fees_liquidity_and_no_shorting():
    cfg = Config()
    book = Book.fresh(cfg)
    book.execute([.2] * 5, np.ones(5) * 100, np.ones(5) * 100000, 't0', cfg)
    assert book.cash >= 0
    assert (book.shares >= 0).all()
    assert book.equity(np.ones(5) * 100) < cfg.initial_cash
    assert sum(t[4] for t in book.trades) > 0
    book.execute([0] * 5, np.ones(5) * 100, np.ones(5) * 100000, 't1', cfg)
    assert np.allclose(book.shares, 0)
    thin = Book.fresh(cfg)
    thin.execute([.2] * 5, np.ones(5) * 100, np.ones(5) * 10, 't', cfg)
    assert np.all(thin.shares <= .1 + 1e-10)


def test_missing_prices_never_fill():
    cfg = Config()
    book = Book.fresh(cfg)
    book.execute([.2] * 5, np.full(5, np.nan), np.ones(5) * 1e5, 't', cfg)
    assert not book.trades
    assert book.cash == 10000


def test_drawdown_uses_peak_and_death_is_permanent():
    cfg = Config()
    book = Book(0, np.array([100., 0, 0, 0, 0]), 10000)
    book.observe(np.ones(5) * 120, 't0', cfg)
    book.observe(np.ones(5) * 102, 't1', cfg)
    assert not book.alive  # Still above starting capital, but 15% below peak.
    n = len(book.trades)
    book.execute(np.zeros(5), np.ones(5) * 102, np.ones(5) * 1e6, 't2', cfg)
    assert len(book.trades) == n


def test_split_dividend_receivable_and_removal():
    book = Book(100, np.array([10., 0, 0, 0, 0]), 1100)
    book.actions(np.array([2,1,1,1,1]), [(0, 1., '2025-01-06')], np.zeros(5, bool), '2025-01-02', 'a')
    assert book.shares[0] == 20
    assert book.cash == 100
    assert book.equity(np.array([49., 1, 1, 1, 1])) == 1100
    book.actions(np.ones(5), [], np.zeros(5, bool), '2025-01-06', 'b')
    assert book.cash == 120
    book.actions(np.array([2,1,1,1,1]), [], np.zeros(5, bool), '2025-01-06', 'a')
    assert book.shares[0] == 20  # Action idempotency.
    book.actions(np.ones(5), [], np.array([True,False,False,False,False]), '2025-01-07', 'c')
    assert book.shares[0] == 0


def test_calendar_holidays_and_early_close():
    assert len(session_grid('2025-12-25', '2025-12-25')) == 0
    assert len(session_grid('2025-11-28', '2025-11-28')) == 42
    assert len(session_grid('2025-11-26', '2025-11-26')) == 78


def test_future_data_cannot_change_earlier_features(dataset, market):
    bars = pd.read_parquet(dataset / 'bars.parquet')
    actions = pd.read_parquet(dataset / 'actions.parquet')
    mask = bars.time >= pd.Timestamp('2025-01-01', tz='UTC')
    bars.loc[mask, ['open','high','low','close']] *= 3
    changed = prepare(bars, actions, Config().symbols, {'synthetic': True})
    before = market.times < pd.Timestamp('2025-01-01', tz='UTC')
    np.testing.assert_array_equal(changed.features[before], market.features[before])
    np.testing.assert_array_equal(changed.meta['scaler_mean'], market.meta['scaler_mean'])


def test_execution_occurs_after_signal_and_uses_actual_open(market):
    cfg = Config(fee_bps=0, slippage_bps=0)
    book = Book.fresh(cfg)
    step(market, 80, book, np.ones(5) * .2, cfg)
    assert book.trades
    assert all(pd.Timestamp(t[0]) > market.times[79] for t in book.trades)
    assert book.trades[0][3] == market.opens[80, 0]


def test_learning_updates_weights_but_evaluation_does_not(market):
    torch.set_num_threads(1)
    cfg = config()
    bot = Bot.fresh('bot', market.features.shape[1] + 7, cfg)
    before = copy.deepcopy(bot.policy.state_dict())
    episode(bot, market, 80, 88, cfg, True)
    assert bot.version == 8
    assert any(not torch.equal(before[k], v) for k,v in bot.policy.state_dict().items())
    frozen = copy.deepcopy(bot.policy.state_dict())
    _, book = episode(bot, market, 80, 88, cfg, False)
    assert all(torch.equal(frozen[k], v) for k,v in bot.policy.state_dict().items())
    assert all(t[5] == bot.version for t in book.trades)


def test_mutation_inheritance_and_new_optimizer(market):
    cfg = config()
    parent = Bot.fresh('parent', market.features.shape[1] + 7, cfg)
    episode(parent, market, 80, 84, cfg, True)
    child = parent.child('child', 1, cfg)
    assert child.parent == parent.id and child.birth == 1
    assert child.version == 0 and not child.optimizer.state
    assert any(not torch.equal(p, c) for p,c in zip(parent.policy.parameters(), child.policy.parameters()))


def test_breed_ratios_and_all_dead_restart(market):
    cfg = config()
    bots = [Bot.fresh(str(i), market.features.shape[1] + 7, cfg).pack() for i in range(5)]
    state = dict(generation=0, bots=bots, results=[{'alive': True, 'score': i} for i in range(5)])
    children = breed(state, market.features.shape[1] + 7, cfg)
    assert len(children) == 5
    assert children[0]['id'] == '4'
    assert sum(bool(b['parent']) for b in children) == 3
    state['results'] = [{'alive':False,'score':-1000}] * 5
    assert all(not b['parent'] for b in breed(state, market.features.shape[1] + 7, cfg))


def test_checkpoint_roundtrip_rng_and_atomicity(tmp_path, market):
    cfg = config()
    bot = Bot.fresh('a', market.features.shape[1] + 7, cfg)
    episode(bot, market, 80, 84, cfg, True)
    saved = {'bot':bot.pack(), 'rng':rng_state()}
    expected = bot.act(observation(market, 80, Book.fresh(cfg)), True)
    store = Store(tmp_path)
    store.save('run', saved)
    restored = store.load('run')
    bot2 = Bot.unpack(restored['bot'], market.features.shape[1] + 7, cfg)
    restore_rng(restored['rng'])
    np.testing.assert_array_equal(expected, bot2.act(observation(market,80,Book.fresh(cfg)),True))
    assert bot2.version == bot.version
    assert not list((tmp_path / 'run').glob('*.tmp'))


def test_holdout_ledger_blocks_reuse(tmp_path):
    store = Store(tmp_path)
    assert store.claim_holdout('2025-real', 'a')
    assert store.claim_holdout('2025-real', 'a')
    assert not store.claim_holdout('2025-real', 'b')
    assert store.claim_holdout('2025-demo', 'b')


def test_trade_audit_idempotency_and_per_fill_versions(tmp_path):
    store = Store(tmp_path)
    trades = [('2025-01-01', 'AAPL', 1, 100, 1, 3), ('2025-01-02', 'AAPL', -1, 101, 1, 4)]
    for _ in range(2):
        store.log_trace('r','b','training',99,trades,[])
    records = store.rows('SELECT * FROM trades')
    assert len(records) == 2
    assert records.version.tolist() == [3,4]


def test_stale_quotes_and_expired_orders_cannot_fill():
    now = pd.Timestamp('2025-01-02T15:00:00Z')
    payload = {'quotes': {'AAPL': {'t':now.isoformat(),'bp':100,'ap':101,'bs':10,'as':20}}}
    assert fresh_quotes(payload, ['AAPL'], now)[0][0] == 100.5
    with pytest.raises(ValueError, match='stale'):
        fresh_quotes(payload, ['AAPL'], now + pd.Timedelta(minutes=2))
    with pytest.raises(ValueError, match='no live'):
        fresh_quotes(payload, ['MSFT'], now)
    assert not pending_is_fresh({'created':str(now)}, now + pd.Timedelta(minutes=5))
    assert pending_is_fresh({'created':str(now)}, now + pd.Timedelta(seconds=5))


def test_data_fingerprint_detects_modification(tmp_path, dataset):
    import shutil
    shutil.copytree(dataset, tmp_path / 'copy')
    bars = pd.read_parquet(tmp_path / 'copy' / 'bars.parquet')
    bars.loc[0, 'close'] += .01
    bars.to_parquet(tmp_path / 'copy' / 'bars.parquet', index=False)
    with pytest.raises(ValueError, match='fingerprint'):
        load_dataset(tmp_path / 'copy', Config().symbols)


def test_pause_resume_is_reproducible(tmp_path, dataset):
    cfg = config()
    a = new_run(tmp_path / 'a', dataset, cfg)
    paused = run_experiment(tmp_path / 'a', a, minutes=0)
    assert paused['status'] == 'paused' and paused['cursor'] == 0
    resumed = run_experiment(tmp_path / 'a', a, minutes=2)
    b = new_run(tmp_path / 'b', dataset, cfg)
    direct = run_experiment(tmp_path / 'b', b, minutes=2)
    assert resumed['status'] == direct['status'] == 'complete'
    for p,q in zip(resumed['bots'], direct['bots']):
        assert all(torch.equal(p['weights'][k], q['weights'][k]) for k in p['weights'])
    assert resumed['results'] == direct['results']
    assert resumed['phase'] == 'finished'
    assert resumed['paper'] is None


def test_promotion_gate():
    assert qualifies({'alive':True,'net_return':.01,'drawdown':.14},.15)
    assert not qualifies({'alive':True,'net_return':-.01,'drawdown':.01},.15)
    assert not qualifies({'alive':True,'net_return':.5,'drawdown':.15},.15)


def test_credentials_never_enter_saved_state(tmp_path, dataset, monkeypatch):
    monkeypatch.setenv('APCA_API_KEY_ID','do-not-store-key')
    monkeypatch.setenv('APCA_API_SECRET_KEY','do-not-store-secret')
    run = new_run(tmp_path, dataset, config())
    assert b'do-not-store' not in (tmp_path / run / 'checkpoint.pt').read_bytes()


def test_alpaca_pagination_and_raw_feed(monkeypatch):
    monkeypatch.setenv('APCA_API_KEY_ID','test')
    monkeypatch.setenv('APCA_API_SECRET_KEY','test')
    client = Alpaca()
    calls = []
    def get(path, params):
        calls.append(dict(params))
        bar = {'t':'2025-01-02T15:00:00Z','o':1,'h':1,'l':1,'c':1,'v':10}
        return {'bars':{'AAPL':[bar]}, 'next_page_token':'next' if len(calls)==1 else None}
    monkeypatch.setattr(client, 'get', get)
    monkeypatch.setattr('evo.data.time.sleep',lambda _: None)
    assert len(client.bars(['AAPL'],'a','b')) == 2
    assert calls[0]['feed'] == 'iex' and calls[0]['adjustment'] == 'raw'
    assert calls[1]['page_token'] == 'next'


def test_short_history_cannot_pass_real_data_coverage(market):
    assert not coverage_qualifies(market, Config())


def test_corporate_action_features_and_no_repurchase_after_removal():
    times = session_grid('2023-01-03', '2023-01-04')
    prices = np.where(np.arange(len(times)) < 78, 100., 50.)
    bars = pd.DataFrame({'time':times,'symbol':'AAPL','open':prices,'high':prices,
                         'low':prices,'close':prices,'volume':10000.})
    actions = pd.DataFrame([['split','AAPL','forward_split','2023-01-04',2.,0.,'']],columns=ACTION_COLUMNS)
    m = prepare(bars,actions,['AAPL'],{})
    assert m.marks[78,0] == 50
    assert abs(m.features[78,0]) < 1e-6  # Split does not appear as a 50% economic loss.
    actions.loc[0] = ['gone','AAPL','worthless_removal','2023-01-04',1.,0.,'']
    m = prepare(bars,actions,['AAPL'],{})
    assert np.isnan(m.opens[78:]).all()
    assert (m.marks[78:] == 0).all()


def test_unsupported_actions_block_data():
    times = session_grid('2023-01-03','2023-01-04')
    bars = pd.DataFrame({'time':times,'symbol':'AAPL','open':100.,'high':100.,
                         'low':100.,'close':100.,'volume':10000.})
    actions = pd.DataFrame([['spin','AAPL','spin_off','2023-01-04',1.,0.,'']],columns=ACTION_COLUMNS)
    with pytest.raises(ValueError,match='Unsupported corporate action'):
        prepare(bars,actions,['AAPL'],{})


def test_emergency_before_worker_initialization_is_not_lost(tmp_path,dataset):
    run = new_run(tmp_path,dataset,config())
    store = Store(tmp_path)
    store.request(run,'emergency')
    state = run_experiment(tmp_path,run,minutes=2)
    assert state['status'] == 'stopped' and state['phase'] == 'finished'
    assert state['cursor'] == 0


def test_interrupted_bot_rolls_back_learning_and_rng(tmp_path,dataset,monkeypatch):
    import evo.engine as engine
    original = engine.episode
    interrupted = [False]
    def abort_once(bot,market,start,end,cfg,learning=False,stop=None):
        if learning and not interrupted[0]:
            interrupted[0] = True
            _,book = original(bot,market,start,start+2,cfg,True)
            return None,book
        return original(bot,market,start,end,cfg,learning,stop)
    cfg = config()
    run = new_run(tmp_path/'resume',dataset,cfg)
    monkeypatch.setattr(engine,'episode',abort_once)
    state = run_experiment(tmp_path/'resume',run,minutes=2)
    assert state['status'] == 'paused' and state['cursor'] == 0
    resumed = run_experiment(tmp_path/'resume',run,minutes=2)
    monkeypatch.setattr(engine,'episode',original)
    run2 = new_run(tmp_path/'direct',dataset,cfg)
    direct = run_experiment(tmp_path/'direct',run2,minutes=2)
    assert resumed['results'] == direct['results']
    assert all(torch.equal(resumed['bots'][0]['weights'][k],direct['bots'][0]['weights'][k])
               for k in resumed['bots'][0]['weights'])


def test_only_one_champion_gets_final_test_and_reuse_blocks_promotion(tmp_path,dataset,market,monkeypatch):
    import evo.engine as engine
    real = copy.deepcopy(market)
    real.meta['synthetic'] = False
    monkeypatch.setattr(engine,'load_dataset',lambda *a,**kw: real)
    monkeypatch.setattr(engine,'coverage_qualifies',lambda *a: True)
    tests = []
    def profitable(bot,m,start,end,cfg,learning=False,interrupted=None):
        if m.times[start].year == 2025:
            tests.append(bot.id)
        book = Book.fresh(cfg)
        book.observe(m.marks[start],m.times[start],cfg)
        return dict(alive=True,net_return=.1,drawdown=.01,score=.08),book
    monkeypatch.setattr(engine,'episode',profitable)
    one = new_run(tmp_path,dataset,config())
    state = run_experiment(tmp_path,one,minutes=2)
    assert state['status'] == 'paper_ready'
    assert len(tests) == 1
    two = new_run(tmp_path,dataset,config())
    reused = run_experiment(tmp_path,two,minutes=2)
    assert reused['status'] == 'complete'
    assert 'already used' in reused['promotion_reason']


def test_app_loads_and_can_configure_a_small_run(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import evo.config as settings
    monkeypatch.setattr(settings,'DATA',tmp_path)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**kw: None)
    demo_dataset(tmp_path/'data'/'demo',Config().symbols)
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    app.select_slider[0].set_value(5)
    app.number_input[0].set_value(1)
    app.run(timeout=30)
    next(b for b in app.button if b.label=='Start experiment').click().run(timeout=30)
    assert not app.exception
    run = app.session_state['selected_run']
    saved = Store(tmp_path/'runs').load(run)
    assert saved['config']['population'] == 5 and saved['config']['generations'] == 1


def test_fast_learning_batches_keep_all_market_steps(market):
    torch.set_num_threads(1)
    cfg=config(learn_every=8)
    bot=Bot.fresh('fast',42,cfg)
    result,book=episode(bot,market,80,99,cfg,True)
    assert len(book.curve)==1+19*2
    assert bot.version==3  # Two full batches plus final partial batch.
    before=copy.deepcopy(bot.policy.state_dict())
    episode(bot,market,80,99,cfg,False)
    assert all(torch.equal(before[k],v) for k,v in bot.policy.state_dict().items())


def test_exact_portfolio_snapshot_includes_cash_holdings_and_dividends(tmp_path):
    from evo.views import portfolio,holdings_frame
    cfg=Config()
    book=Book.fresh(cfg)
    prices=np.ones(5)*100
    book.execute([.2]*5,prices,np.ones(5)*100000,'2025-01-02T15:00:00Z',cfg)
    book.observe(prices,'2025-01-02T15:05:00Z',cfg)
    store=Store(tmp_path)
    store.log_trace('r','b','training:g0',0,book.trades,book.curve,book=book)
    snapshot=portfolio(store,{'run':'r'},'b','training:g0',pd.DataFrame(),pd.DataFrame())
    assert snapshot['cash']==book.cash
    np.testing.assert_allclose(snapshot['shares'],book.shares)
    frame=holdings_frame(snapshot,cfg.symbols)
    assert frame['Value ($)'].sum()+snapshot['cash']==pytest.approx(snapshot['equity'])


def test_candles_have_zoom_volume_and_range_slider(dataset):
    from evo.views import stock_chart
    bars=pd.read_parquet(dataset/'bars.parquet')
    bars=bars[bars.symbol=='AAPL'].head(78)
    trades=pd.DataFrame(columns=['time','symbol','shares','price'])
    fig=stock_chart(bars,trades,'AAPL','15min','test-chart')
    assert fig.layout.dragmode=='zoom'
    assert fig.layout.uirevision=='test-chart'
    assert fig.layout.xaxis2.rangeslider.visible
    assert fig.data[0].type=='candlestick' and len(fig.data[0].open)==26
    assert fig.data[1].type=='bar'


def test_bot_personal_page_loads_portfolio(tmp_path,dataset,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import evo.config as settings
    monkeypatch.setattr(settings,'DATA',tmp_path)
    run=new_run(tmp_path/'runs',dataset,config())
    state=run_experiment(tmp_path/'runs',run,minutes=2)
    bot=state['bots'][0]['id']
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
    app.query_params['run']=run
    app.query_params['bot']=bot
    app.run(timeout=30)
    assert not app.exception
    assert app.radio[0].value=='Bot profile'
    assert any(h.value=='What this bot owns' for h in app.get('subheader')) or any('What this bot owns' in h.value for h in app.markdown)


def test_follow_best_compares_latest_generation_survivors():
    from evo.views import best_bot
    m=pd.DataFrame([
        dict(bot='older',generation=0,alive=1,net_return=.9,score=.8,drawdown=.01),
        dict(bot='dead',generation=1,alive=0,net_return=.5,score=.4,drawdown=.15),
        dict(bot='profit',generation=1,alive=1,net_return=.2,score=.05,drawdown=.08),
        dict(bot='steady',generation=1,alive=1,net_return=.1,score=.09,drawdown=.01)])
    original=m.copy(deep=True)
    assert best_bot(m,{})['bot']=='profit'
    assert best_bot(m,{},'Risk-adjusted score')['bot']=='steady'
    assert best_bot(m,{})['mode']=='evaluation:g1'
    pd.testing.assert_frame_equal(m,original)
    m.loc[m.generation==1,'alive']=0
    assert best_bot(m,{})['bot']=='older'
    assert best_bot(m,{})['using_previous']
    m['alive']=0
    assert best_bot(m,{}) is None
    assert best_bot(pd.DataFrame(),{}) is None


def test_follow_best_switches_as_new_results_arrive():
    from evo.views import best_bot
    first=pd.DataFrame([dict(bot='first',generation=2,alive=1,net_return=.1,score=.1,drawdown=.01)])
    assert best_bot(first,{})['bot']=='first'
    second=pd.concat([first,pd.DataFrame([dict(bot='second',generation=2,alive=1,net_return=.2,score=.2,drawdown=.02)])])
    assert best_bot(second,{})['bot']=='second'
    book=Book.fresh(Config())
    state={'paper':{'book':book,'bot':{'id':'champion'}}}
    assert best_bot(second,state)==dict(bot='champion',mode='paper',reason='Live paper portfolio')
    book.alive=False
    assert best_bot(second,state) is None


@pytest.mark.parametrize('losing',[False,True])
def test_follow_best_ui_does_not_write_experiment_data(tmp_path,dataset,monkeypatch,losing):
    import hashlib
    from streamlit.testing.v1 import AppTest
    import evo.config as settings
    monkeypatch.setattr(settings,'DATA',tmp_path)
    run=new_run(tmp_path/'runs',dataset,config())
    state=run_experiment(tmp_path/'runs',run,minutes=2)
    store=Store(tmp_path/'runs')
    if losing:
        with store.connect() as db:
            db.execute('UPDATE metrics SET net_return=-abs(net_return)-0.01 WHERE run=?',(run,))
    def fingerprint():
        with store.connect() as db:
            digest=hashlib.sha256('\n'.join(db.iterdump()).encode()).hexdigest()
        return digest,hashlib.sha256((tmp_path/'runs'/run/'checkpoint.pt').read_bytes()).hexdigest(),hashlib.sha256((dataset/'bars.parquet').read_bytes()).hexdigest()
    before=fingerprint()
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
    app.query_params['run']=run
    app.run(timeout=30)
    next(t for t in app.toggle if t.label=='Follow best bot').set_value(True).run(timeout=30)
    assert not app.exception
    assert any('Following ' in s.value for s in list(app.success)+list(app.warning))
    if losing:
        assert any('Showing the best eligible survivor with its current loss' in s.value for s in app.warning)
        assert any(t.label=='Following trader' for t in app.text_input)
    assert fingerprint()==before


def test_dashboard_choices_do_not_scan_raw_archive():
    from evo.dashboard_queries import portfolio_choices
    class IndexedStore:
        def rows(self,sql,args):
            assert 'FROM portfolios' in sql
            assert 'equity' not in sql and 'trades' not in sql
            return pd.DataFrame([{'bot':'a','mode':'evaluation:g2'}])
    result=portfolio_choices(IndexedStore(),{'run':'r'},pd.DataFrame())
    assert result.iloc[0].bot=='a'
    class LegacyStore:
        def rows(self,sql,args):
            assert 'FROM portfolios' in sql
            return pd.DataFrame(columns=['bot','mode'])
    metrics=pd.DataFrame([dict(bot='a',generation=2,alive=1),dict(bot='b',generation=2,alive=0)])
    legacy=portfolio_choices(LegacyStore(),{'run':'r'},metrics)
    assert ('a','evaluation:g2') in set(legacy.itertuples(index=False,name=None))
    assert ('b','evaluation:g2') not in set(legacy.itertuples(index=False,name=None))


def test_bot_directory_limits_rendered_cards():
    from evo.dashboard_queries import bot_directory
    state={'bots':[dict(id=f'bot-{i}',parent='',birth=0,alive=True) for i in range(10000)]}
    rows,total,pages=bot_directory(state,pd.DataFrame(),page=2)
    assert len(rows)==25 and total==10000 and pages==400
    assert rows[0][0]=='bot-25'
    rows,total,pages=bot_directory(state,pd.DataFrame(),query='does not exist')
    assert rows==[] and total==0 and pages==1


def test_day_trader_v2_training_and_score_reports(tmp_path,dataset,monkeypatch):
    from evo.score_ui import load_metrics
    from streamlit.testing.v1 import AppTest
    import evo.config as settings
    cfg=Config(rules_version=2,population=5,generations=2,train_steps=78,eval_steps=78,eval_windows=1,learn_every=8)
    run=new_run(tmp_path/'runs',dataset,cfg)
    state=run_experiment(tmp_path/'runs',run,minutes=2)
    assert state['status']=='complete'
    assert state['bots'][0]['weights']['body.0.weight'].shape[1]==44
    store=Store(tmp_path/'runs')
    reports=store.rows('SELECT payload FROM diagnostics WHERE run=?',(run,))
    assert len(reports)==10
    r=json.loads(reports.iloc[0].payload)
    assert r['rules_version']==2 and 'mean_exposure' in r and 'score_parts' in r
    metrics=load_metrics(store,run)
    assert len(metrics)==10 and 'selection_eligible' in metrics
    monkeypatch.setattr(settings,'DATA',tmp_path)
    bot=metrics.iloc[0].bot
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
    app.query_params['run']=run;app.query_params['bot']=bot
    app.run(timeout=30)
    assert not app.exception
    labels=[m.label for m in app.metric]
    assert 'Net profit' in labels and 'Average exposure' in labels and 'Completed round trips' in labels
    assert any('Positive-profit gate' in m.value for m in app.markdown)


def test_completed_experiment_shows_champion_and_saved_survivors(tmp_path,dataset,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import evo.config as settings
    monkeypatch.setattr(settings,'DATA',tmp_path)
    run=new_run(tmp_path/'runs',dataset,config())
    state=run_experiment(tmp_path/'runs',run,minutes=2)
    store=Store(tmp_path/'runs')
    winner=state['bots'][0]
    state.update(status='complete',phase='finished',champion=winner,
                 promotion_reason='Passed validation and untouched final test.',
                 test={'alive':True,'net_return':.01,'drawdown':.01})
    store.save(run,state)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
    app.query_params['run']=run
    app.run(timeout=30)
    next(r for r in app.radio if r.label=='Workspace').set_value('Bots').run(timeout=30)
    assert not app.exception
    assert any('Champion:' in s.value for s in app.success)
    assert any(m.value=='**Champion**' for m in app.markdown)
    assert not any(m.value=='**Training**' for m in app.markdown)
    assert any('training is finished' in c.value for c in app.caption)
    next(r for r in app.radio if r.label=='Workspace').set_value('Reports').run(timeout=30)
    assert not app.exception
    assert any('survived_risk_checks' in j.value for j in app.json)
