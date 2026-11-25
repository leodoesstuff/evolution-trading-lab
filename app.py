"""Run with: streamlit run app.py --server.address 127.0.0.1"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
# Upgrade Python's module cache without restarting the server or losing in-memory keys.
import importlib
import evo
if not getattr(evo,'_day_rules_v2_loaded',False):
    cached_config=sys.modules.get('evo.config')
    if cached_config is not None and not hasattr(cached_config.Config,'rules_version'):
        for module in ('evo.config','evo.day_rules','evo.sim','evo.model','evo.store','evo.engine','evo.paper','evo.arena','evo.views','evo.leaders','evo.dashboard_queries','evo.live_ui'):
            if module in sys.modules:importlib.reload(sys.modules[module])
    evo._day_rules_v2_loaded=True
if not getattr(evo,'_decision_cadence_loaded',False):
    cached_config=sys.modules.get('evo.config')
    if cached_config is not None and not hasattr(cached_config.Config,'decision_bars'):
        for module in ('evo.config','evo.sim','evo.model','evo.engine','evo.paper','evo.arena','evo.views','evo.live_ui','evo.training_scope'):
            if module in sys.modules:importlib.reload(sys.modules[module])
    evo._decision_cadence_loaded=True
if not getattr(evo,'_active_survivor_training_loaded',False):
    cached_config=sys.modules.get('evo.config')
    if cached_config is not None and not hasattr(cached_config.Config,'training_selection'):
        for module in ('evo.config','evo.day_rules','evo.sim','evo.model','evo.engine','evo.paper','evo.arena','evo.views','evo.training_scope','evo.score_ui','evo.live_ui'):
            if module in sys.modules:importlib.reload(sys.modules[module])
    evo._active_survivor_training_loaded=True
from evo.config import Config, DATA, ROOT
from evo.data import Alpaca, demo_dataset
from evo.engine import new_run
from evo.store import Store
from evo.views import session_name, portfolio, holdings_frame, stock_chart, balance_chart
from evo.names import bot_name
import evo.leaders as leadership
if getattr(leadership,'LEADER_VIEW_VERSION',0)<3:
    importlib.reload(leadership)
from evo.leaders import best_bot, leader_counts, leader_wait_message
from evo.dashboard_queries import portfolio_choices, bot_directory
from evo.score_ui import load_metrics, score_report
from evo.status_labels import qualified_champion, bot_status
from evo.training_scope import COVERAGE, coverage_config, continue_with_coverage
from evo.champion_seed import latest_champion, seed_from_champion

st.set_page_config(page_title='Evolution • Trading Lab', page_icon='🧬', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container {padding-top:2rem;max-width:1600px} [data-testid=stSidebar] {min-width:285px!important}
.stApp {background:#0b1019;color:#e5edf9} h1 {letter-spacing:-1.8px!important}
[data-testid="stSidebar"] {background:#111827;border-right:1px solid #253044}
[data-testid="stMetric"] {background:#131d2c;border:1px solid #26344b;padding:16px;border-radius:14px}
.eyebrow {color:#71e6bf;font:600 12px monospace;letter-spacing:3px;margin-bottom:8px}
.muted {color:#91a0b7} .pill {background:#183d35;color:#81f5c8;padding:6px 12px;border-radius:30px;font-size:12px}
button {border-radius:9px!important} [data-testid="stHeader"] {background:transparent}
</style>''', unsafe_allow_html=True)
HOME = DATA / 'runs'
DATASETS = DATA / 'data'
store = Store(HOME)


def worker_alive(run):
    lock_path = HOME / run / 'worker.lock'
    if not lock_path.exists():
        return False
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def launch(run, minutes):
    if worker_alive(run):
        return
    store.request(run, 'run')
    with (HOME / run / 'worker.log').open('a') as log:
        subprocess.Popen([sys.executable, '-m', 'evo.worker', '--home', str(HOME), '--run', run,
                          '--minutes', str(minutes)], cwd=ROOT, stdout=log, stderr=log,
                         start_new_session=True)


with st.sidebar:
    st.markdown('### 🧬 Evolution')
    st.caption('Training settings')
    with st.expander('Data connection'):
        st.markdown('**Market data**')
        datasets = sorted([p for p in DATASETS.glob('*') if (p / 'metadata.json').exists()]) if DATASETS.exists() else []
        if st.button('Create demo data', width='stretch'):
            with st.spinner('Generating clearly labeled synthetic prices…'):
                demo_dataset(DATASETS / 'demo', Config().symbols)
            st.rerun()
        connected = bool(os.environ.get('APCA_API_KEY_ID') and os.environ.get('APCA_API_SECRET_KEY'))
        st.caption('Alpaca credentials: ' + ('available ✓' if connected else 'not connected'))
        with st.expander('Connect a free Alpaca account'):
            st.markdown('[Open Alpaca](https://app.alpaca.markets/signup) and create paper API keys. '
                        'This app only reads market data.')
            key = st.text_input('API key ID', type='password', key='alpaca_key')
            secret = st.text_input('API secret', type='password', key='alpaca_secret')
            if st.button('Connect for this app session'):
                if key and secret:
                    os.environ['APCA_API_KEY_ID'] = key
                    os.environ['APCA_API_SECRET_KEY'] = secret
                    st.success('Connected in memory. Keys are not written to disk.')
                else:
                    st.error('Enter both fields.')
            st.caption('Reconnect after restarting the app, or use environment variables.')
        if st.button('Download 2020–2025 IEX history', disabled=not bool(os.environ.get('APCA_API_KEY_ID')),
                     width='stretch'):
            try:
                with st.status('Downloading historical market data…', expanded=True) as status:
                    Alpaca().download(DATASETS / 'alpaca-iex', Config().symbols, st.write)
                    status.update(label='Data downloaded and validated', state='complete')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.divider()
    dataset = st.selectbox('Dataset', datasets, format_func=lambda p: 'Real stocks · IEX' if p.name=='alpaca-iex' else 'Demo · generated prices', index=0 if datasets else None)
    population = st.select_slider('Bots per generation', [5,10,20,50,100,200], value=100)
    generations = st.number_input('Generations',1,1000,100)
    minutes = st.select_slider('Session budget · minutes',[15,30,60,120,180],value=15)
    speed = st.selectbox('Learning speed',['Fast','Standard','Turbo'],help='Fast batches 8 learning updates. Turbo batches 32. All trades and risk checks still run. Learning behavior changes, not the market clock.')
    batch = {'Standard':1,'Fast':8,'Turbo':32}[speed]
    synthetic = dataset is not None and json.loads((dataset/'metadata.json').read_text()).get('synthetic',False)
    coverage = st.selectbox('History per generation', ['Quick'] if synthetic else list(COVERAGE),
                            index=0 if synthetic else 1,
                            help='Longer history exposes bots to more market conditions. It takes more computing time per generation.')
    learning_days, evaluation_days, windows = COVERAGE[coverage]
    st.caption(f'Each bot learns on about {learning_days} trading days, then is tested on {windows} periods of about {evaluation_days} days spread across 2020–2023. 2024 and 2025 stay separate.')
    if synthetic: st.caption('The short synthetic demo supports Quick coverage. Choose real IEX data for Broad or Deep.')
    previous_champion=latest_champion(HOME)
    starting_bots=st.selectbox('Starting bots',['Fresh bots','Last champion'] if previous_champion else ['Fresh bots'])
    if starting_bots=='Last champion':
        st.caption(f'Seed: {bot_name(previous_champion["bot"])} · one transferred copy, mutated descendants, and fresh bots. Prior validation/test years are reused research periods.')
    st.caption('History runs faster than real time. There is no playback delay. Live paper trading follows the real market clock.')
    with st.expander('Advanced settings'):
        seed = st.number_input('Experiment seed',0,999999,7)
        decision_minutes=st.selectbox('Strategy decision interval · minutes',[5,15,30,60],index=3,
                                      help='Hold positions between strategy decisions. Loss stops and closing exits still run on every five-minute historical bar and every fresh live quote.')
        st.caption('$10,000 per bot · day-trader v2 · daily loss stop 3% · 15% drawdown limit · maximum 20% per stock')
    if st.button('Start experiment',type='primary',width='stretch',disabled=dataset is None):
        try:
            cfg=Config(rules_version=2,population=int(population),generations=int(generations),minutes=minutes,seed=int(seed),learn_every=batch,
                       decision_bars=decision_minutes//5,stop_on_extinction=True,
                       training_selection='active_survivors',session_aligned=True)
            cfg=coverage_config(cfg,coverage)
            run=(seed_from_champion(HOME,previous_champion['run'],cfg,dataset.resolve())
                 if starting_bots=='Last champion' else new_run(HOME,dataset.resolve(),cfg))
            st.session_state['selected_run']=run
            st.query_params.clear()
            launch(run,minutes)
            st.rerun()
        except Exception as exc: st.error(str(exc))

st.markdown('<div class="eyebrow">EVOLUTION / STOCK LAB</div>',unsafe_allow_html=True)
st.title('Your AI trading desk')
st.caption('Follow the market. Meet your bots. Watch them learn.')
runs=sorted([p.name for p in HOME.iterdir() if p.is_dir() and (p/'checkpoint.pt').exists()],reverse=True)
if not runs:
    st.info('Open the sidebar to create demo data and start your first experiment.')
    st.stop()
selected=st.query_params.get('run',st.session_state.get('selected_run',runs[0]))
run=st.selectbox('Experiment',runs,index=runs.index(selected) if selected in runs else 0)
st.session_state['selected_run']=run
page=st.radio('Workspace',['Market','Bots','Bot profile','Reports','Live room'],horizontal=True,
              index=2 if st.query_params.get('bot') else 0)
if page == 'Live room':
    from evo.live_ui import render_live_room
    render_live_room(store.load(run), DATA / 'practice')
    st.stop()
follow = st.toggle('Follow best bot', value=False,
                   help='Follow the strongest active survivor, including when it is losing money. Keep the latest saved survivor visible while new tests finish. This view does not promote it. Updates every 10 seconds.')
ranking = st.selectbox('Best means', ['Highest return','Risk-adjusted score'], disabled=not follow,
                       help='Return compares the same practice windows after costs. Risk-adjusted score also penalizes losses from the peak and inconsistent results.') if follow else 'Highest return'
auto=st.toggle('Live updates',value=False,help='Refresh every 10 seconds. Turn off while inspecting or zooming charts.')
if st.button('Refresh results'): st.rerun()


@st.fragment(run_every='10s' if auto or follow else None)
def floor(run):
    state=store.load(run)
    cfg=Config(**state['config'])
    live=worker_alive(run)
    terminal=state['status'] in ('stopped','complete','paper_eliminated')
    labels={'training':'Bots are learning from historical prices','validation':'Checking the strongest bots on 2024',
            'test':'Final exam on unseen 2025 prices','paper':'Practicing on live prices','finished':'Experiment finished'}
    st.info(labels.get(state['phase'],state['phase'])+' · '+state['status'].replace('_',' ').capitalize())
    winner=qualified_champion(state)
    if winner:
        st.success(f'Champion: {bot_name(winner)}')
        st.caption('Historical training and qualification are finished. Other survivors remain saved; they are not still training.')
    elif state['phase']=='finished':
        st.info('This experiment ended without a qualified champion.')
        if state.get('promotion_reason'):st.caption(state['promotion_reason'])
    if cfg.rules_version>=2:
        st.caption('Day-trader v2 · exit window starts 10 minutes before close · active survivors within risk limits keep learning · positive profit required for qualification' if cfg.training_selection=='active_survivors' else 'Day-trader v2 · legacy selection: activity and profit required to reproduce after warmup')
    else:
        st.caption('Legacy overnight rules. This saved run is preserved; start a new experiment for day-trader v2.')
    if state['meta'].get('synthetic'): st.warning('Demo prices — these results are for exploring the app, not real stock performance.')
    if state.get('last_error'): st.error(state['last_error'])
    st.caption(f'History coverage: about {cfg.train_steps/78:g} learning days per bot · {cfg.eval_windows} evaluation periods × about {cfg.eval_steps/78:g} days. Days are regular-session equivalents; early closes can change the count.')
    st.caption(f'Strategy decisions every {cfg.decision_bars*5} minutes · risk checks on every historical bar and fresh live quote.')
    if cfg.session_aligned:st.caption('Practice periods use complete trading sessions, including early closes.')
    if state.get('seed_source'):
        seed=state['seed_source']
        st.info(f'Started from champion {bot_name(seed["bot"])} · {seed["exact_copies"]} transferred copy · {seed["mutated_descendants"]} mutated descendants · {seed["fresh_bots"]} fresh bots')
        st.caption(seed['holdout_note'])
    if state.get('continuation'):
        st.caption(f'Continued from {state["continuation"]["source_run"]}. Learned bots were copied; the source results remain saved. Scores here start fresh using the longer periods.')
    if state['phase']=='training' and not state['meta'].get('synthetic') and cfg.train_steps < 120*78:
        with st.expander('Give these bots more historical practice'):
            options=[name for name,scope in COVERAGE.items() if scope[0]*78>cfg.train_steps]
            more=st.selectbox('Longer history',options,key=f'coverage_{run}')
            days,exam_days,exams=COVERAGE[more]
            st.caption(f'Copy the learned bots into a separate experiment: about {days} learning days and {exams} × {exam_days}-day evaluations, with strategy decisions every {max(cfg.decision_bars,12)*5} minutes. Their weights are retained and all bots are reevaluated. The original experiment is preserved.')
            if st.button('Continue bots with more history',disabled=live):
                try:
                    continued=continue_with_coverage(HOME,run,more,decision_bars=max(cfg.decision_bars,12))
                    st.session_state['selected_run']=continued
                    st.query_params.clear()
                    launch(continued,minutes)
                    st.rerun()
                except Exception as exc: st.error(str(exc))
            if live: st.caption('Pause this experiment before creating a continuation.')
    if state['phase']=='training' and cfg.decision_bars<12 and not state['meta'].get('synthetic'):
        with st.expander('Reduce trading-cost churn'):
            st.caption('Copy these learned policies into a separate run with hourly strategy decisions. It keeps the same history periods, costs, activity requirements, and risk limits. Scores are reevaluated and the source stays saved.')
            matching=next((name for name,scope in COVERAGE.items() if (scope[0]*78,scope[1]*78,scope[2])==(cfg.train_steps,cfg.eval_steps,cfg.eval_windows)),None)
            if matching and st.button('Continue with hourly decisions',disabled=live):
                try:
                    continued=continue_with_coverage(HOME,run,matching,decision_bars=12)
                    st.session_state['selected_run']=continued
                    st.query_params.clear()
                    launch(continued,minutes)
                    st.rerun()
                except Exception as exc:st.error(str(exc))
            if live:st.caption('Pause this experiment before creating a continuation.')
    controls=st.columns(4)
    if controls[0].button('Resume',disabled=live or terminal): launch(run,minutes);st.rerun()
    if controls[1].button('Pause',disabled=not live): store.request(run,'pause');st.toast('Pausing safely…')
    if controls[2].button('Stop',disabled=terminal):
        store.request(run,'stop')
        if not live: state.update(status='stopped',phase='finished');store.save(run,state)
        st.rerun()
    if controls[3].button('Emergency stop',disabled=terminal):
        store.request(run,'emergency')
        if not live: state.update(status='stopped',phase='finished');store.save(run,state)
        st.rerun()
    metrics=load_metrics(store,run)
    latest=metrics[metrics.generation==metrics.generation.max()] if not metrics.empty else metrics
    leader = best_bot(metrics, state, ranking) if follow else None
    if follow:
        counts=leader_counts(metrics)
        if counts['generation'] is not None:
            st.caption(f'Generation {counts["generation"]+1}: {counts["completed"]} completed evaluations · {counts["survived"]} survived · {counts["eligible"]} meet activity and risk rules · {counts["profitable"]} of those profitable')
        if leader:
            banner=st.warning if leader['mode']!='paper' and not leader['profitable'] else st.success
            banner(f'Following {bot_name(leader["bot"])} · {leader["reason"]}')
            if leader['mode'] != 'paper':
                if leader.get('using_previous'):
                    st.info(f'Showing a saved survivor from generation {leader["generation"]+1}. The current generation has no completed survivor meeting activity and risk rules yet; this view switches when one becomes available.' if state['phase']=='training' else f'Showing a saved survivor from generation {leader["generation"]+1}.')
                st.caption(f'Average practice-test return: {leader["net_return"]:+.2%}. Compared over the same historical windows after costs. The leader changes as new tests finish; unfinished lives are not streamed.')
                if not leader['profitable']:
                    st.warning('This saved evaluation is losing money. Showing the best eligible survivor with its current loss. Active bots within risk limits can keep learning, but positive profit is still required for champion qualification.' if cfg.training_selection=='active_survivors' else 'This saved evaluation is losing money. Showing the best eligible survivor with its current loss. It has not met the positive-profit requirement for reproduction after warmup or qualification.')
        else:
            st.info(leader_wait_message(metrics,state,live))
        st.caption('Refreshing every 10 seconds. Viewing a leader does not promote it or change its training. Turn off Follow best bot to inspect another trader or zoom a chart.')
        if leader and not live and state['phase']=='training':
            st.caption('Training is paused. The saved leader remains visible; new results require training to resume.')
    cells=st.columns(4)
    cells[0].metric('Generation',f'{state["generation"]+1} / {cfg.generations}')
    cells[1].metric('Bots tested this round',f'{state["cursor"]} / {cfg.population}')
    cells[2].metric('Survived latest test',str(int(latest.alive.sum())) if not latest.empty else '—')
    bars=state.get('simulated_bars',0)
    cells[3].metric('Historical bars / second',f'{bars/max(state["elapsed"],.001):,.0f}' if bars else '—')
    if state['phase']=='training':
        st.progress((state['generation']*cfg.population+state['cursor'])/(cfg.generations*cfg.population))
        with st.expander('Training speed explained'):
            st.write('Each bar represents five market minutes. Historical simulations have no waiting between bars. Throughput counts work across all bots, including their practice tests.')
            st.caption(f'Current learning batch: {cfg.learn_every} bar(s). Pause to change it; new experiments use the sidebar setting.')
            choice=st.selectbox('Change this experiment to',['Standard','Fast','Turbo'],index=[1,8,32].index(cfg.learn_every),key=f'speed:{run}')
            if st.button('Apply speed',disabled=live):
                state['config']['learn_every']={'Standard':1,'Fast':8,'Turbo':32}[choice]
                store.save(run,state);store.event(run,'learning_speed_changed',learn_every=state['config']['learn_every']);st.rerun()
    identities=portfolio_choices(store,state,metrics) if page in ('Market','Bot profile') else None
    if page=='Bots':
        st.subheader('Meet your traders')
        st.caption('Survived means it passed the risk limit, not that it is still training or won. Only the qualified winner is labeled Champion. Each session has its own $10,000 portfolio.')
        if winner:st.link_button('Open champion’s portfolio',f'?run={run}&bot={winner}',width='stretch')
        search=st.text_input('Find a bot',placeholder='Search name or bot ID')
        _,total,pages=bot_directory(state,metrics,search)
        directory_page=st.number_input('Bot list page',1,pages,1,key=f'directory:{run}:{search}')
        entries,_,_=bot_directory(state,metrics,search,int(directory_page))
        st.caption(f'{total:,} matching bots · page {directory_page} of {pages} · showing up to 25 at a time')
        for bot,info,record in entries:
            with st.container(border=True):
                left,right=st.columns([4,1])
                left.markdown(f'**{bot_name(bot)}**')
                status=bot_status(state,bot,record,live)
                left.markdown(f'**{status["label"]}**')
                left.caption(f'{status["detail"]} · Born in generation {info["birth"]+1} · ID: {bot}')
                right.link_button('Open portfolio',f'?run={run}&bot={bot}',width='stretch')
        return
    if page in ('Market','Bot profile'):
        if follow and not leader:
            return
        if identities.empty:
            st.info('Your first bot is learning. A portfolio appears when its first session finishes.');return
        bot_ids=sorted(set(identities.bot)-{'buy-and-hold'})
        bot_ids=sorted(set(bot_ids)|{b['id'] for b in state['bots']})
        requested=st.query_params.get('bot')
        preferred=requested if requested in bot_ids else latest.bot.iloc[0] if not latest.empty and latest.bot.iloc[0] in bot_ids else bot_ids[0]
        if follow:
            bot = leader['bot']
            st.text_input('Following trader',value=bot_name(bot),disabled=True)
        else:
            bot=st.selectbox('Your trader',bot_ids,index=bot_ids.index(preferred),format_func=bot_name,key=f'profile:{run}:{page}')
        own=metrics[metrics.bot==bot] if not metrics.empty else metrics
        lifecycle=bot_status(state,bot,own.iloc[0].to_dict() if not own.empty else None,live)
        st.write(f'**Status: {lifecycle["label"]}** · {lifecycle["detail"]}')
        st.caption(f'Bot ID: {bot}')
        if page=='Bot profile':
            st.subheader(bot_name(bot))
            own=metrics[metrics.bot==bot] if not metrics.empty else metrics
            if not own.empty:
                r=own.iloc[0]
                st.caption(f'Born in generation {r.birth+1} · Parent: {bot_name(r.parent) if r.parent else "Original bot"}')

        modes=identities[identities.bot==bot]['mode'].tolist()
        if not modes: st.info('This bot has not finished its first session yet.');return
        preferred_mode='paper' if 'paper' in modes else 'final_test' if bot==winner and 'final_test' in modes else f'evaluation:g{state["generation"]}'
        if follow:
            mode=leader['mode']
            if mode not in modes:
                st.info('The latest portfolio is being saved. Checking again shortly…');return
            st.caption(f'Following session: {session_name(mode)}')
        else:
            mode=st.selectbox('Portfolio session',modes,index=modes.index(preferred_mode) if preferred_mode in modes else len(modes)-1,format_func=session_name,key=f'session:{run}:{bot}:{page}')
        st.caption('A session is one simulated life. Learning and practice tests start with separate balances. Holdings below belong only to the selected session.')
        curve=store.rows('SELECT * FROM equity WHERE run=? AND bot=? AND mode=? ORDER BY time,id',(run,bot,mode)).drop_duplicates('time',keep='last')
        trades=store.rows('SELECT * FROM trades WHERE run=? AND bot=? AND mode=? ORDER BY time,id',(run,bot,mode))
        if curve.empty: return
        snapshot=portfolio(store,state,bot,mode,curve,trades)
        cols=st.columns(4)
        cols[0].metric('Portfolio value',f'${snapshot["equity"]:,.2f}',f'{snapshot["equity"]/cfg.initial_cash-1:+.2%}')
        cols[1].metric('Available cash',f'${snapshot["cash"]:,.2f}')
        cols[2].metric('Biggest drop from peak',f'{curve.drawdown.max():.2%}')
        cols[3].metric('Trades made',str(len(trades)))
        score_report(store,run,bot,mode,snapshot,cfg)
        chart_col,watch=st.columns([4,1])
        with watch:
            st.markdown('**Watchlist**')
            symbol=st.radio('Stock',list(cfg.symbols),label_visibility='collapsed',key=f'symbol:{run}:{page}')
            interval=st.selectbox('Candles',['5min','15min','1h'],format_func=lambda x:{'5min':'5 minutes','15min':'15 minutes','1h':'1 hour'}[x])
            span=st.selectbox('Chart range',['Session','Last day','Last 5 days','Last month'],index=2 if mode in ('validation','final_test') else 0)
        with chart_col:
            st.markdown(f'#### {symbol} · {interval} · {session_name(mode)}')
            path=HOME/run/'live-bars.parquet' if mode=='paper' else Path(state['dataset'])/'bars.parquet'
            bars=pd.read_parquet(path,filters=[('symbol','=',symbol)])
            bars['time']=pd.to_datetime(bars.time,utc=True)
            end=pd.to_datetime(curve.time,utc=True).max()
            start=pd.to_datetime(curve.time,utc=True).min()
            if span!='Session': start=max(start,end-pd.Timedelta(days={'Last day':1,'Last 5 days':5,'Last month':30}[span]))
            bars=bars[(bars.time>=start)&(bars.time<=end)]
            plotted=trades[(pd.to_datetime(trades.time,utc=True)>=start)&(pd.to_datetime(trades.time,utc=True)<=end)]
            if not bars.empty:
                st.plotly_chart(stock_chart(bars,plotted,symbol,interval,f'{run}:{bot}:{mode}:{symbol}:{span}:{interval}'),width='stretch',key=f'candles:{run}:{page}',config={'scrollZoom':True,'displaylogo':False})
            else: st.info('No stock bars available in this range.')
            st.caption('Scroll to zoom · drag a rectangle to zoom in · use the toolbar to pan · double-click to reset. Times are New York time. Prices are unadjusted; splits may show price jumps.')
        holdings, performance, activity=st.tabs(['Portfolio','Balance over time','Trade history'])
        with holdings:
            st.markdown('#### What this bot owns')
            frame=holdings_frame(snapshot,cfg.symbols)
            st.dataframe(frame,hide_index=True,width='stretch',column_config={'Allocation':st.column_config.NumberColumn(format='percent'),'Shares':st.column_config.NumberColumn(format='%.4f'),'Price ($)':st.column_config.NumberColumn(format='$%.2f'),'Value ($)':st.column_config.NumberColumn(format='$%.2f')})
            st.caption(f'Portfolio as of {snapshot["asof"]}. Uninvested cash: ${snapshot["cash"]:,.2f}.')
            if snapshot.get('receivables'): st.write('Dividends awaiting payment:',sum(d['amount'] for d in snapshot['receivables']))
        with performance: st.plotly_chart(balance_chart(curve,f'balance:{run}:{bot}:{mode}'),width='stretch',config={'scrollZoom':True,'displaylogo':False})
        with activity:
            display=trades.copy();display['Action']=np.where(display.shares>0,'Buy','Sell');display['Shares']=display.shares.abs()
            st.dataframe(display[['time','symbol','Action','Shares','price','fee','version']].iloc[::-1],hide_index=True,width='stretch')
            st.download_button('Download this bot’s trades',display.to_csv(index=False),file_name=f'{bot}-{mode}.csv')
        if page=='Market': st.link_button('Open this bot’s personal page',f'?run={run}&bot={bot}')
    if page == 'Reports':
        st.markdown('#### Evaluation boundaries')
        st.write(f'Train: 2020–2023 · Validation: 2024 · Final test: 2025')
        st.caption('Validation and final testing disable learning. Live paper learning is tracked separately. '
                   'A reused final-test period blocks automatic promotion.')
        if state['validation']:
            st.dataframe(pd.DataFrame([dict(Trader=bot_name(r['bot']), **r['result']) for r in state['validation']]).rename(columns={'alive':'Survived risk checks'}), hide_index=True)
        if state['test']:
            st.json({('survived_risk_checks' if k=='alive' else k):v for k,v in state['test'].items()})
        if state.get('baselines'):
            st.markdown('#### Buy-and-hold and cash comparisons')
            comparisons = []
            for period, baseline in state['baselines'].items():
                candidate = state.get('champion_validation') if period == 'validation' else state.get('test')
                comparisons.append({'Period': period,
                                    ('Champion return' if winner else 'Finalist return'): candidate['net_return'] if candidate else None,
                                    'Buy-and-hold return': baseline['buy_and_hold']['net_return'],
                                    'Buy-and-hold drawdown': baseline['buy_and_hold']['drawdown'],
                                    'Cash return': 0.0})
            st.dataframe(pd.DataFrame(comparisons), hide_index=True, width='stretch')
        st.markdown('#### Data & assumptions')
        st.json({'feed': state['meta']['feed'], 'coverage': state['meta']['coverage'],
                 'fingerprint': state['fingerprint'], 'fee_bps': cfg.fee_bps,
                 'slippage_bps': cfg.slippage_bps, 'bar_volume_participation': cfg.participation,
                 'universe': cfg.symbols})
        report = HOME / run / 'report.json'
        if report.exists():
            st.download_button('Download experiment report', report.read_text(), file_name=f'{run}-report.json', mime='application/json')
        st.download_button('Download leaderboard CSV', metrics.to_csv(index=False), file_name=f'{run}-leaderboard.csv')
        events = store.rows('SELECT time,kind,payload FROM events WHERE run=? ORDER BY id DESC LIMIT 100', (run,))
        st.dataframe(events, hide_index=True, width='stretch')


floor(run)
