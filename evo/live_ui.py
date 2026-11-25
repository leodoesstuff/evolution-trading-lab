"""Streamlit live room. Practice storage is separate from training experiments."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .arena import create_practice,saved_candidates
from .config import Config,ROOT
from .data import session_grid
from .store import Store
from .score_ui import score_report
from .status_labels import practice_status
from .views import holdings_frame,stock_chart,style_chart
from .names import bot_name


def active(home,session):
    path=Path(home)/session/'worker.lock'
    if not path.exists():return False
    with path.open('r') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return False
        except BlockingIOError:return True


def launch_practice(home,session,minutes):
    if active(home,session):return
    store=Store(home);store.request(session,'run')
    with (Path(home)/session/'worker.log').open('a') as log:
        subprocess.Popen([sys.executable,'-m','evo.practice_worker','--home',str(home),'--session',session,'--minutes',str(minutes)],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)


def ranking_frame(state,worker_running=False):
    cfg=Config(**state['config']);rows=[]
    for p in state['participants']:
        b=p['book'];equity=float(b.curve[-1][1]) if b.curve else cfg.initial_cash
        rows.append({'Trader':bot_name(p['source_bot']),'Entrant':p['id'],'Status':practice_status(state,p,worker_running),
                     'Balance ($)':equity,'Return':equity/cfg.initial_cash-1,'Cash ($)':b.cash,
                     'Max drawdown':b.max_drawdown,'Trades':p['trade_count'],'Fees ($)':p['costs']})
    return pd.DataFrame(rows).sort_values(['Return','Max drawdown','Entrant'],ascending=[False,True,True])


def next_open(now):
    grid=session_grid(str(now.date()),str((now+pd.Timedelta(days=10)).date()))
    future=grid[grid>now]
    if not len(future):return 'Unavailable'
    # First future regular-session bar is a useful next update estimate during market hours.
    return future[0].tz_convert('America/New_York').strftime('%a %b %d, %I:%M %p ET')


def render_live_room(source,home):
    home=Path(home);store=Store(home)
    st.subheader('Live trading room')
    st.caption('Real IEX prices. Fake money. No orders are sent to Alpaca. Sessions have their own portfolios and do not change training results.')
    available=saved_candidates(source)
    with st.expander('Start a practice session or competition',expanded=not any(home.glob('*/checkpoint.pt'))):
        st.write('Pick one saved bot for practice, or several to compete side by side. Every entrant starts with $10,000 at the same time.')
        name=st.text_input('Practice session name',value='Live competition')
        selected=st.multiselect('Saved bot snapshots',list(available),default=list(available)[:min(3,len(available))],
                               format_func=lambda k:f'{bot_name(available[k]["id"])} · {k.split(":")[0]} · version {available[k]["version"]}')
        learning=st.checkbox('Keep learning during live practice',value=True)
        duration=st.selectbox('Practice session budget',[15,30,60,180],index=2,format_func=lambda n:f'{n} minutes')
        st.warning('Manual practice is experimental. Starting it does not qualify or promote a bot, including one that failed historical tests.')
        if source['meta'].get('synthetic'):st.caption('These snapshots were trained on generated demo prices. Live practice will still use real market data.')
        connected=bool(os.environ.get('APCA_API_KEY_ID') and os.environ.get('APCA_API_SECRET_KEY'))
        if not connected:st.info('Connect Alpaca in the sidebar’s Data connection panel to start. No keys are saved in practice checkpoints.')
        if st.button('Start live practice',disabled=not connected or not selected or len(selected)>10,type='primary'):
            session=create_practice(home,source,selected,learning,name)
            st.session_state['practice_session']=session
            launch_practice(home,session,duration);st.rerun()
    sessions=sorted([p.parent.name for p in home.glob('*/checkpoint.pt')],reverse=True)
    if not sessions:
        st.info('No live practice sessions yet. Training can keep running while you set one up.');return
    preferred=st.session_state.get('practice_session',sessions[0])
    session=st.selectbox('Live practice session',sessions,index=sessions.index(preferred) if preferred in sessions else 0)
    st.session_state['practice_session']=session
    refresh=st.toggle('Refresh live room automatically',value=True,help='Updates every ten seconds. Turn off to inspect chart zoom.')

    @st.fragment(run_every='10s' if refresh else None)
    def room():
        state=store.load(session);cfg=Config(**state['config']);running=active(home,session)
        now=pd.Timestamp.now(tz='UTC');day=str(now.tz_convert('America/New_York').date());grid=session_grid(day,day)
        opened=bool(len(grid) and grid[0]<=now<grid[-1]+pd.Timedelta(minutes=5))
        cells=st.columns(4)
        cells[0].metric('US stock market','Open' if opened else 'Closed')
        cells[1].metric('Data connection',state['connection'].capitalize())
        age=(now-pd.Timestamp(state['quote_time'])).total_seconds() if state['quote_time'] else None
        cells[2].metric('Last quote',f'{max(0,age):.0f}s ago' if age is not None else 'Waiting')
        cells[3].metric('Practice traders',str(len(state['participants'])))
        st.write(f'**{state["name"]}** · {state["status"].capitalize()} · Experimental practice')
        st.caption(state['message'])
        if not opened:st.info(f'Market closed. Positions are held; no simulated orders execute. Next regular-session bar: {next_open(now)}.')
        elif age is not None and age>60:st.warning('Quotes are stale. New fills are blocked until fresh data arrives.')
        if not running and state['status'] in ('running','waiting'):st.warning('The practice worker is no longer running. Resume uses its saved balances.')
        c=st.columns(4);terminal=state['status'] in ('finished','stopped')
        if c[0].button('Resume practice',disabled=running or terminal):launch_practice(home,session,60);st.rerun()
        if c[1].button('Pause practice',disabled=not running):store.request(session,'pause');st.toast('Pausing practice…')
        if c[2].button('Stop practice',disabled=terminal):
            store.request(session,'stop')
            if not running:state.update(status='stopped',message='Stopped; portfolios retained.');store.save(session,state)
            st.rerun()
        if c[3].button('Emergency stop practice',disabled=terminal):
            store.request(session,'emergency')
            if not running:state.update(status='stopped',message='Stopped; portfolios retained.');store.save(session,state)
            st.rerun()
        board=ranking_frame(state,running)
        st.markdown('#### Live competition')
        st.dataframe(board.drop(columns='Entrant'),hide_index=True,width='stretch',column_config={'Return':st.column_config.NumberColumn(format='percent'),'Max drawdown':st.column_config.NumberColumn(format='percent')})
        st.caption('All entrants share one feed and start with equal balances. Eliminated portfolios stop trading and retain their final recorded value. Fees are shown separately; returns also include spread and slippage.')
        choices=[p['id'] for p in state['participants']]
        names={p['id']:f'{bot_name(p["source_bot"])} · {p["id"].replace("entrant-","Seat ")}' for p in state['participants']}
        follow=st.toggle('Follow the best live practice bot',value=True,key=f'practice-follow:{session}')
        entrant=board.iloc[0]['Entrant'] if follow else st.selectbox('Practice trader',choices,format_func=names.get)
        p=next(p for p in state['participants'] if p['id']==entrant);book=p['book']
        st.caption(f'Bot ID: {p["source_bot"]} · {p["id"].replace("entrant-","Seat ")}')
        st.caption('Status: '+practice_status(state,p,running))
        st.markdown('#### '+names[entrant])

        prices=np.asarray(getattr(book,'last_prices',state['quotes'] or [0.]*len(cfg.symbols)))
        equity=float(book.curve[-1][1]) if book.curve else cfg.initial_cash
        snapshot=dict(shares=book.shares.tolist(),prices=prices.tolist(),cash=book.cash,equity=equity,analytics=getattr(book,'analytics',{}))
        st.caption('Day-trader v2 exits before close.' if cfg.rules_version>=2 else 'Legacy overnight snapshot. New day-trader training experiments use v2 rules.')
        score_report(store,session,entrant,'practice',snapshot,cfg)
        st.dataframe(holdings_frame(snapshot,cfg.symbols),hide_index=True,width='stretch',column_config={'Allocation':st.column_config.NumberColumn(format='percent')})
        st.caption(f'Cash: ${book.cash:,.2f} · Model version {p["bot"]["version"]} · Learning {"on" if state["learning"] else "off"} · Saved snapshot: {p["snapshot"]}')
        symbol=st.selectbox('Live stock chart',cfg.symbols,key=f'live-symbol:{session}')
        trades=store.rows('SELECT * FROM trades WHERE run=? AND bot=? ORDER BY time,id',(session,entrant))
        path=home/session/'live-bars.parquet'
        if path.exists():
            bars=pd.read_parquet(path,filters=[('symbol','=',symbol)])
            bars=bars.tail(156)
            st.plotly_chart(stock_chart(bars,trades,symbol,'5min',f'live:{session}:{entrant}:{symbol}'),width='stretch',config={'scrollZoom':True,'displaylogo':False},key=f'practice-chart:{session}')
            st.caption('Five-minute completed bars · scroll to zoom, drag to select, double-click to reset.')
        else:st.info('Live chart will appear after the first completed-bar download during market hours.')
        if state['quotes']:
            st.markdown('#### Latest live prices')
            st.dataframe(pd.DataFrame({'Stock':cfg.symbols,'IEX midpoint ($)':state['quotes']}),hide_index=True,width='stretch')
        st.markdown('#### Performance against holding stocks and cash')
        curves=store.rows('SELECT bot,time,equity FROM equity WHERE run=? ORDER BY time,id',(session,))
        if not curves.empty:
            fig=go.Figure()
            for key,group in curves.groupby('bot'):
                label=names.get(key,{'buy-and-hold':'Buy & hold basket','cash':'Hold cash'}.get(key,key))
                fig.add_trace(go.Scatter(x=pd.to_datetime(group.time,utc=True),y=(group.equity/cfg.initial_cash-1)*100,name=label,mode='lines'))
            fig.update_yaxes(title='Return after costs (%)')
            st.plotly_chart(style_chart(fig,f'comparison:{session}',360),width='stretch',config={'scrollZoom':True,'displaylogo':False})
            benchmark=state['benchmark'];b=benchmark['book'];bv=float(b.curve[-1][1]) if b.curve else cfg.initial_cash
            st.write(f'Buy & hold: **{bv/cfg.initial_cash-1:+.2%}** · Hold cash: **0.00%** · Selected bot vs basket: **{(equity-bv)/cfg.initial_cash:+.2%}**')
            st.caption('Basket buys up to 20% per stock once using the same fresh quotes, fees, slippage and quote-size limits, then holds. It is a same-stock benchmark, not the S&P 500. Cash earns no interest. Corporate actions apply to both bots and basket.')
        st.markdown('#### Trades as they happen')
        alltrades=store.rows('SELECT * FROM trades WHERE run=? ORDER BY id DESC LIMIT 250',(session,))
        if not alltrades.empty:
            alltrades['Trader']=alltrades.bot.map(lambda k:names.get(k,k));alltrades['Action']=np.where(alltrades.shares>0,'Buy','Sell')
            st.dataframe(alltrades[['time','Trader','symbol','Action','shares','price','fee','version']],hide_index=True,width='stretch')
        else:st.caption('No simulated fills yet. Orders require an open market, completed bars, and fresh quotes.')
        st.download_button('Download practice trades',store.rows('SELECT * FROM trades WHERE run=? ORDER BY id',(session,)).to_csv(index=False),file_name=f'{session}-trades.csv')
        st.download_button('Download practice comparison',curves.to_csv(index=False),file_name=f'{session}-comparison.csv')
        report=home/session/'report.json'
        if report.exists():st.download_button('Download practice report',report.read_text(),file_name=f'{session}-report.json')
    room()
