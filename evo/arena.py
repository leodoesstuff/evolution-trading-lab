"""Independent live-practice sessions with one shared, read-only IEX feed."""
import copy
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import requests
import torch
from .config import Config
from .data import Alpaca, prepare, session_grid
from .model import Bot
from .paper import fresh_quotes, pending_is_fresh, utcnow
from .sim import Book, allocation, observation
from .store import Store
from .day_rules import live_exit_required, flag_unclosed
from .cadence import decision_due, transition_bars


def saved_candidates(source):
    """Only weights that are actually saved, with distinct snapshot labels."""
    result = {}
    for b in source['bots']:
        result[f'population:{b["id"]}'] = copy.deepcopy(b)
    for h in source.get('hall', []):
        b=h['bot']; result[f'best-snapshot:{b["id"]}'] = copy.deepcopy(b)
    if source.get('champion'):
        result['champion'] = copy.deepcopy(source['champion'])
    if source.get('paper'):
        result['paper-latest'] = copy.deepcopy(source['paper']['bot'])
    return result


def create_practice(home, source, choices, learning=True, name='Live practice'):
    available=saved_candidates(source)
    if not choices or len(choices)>10 or len(choices)!=len(set(choices)) or any(k not in available for k in choices):
        raise ValueError('Choose 1–10 distinct saved snapshots.')
    cfg=Config(**source['config']);cfg.validate()
    expected=len(cfg.symbols)*6
    if len(source['meta'].get('scaler_mean',[]))!=expected:
        raise ValueError('This snapshot has no compatible saved feature scaler.')
    session=time.strftime('practice-%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:6]
    participants=[]
    for i,key in enumerate(choices):
        packed=available[key]
        participants.append(dict(id=f'entrant-{i+1}',source_bot=packed['id'],snapshot=key,
                                 bot=packed,book=Book.fresh(cfg),pending=None,transition=None,
                                 last_bar=None,status='waiting',enabled=True,trade_count=0,costs=0.))
    state=dict(run=session,name=name.strip()[:80] or 'Live practice',kind='practice',
               status='ready',message='Waiting to connect',source_run=source['run'],
               source_fingerprint=source['fingerprint'],source_synthetic=source['meta'].get('synthetic',False),
               created=utcnow().isoformat(),started=None,config=cfg.to_dict(),meta=copy.deepcopy(source['meta']),
               learning=bool(learning),participants=participants,
               benchmark=dict(book=Book.fresh(cfg),pending=None,entered=False,trade_count=0,costs=0.),
               elapsed=0.,quotes=None,quote_time=None,heartbeat=None,actions_through=None,
               last_bar=None,connection='not connected',market_open=False,rng=torch.get_rng_state())
    store=Store(home);store.save(session,state)
    store.event(session,'practice_created',source_run=source['run'],snapshots=choices,learning=learning,
                qualification='experimental',source_synthetic=state['source_synthetic'])
    return session


def apply_actions(book, actions, after, day, symbols):
    due=actions[(actions['date']>=after)&(actions['date']<=day)]
    for action_day,group in due.groupby('date',sort=True):
        fresh=group[~group.id.isin(book.applied)]
        splits=np.ones(len(symbols));removed=np.zeros(len(symbols),bool);dividends=[]
        for a in fresh.to_dict('records'):
            if a['symbol'] not in symbols: continue
            j=symbols.index(a['symbol'])
            if a['kind'] in ('forward_split','reverse_split'):splits[j]*=a['ratio']
            elif a['kind']=='cash_dividend':dividends.append((j,a['cash'],a['payable']))
            elif a['kind']=='worthless_removal':removed[j]=True
            else:raise ValueError(f'Unsupported corporate action: {a["kind"]}. Practice halted for reconciliation.')
        if len(fresh):
            book.actions(splits,dividends,removed,action_day,'events:'+':'.join(sorted(fresh.id.astype(str))))
            book.applied.update(fresh.id.tolist())
    # Use a unique settlement key per call: newly discovered payable entitlements still settle.
    book.actions(np.ones(len(symbols)),[],np.zeros(len(symbols),bool),day,'settle:'+str(len(book.applied)))


def discard_orders(state):
    for p in state['participants']:
        p['pending']=p['transition']=None
    state['benchmark']['pending']=None


def save_practice(store,state,bots):
    for p,bot in zip(state['participants'],bots):p['bot']=bot.pack()
    state['rng']=torch.get_rng_state();state['heartbeat']=utcnow().isoformat()
    store.save(state['run'],state)
    report={k:state.get(k) for k in ('run','name','status','source_run','created','started','elapsed','learning','quote_time','connection','message')}
    report['qualification']='experimental';report['order_execution']='local simulation only'
    path=store.home/state['run']/'report.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,path)


def flush_book(store,state,key,participant,prices):
    book=participant['book']
    participant['trade_count']+=len(book.trades)
    participant['costs']+=sum(t[4] for t in book.trades)
    store.log_trace(state['run'],key,'practice',book.version,book.trades,book.curve,book=book,prices=prices)
    book.trades.clear()
    if book.curve:book.curve[:]=book.curve[-1:]


def process_quote(state,bots,markets,mids,bids,asks,capacity,now,store):
    """All entrants see the identical quote timestamp and completed-bar set."""
    cfg=Config(**state['config']);session=state['run']
    for p,bot,market in zip(state['participants'],bots,markets):
        book=p['book'];book.version=bot.version
        if not book.alive:
            p['status']='eliminated';p['pending']=p['transition']=None
            # Keep a comparable frozen value in the competition after elimination.
            if book.curve:book.curve.append((str(now),*book.curve[-1][1:]))
            flush_book(store,state,p['id'],p,getattr(book,'last_prices',mids));continue
        equity=book.observe(mids,now,cfg)
        if not book.alive:
            p['status']='eliminated';p['pending']=p['transition']=None
            store.event(session,'practice_eliminated',entrant=p['id'],source_bot=p['source_bot'])
        elif p['enabled']:
            exiting=live_exit_required(book,cfg,mids,now)
            if exiting and p['pending'] and not p['pending'].get('liquidate'):p['pending']=p['transition']=None
            if exiting and p['pending'] is None and np.any(book.shares>1e-8):
                p['pending']=dict(created=now.isoformat(),targets=[0.]*len(cfg.symbols),version=bot.version,liquidate=True)
            pending=p['pending']
            if pending and pending['created']==now.isoformat():pending=None
            if pending:
                if pending_is_fresh(pending,now):
                    target=np.asarray(pending['targets']);weights=book.shares*mids/max(equity,1e-8)
                    book.version=pending['version']
                    book.execute(target,np.where(target>weights,asks,bids),capacity/cfg.participation,now,cfg,liquidate=pending.get('liquidate',False))
                    equity=book.observe(mids,now,cfg)
                    p['transition']=None if pending.get('liquidate') else pending
                else:store.event(session,'practice_order_expired',entrant=p['id'])
                p['pending']=None
            latest=market.times[-1] if market is not None else None
            recent=latest is not None and 0<=(now-latest-pd.Timedelta(minutes=5)).total_seconds()<=120
            if book.alive and not exiting and recent and decision_due(p['last_bar'],latest,cfg):
                obs=observation(market,len(market.times)-1,book)
                obs[-(len(cfg.symbols)+2):]=np.r_[book.shares*mids/max(equity,1e-8),book.cash/max(equity,1e-8),book.drawdown(mids)]
                previous=p['transition']
                if previous and state['learning']:
                    reward=100*np.log(max(equity,1e-8)/max(previous['equity'],1e-8))-10*max(0,book.max_drawdown-previous['dd'])
                    bot.learn(np.asarray(previous['obs'],dtype=np.float32),np.asarray(previous['action']),float(reward),obs,False,
                              steps=transition_bars(previous.get('bar') or p['last_bar'],latest,cfg))
                    # Entrant-specific archive IDs avoid collisions when snapshots share a source ID.
                    archive=copy.copy(bot);archive.id=p['id'];store.save_model(session,archive)
                p['transition']=None
                action=bot.act(obs,state['learning'])
                p['pending']=dict(created=now.isoformat(),targets=allocation(action,book,mids,cfg).tolist(),
                                  obs=obs.tolist(),action=action.tolist(),equity=equity,dd=book.max_drawdown,version=bot.version,bar=str(latest))
                p['last_bar']=str(latest)
            p['status']='active' if book.alive else 'eliminated'
        flush_book(store,state,p['id'],p,mids)
    benchmark=state['benchmark'];book=benchmark['book']
    # Benchmark keeps holding through drawdowns, unlike eliminated contestants.
    book.alive=True
    book.observe(mids,now,cfg)
    if benchmark['pending'] and pending_is_fresh(benchmark['pending'],now):
        book.alive=True
        book.execute(np.full(len(cfg.symbols),cfg.max_weight),asks,capacity/cfg.participation,now,cfg)
        book.observe(mids,now,cfg);benchmark['pending']=None;benchmark['entered']=bool(book.trades)
    elif benchmark['pending']:
        benchmark['pending']=None  # Never replay an expired initial fill.
    if not benchmark['entered'] and not benchmark['pending']:
        benchmark['pending']={'created':now.isoformat()}
    flush_book(store,state,'buy-and-hold',benchmark,mids)
    # Cash benchmark shares the same timestamps and starting balance, with no interest.
    store.log_trace(session,'cash','practice',0,[],[(str(now),cfg.initial_cash,cfg.initial_cash,0.,0)])
    if state['started'] is None:state['started']=now.isoformat()
    state['quotes']=mids.tolist();state['quote_time']=now.isoformat()


def run_practice(home,session,minutes=60,client=None):
    torch.set_num_threads(1)
    store=Store(home);state=store.load(session)
    if state['status'] in ('stopped','finished'):return state
    cfg=Config(**state['config']);torch.set_rng_state(state['rng'])
    bots=[Bot.unpack(p['bot'],len(state['meta']['scaler_mean'])+len(cfg.symbols)*2+2+(2 if getattr(cfg,'rules_version',1)>=2 else 0),cfg) for p in state['participants']]
    client=client or Alpaca(cancelled=lambda:store.control(session)!='run')
    discard_orders(state);markets=[None]*len(bots);last_fetch=None;retry=0
    began=time.monotonic();deadline=began+minutes*60
    def save():
        nonlocal began
        state['elapsed']+=time.monotonic()-began;began=time.monotonic();save_practice(store,state,bots)
    state['status']='running';save()
    try:
        while time.monotonic()<deadline and store.control(session)=='run':
            now=utcnow();day=str(now.tz_convert('America/New_York').date());grid=session_grid(day,day)
            state['market_open']=bool(len(grid) and grid[0]<=now<grid[-1]+pd.Timedelta(minutes=5))
            if not state['market_open']:
                for p in state['participants']:
                    if flag_unclosed(p['book'],cfg,now):p['status']='unable to exit before close'
                discard_orders(state);state.update(connection='market closed',message='Waiting for the next regular US market session.',status='waiting')
                save();time.sleep(5);continue
            try:
                desired=str(now.floor('5min')-pd.Timedelta(minutes=5))
                if last_fetch!=desired and time.monotonic()<retry:
                    save();time.sleep(5);continue
                if last_fetch!=desired and time.monotonic()>=retry:
                    after=state['actions_through'] or day
                    start=(now-pd.Timedelta(days=14)).isoformat()
                    bars=client.bars(cfg.symbols,start,now.isoformat())
                    bars=bars[pd.to_datetime(bars.time,utc=True)+pd.Timedelta(minutes=5)<=now]
                    actions=client.actions(cfg.symbols,min(start[:10],after),str((now+pd.Timedelta(days=45)).date()))
                    meta=dict(state['meta'],synthetic=False)
                    market=prepare(bars,actions,cfg.symbols,meta,cfg.train_end)
                    ids=np.flatnonzero(market.times+pd.Timedelta(minutes=5)<=now)
                    if len(ids)<79:raise ValueError('Waiting for enough completed bars.')
                    i=int(ids[-1])
                    if not (np.isfinite(market.closes[i])&(market.volumes[i]>0)).all():raise ValueError('Waiting: a latest completed bar is missing.')
                    # Trim future grid rows so a policy cannot accidentally inspect them.
                    for field in ('times','opens','closes','volumes','marks','features','splits','dividends','removed'):
                        setattr(market,field,getattr(market,field)[:i+1])
                    if (now-market.times[-1]-pd.Timedelta(minutes=5)).total_seconds()>120:raise ValueError('Waiting for a fresh completed bar.')
                    for p in state['participants']+[state['benchmark']]:apply_actions(p['book'],actions,after,day,cfg.symbols)
                    state['actions_through']=day;markets=[market]*len(bots)
                    folder=store.home/session;bars.to_parquet(folder/'live-bars.tmp.parquet',index=False);os.replace(folder/'live-bars.tmp.parquet',folder/'live-bars.parquet')
                    last_fetch=desired;state['last_bar']=str(market.times[-1])
                if last_fetch!=desired:raise ValueError('Waiting for the next completed market bar.')
                payload=client.get('/v2/stocks/quotes/latest',{'symbols':','.join(cfg.symbols),'feed':'iex'})
                stamp=utcnow();mids,bids,asks,capacity=fresh_quotes(payload,cfg.symbols,stamp)
                if not all(np.isfinite(v).all() for v in (mids,bids,asks,capacity)):
                    raise ValueError('Waiting: non-finite quote data; queued orders discarded.')
                if store.control(session)!='run':break
                process_quote(state,bots,markets,mids,bids,asks,capacity,stamp,store)
                state.update(status='running',connection='connected',message='Live IEX data · local simulated fills · shared competition feed')
                if all(not p['book'].alive for p in state['participants']):state['status']='finished';save();return state
                save()
            except (ValueError,RuntimeError,requests.RequestException) as exc:
                discard_orders(state);last_fetch=None;retry=time.monotonic()+15
                state.update(status='waiting',connection='waiting for data',message='Network unavailable; queued orders discarded.' if isinstance(exc,requests.RequestException) else str(exc))
                save()
            time.sleep(5)
    except Exception:
        discard_orders(state);state.update(status='error',connection='error',message='Practice worker stopped unexpectedly. Resume to retry from its checkpoint.');save();raise
    discard_orders(state)
    state['status']='stopped' if store.control(session) in ('stop','emergency') else 'paused'
    state['message']='Session stopped; recorded holdings retained.' if state['status']=='stopped' else 'Paused. Resume continues these portfolios without resetting balances.'
    save();return state
