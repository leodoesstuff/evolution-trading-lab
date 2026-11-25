"""Versioned day-trading rules and separately reported activity/performance metrics."""
from functools import lru_cache
import math
import numpy as np
import pandas as pd
from .data import session_grid


@lru_cache(maxsize=4096)
def close_time(day):
    bars=session_grid(day,day)
    return bars[-1]+pd.Timedelta(minutes=5) if len(bars) else None


def day_context(timestamp):
    t=pd.Timestamp(timestamp)
    if t.tzinfo is None:t=t.tz_localize('UTC')
    day=str(t.tz_convert('America/New_York').date())
    return day,t,close_time(day)


def enabled(config):
    return getattr(config,'rules_version',1)>=2


def summarize(book, config, net):
    a=getattr(book,'analytics',{})
    bars=a.get('bars',0)
    daily=a.get('days',{})
    returns=[v['end']/max(v['start'],1e-8)-1 for v in daily.values()]
    active_days=sum(v.get('active',False) for v in daily.values())
    mean_exposure=a.get('exposure_sum',0)/max(bars,1)
    roundtrips=a.get('roundtrips',0)
    days=len(daily)
    meaningful=bool(days and roundtrips>=min(config.min_roundtrips,days) and
                    mean_exposure>=config.min_mean_exposure and
                    active_days>=max(1,math.ceil(days*config.min_active_day_fraction)))
    flat=not a.get('flat_failed',False)
    reasons=[]
    if not meaningful:reasons.append('Insufficient meaningful trading activity')
    if not flat:reasons.append('Could not exit positions within available liquidity')
    if book.max_drawdown>=config.death_drawdown:reasons.append('Drawdown limit exceeded')
    return dict(rules_version=2,profit_dollars=float(net*config.initial_cash),
                mean_exposure=float(mean_exposure),active_bar_fraction=a.get('active_bars',0)/max(bars,1),
                roundtrips=int(roundtrips),days=days,active_days=active_days,
                turnover=a.get('notional',0)/config.initial_cash,fees=a.get('fees',0),
                slippage_cost=a.get('slippage',0),daily_returns=returns,
                profitable_days=sum(r>0 for r in returns),worst_day=min(returns,default=0),
                daily_loss_stops=a.get('loss_stops',0),flat_at_close=flat,
                activity_eligible=meaningful,selection_eligible=meaningful and flat and book.alive,
                eligibility_reason='; '.join(reasons) or 'Active and within risk limits',
                score_parts=dict(profit=net,drawdown_penalty=.5*book.max_drawdown,
                                 consistency_penalty=float(np.std(returns)) if returns else 0.))


def eligible(result, require_profit=False):
    if result.get('rules_version',1)<2:return bool(result['alive'])
    return bool(result.get('selection_eligible',False) and (not require_profit or result['net_return']>0))


def breeding_eligible(result, config, generation):
    if not result['alive']:
        return False
    if not enabled(config):
        return True
    require_profit=(getattr(config,'training_selection','profit_after_warmup')!='active_survivors'
                    and generation>=config.warmup_generations)
    return eligible(result,require_profit)


def clock_features(timestamp):
    day,stamp,closing=day_context(timestamp)
    if closing is None:return [0.,0.]
    opening=pd.Timestamp(day+' 09:30',tz='America/New_York').tz_convert('UTC')
    observed=stamp+pd.Timedelta(minutes=5)
    return [float(np.clip((observed-opening).total_seconds()/max((closing-opening).total_seconds(),1),0,1)),
            float(np.clip((closing-observed).total_seconds()/(390*60),0,1))]


def live_exit_required(book,config,prices,now):
    if not enabled(config):return False
    day,stamp,closing=day_context(now)
    equity=book.equity(prices)
    a=book.analytics
    record=a.setdefault('days',{}).setdefault(day,dict(start=equity,end=equity,active=False,halted=False))
    if not record['halted'] and equity/max(record['start'],1e-8)-1<=-config.max_daily_loss:
        record['halted']=True;a['loss_stops']=a.get('loss_stops',0)+1
    exposure=float(np.dot(book.shares,np.nan_to_num(prices))/max(equity,1e-8))
    record['end']=equity;record['active']=record['active'] or exposure>=config.min_mean_exposure
    bucket=str(stamp.floor('5min'))
    if a.get('last_bucket')!=bucket:
        a['last_bucket']=bucket;a['bars']=a.get('bars',0)+1;a['exposure_sum']=a.get('exposure_sum',0)+exposure
        a['active_bars']=a.get('active_bars',0)+int(exposure>=config.min_mean_exposure)
    return record['halted'] or closing is not None and stamp>=closing-pd.Timedelta(minutes=10)


def flag_unclosed(book,config,now):
    if not enabled(config) or not np.any(book.shares>1e-8):return False
    day,stamp,closing=day_context(now)
    held_days=list(book.analytics.get('days',{}))
    if (closing is not None and stamp>=closing) or held_days and max(held_days)<day:
        book.analytics['flat_failed']=True;book.alive=False
        return True
    return False
