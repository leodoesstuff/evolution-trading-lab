"""Presentation helpers: bot identities, portfolios and interactive charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from .names import bot_name


def session_name(mode):
    if ':g' in mode:
        kind, gen = mode.split(':g')
        return f"{'Learning' if kind == 'training' else 'Practice test'} · generation {int(gen)+1}"
    return {'validation':'Qualification · 2024', 'final_test':'Final exam · 2025', 'paper':'Live practice'}.get(mode,mode)


from .leaders import best_bot


def portfolio(store, state, bot, mode, curve, trades):
    """Exact saved books for new runs; corporate-action-aware reconstruction for older runs."""
    cached = store.rows('SELECT payload FROM portfolios WHERE run=? AND bot=? AND mode=?', (state['run'],bot,mode))
    if not cached.empty:
        return json.loads(cached.iloc[0].payload)
    if curve.empty:
        return None
    symbols = state['config']['symbols']
    start, end = pd.to_datetime(curve.time,utc=True).min(), pd.to_datetime(curve.time,utc=True).max()
    shares = {s:0. for s in symbols}
    events = []
    for r in trades.itertuples():
        events.append((pd.Timestamp(r.time),1,r.symbol,r.shares,'trade'))
    actions = pd.read_parquet(Path(state['dataset'])/'actions.parquet')
    for a in actions.to_dict('records'):
        t = pd.Timestamp(str(a['date'])[:10],tz='America/New_York').tz_convert('UTC')
        if start < t <= end and a['symbol'] in shares:
            events.append((t,0,a['symbol'],a['ratio'],a['kind']))
    for _,_,symbol,value,kind in sorted(events):
        if kind == 'trade': shares[symbol] += value
        elif kind in ('forward_split','reverse_split'): shares[symbol] *= value
        elif kind == 'worthless_removal': shares[symbol] = 0
    path = Path(state['dataset'])/'bars.parquet'
    if mode == 'paper': path = store.home/state['run']/'live-bars.parquet'
    bars = pd.read_parquet(path,columns=['time','symbol','close'])
    bars['time'] = pd.to_datetime(bars.time,utc=True)
    marks = bars[bars.time + pd.Timedelta(minutes=5) <= end].sort_values('time').groupby('symbol').tail(1).set_index('symbol').close.to_dict()
    return dict(cash=float(curve.cash.iloc[-1]),shares=[max(0.,shares[s]) for s in symbols],
                prices=[marks.get(s,0.) for s in symbols], asof=str(end),
                equity=float(curve.equity.iloc[-1]), reconstructed=True)


def holdings_frame(snapshot, symbols):
    quantities = np.asarray(snapshot['shares'])
    prices = np.asarray(snapshot['prices'])
    values = quantities*prices
    return pd.DataFrame({'Stock':symbols,'Shares':quantities,'Price ($)':prices,
                         'Value ($)':values,'Allocation':values/max(snapshot['equity'],1e-8)})


def style_chart(fig, identity, height=420):
    fig.update_layout(template='plotly_dark',paper_bgcolor='#101722',plot_bgcolor='#101722',
                      font=dict(color='#aebdd0'),height=height,margin=dict(l=12,r=18,t=32,b=20),
                      hovermode='x unified',dragmode='zoom',uirevision=identity,
                      legend=dict(orientation='h',y=1.08),xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor='#233044',fixedrange=False))
    return fig


def stock_chart(bars, trades, symbol, interval='5min', identity='stock'):
    b = bars.copy()
    b['time'] = pd.to_datetime(b.time,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    b = b.set_index('time').sort_index()
    if interval != '5min':
        b = b.resample(interval).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.8,.2])
    fig.add_trace(go.Candlestick(x=b.index,open=b.open,high=b.high,low=b.low,close=b.close,
                                increasing_line_color='#3bdbac',decreasing_line_color='#fa6a80',name=symbol),row=1,col=1)
    fig.add_trace(go.Bar(x=b.index,y=b.volume,name='IEX volume',marker_color=np.where(b.close>=b.open,'#246956','#843d50')),row=2,col=1)
    if not trades.empty:
        for buy, label, color, marker in [(True,'Buy','#3bdbac','triangle-up'),(False,'Sell','#fa6a80','triangle-down')]:
            t = trades[(trades.symbol==symbol)&((trades.shares>0)==buy)]
            fig.add_trace(go.Scatter(x=pd.to_datetime(t.time,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None),
                                    y=t.price,mode='markers',name=label,marker=dict(color=color,size=10,symbol=marker)),row=1,col=1)
    fig.update_xaxes(rangeslider_visible=False,rangebreaks=[dict(bounds=['sat','mon']),dict(bounds=[16,9.5],pattern='hour')])
    fig.update_xaxes(rangeslider=dict(visible=True,thickness=.1),row=2,col=1)
    fig.update_yaxes(title_text='Price ($)',row=1,col=1)
    return style_chart(fig,identity,520)


def balance_chart(curve, identity):
    fig=go.Figure(go.Scatter(x=pd.to_datetime(curve.time,utc=True),y=curve.equity,mode='lines',
                             name='Portfolio',line=dict(color='#3bdbac',width=2)))
    fig.update_yaxes(title='Portfolio ($)',rangemode='normal')
    return style_chart(fig,identity,280)
