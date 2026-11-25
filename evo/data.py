"""Raw IEX bars, exchange sessions, causal features, and explicit corporate actions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import requests
import exchange_calendars as xcals

ACTION_COLUMNS = ['id', 'symbol', 'kind', 'date', 'ratio', 'cash', 'payable']


def session_grid(start, end):
    # A calendar constructor requires a nonempty date range even on a holiday.
    cal = xcals.get_calendar('XNYS', start=pd.Timestamp(start) - pd.Timedelta(days=7),
                             end=pd.Timestamp(end) + pd.Timedelta(days=7))
    schedule = cal.schedule.loc[str(start)[:10]:str(end)[:10]]
    parts = [pd.date_range(row['open'], row['close'], freq='5min', inclusive='left')
             for _, row in schedule.iterrows()]
    if not parts:
        return pd.DatetimeIndex([], tz='UTC')
    return parts[0].append(parts[1:])


@dataclass
class Market:
    times: pd.DatetimeIndex
    symbols: tuple
    opens: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    marks: np.ndarray
    features: np.ndarray
    splits: np.ndarray
    dividends: np.ndarray
    payable: dict
    removed: np.ndarray
    meta: dict

    def bounds(self, start, end):
        ids = np.flatnonzero((self.times >= pd.Timestamp(start, tz='UTC')) &
                            (self.times < pd.Timestamp(end, tz='UTC')))
        if len(ids) < 80:
            raise ValueError(f'Not enough bars in {start} through {end}.')
        return int(ids[0]), int(ids[-1])


def prepare(bars, actions, symbols, meta, train_end='2024-01-01'):
    bars = bars.copy()
    bars['time'] = pd.to_datetime(bars['time'], utc=True)
    symbols = tuple(symbols)
    if bars.empty or set(symbols) - set(bars.symbol):
        raise ValueError('Missing symbols or empty bar data.')
    if bars.duplicated(['time', 'symbol']).any():
        raise ValueError('Duplicate symbol/timestamp bars.')
    if not bars.time.dt.minute.mod(5).eq(0).all() or not bars.time.dt.second.eq(0).all():
        raise ValueError('Bars must be aligned to five-minute boundaries.')
    for col in ('open', 'high', 'low', 'close', 'volume'):
        bars[col] = pd.to_numeric(bars[col], errors='raise')
        if not np.isfinite(bars[col]).all():
            raise ValueError(f'Non-finite {col} in bars.')
    if (bars[['open', 'high', 'low', 'close']] <= 0).any().any() or (bars.volume < 0).any():
        raise ValueError('Prices must be positive and volume nonnegative.')
    if ((bars.high < bars[['open', 'close', 'low']].max(axis=1)) |
            (bars.low > bars[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC prices.')
    start = bars.time.min().tz_convert('America/New_York').date()
    end = bars.time.max().tz_convert('America/New_York').date()
    grid = session_grid(start, end)
    if meta.get('synthetic'):
        # Demo only contains selected sessions, explicitly not a market-history test.
        dates = set(bars.time.dt.tz_convert('America/New_York').dt.date)
        grid = grid[np.array([t.tz_convert('America/New_York').date() in dates for t in grid])]
    bars = bars[bars.time.isin(grid)]
    if bars.empty:
        raise ValueError('No regular-session bars.')
    def matrix(col):
        return bars.pivot(index='time', columns='symbol', values=col).reindex(
            index=grid, columns=symbols).to_numpy(dtype=np.float64)
    opens, closes, volumes = matrix('open'), matrix('close'), matrix('volume')
    splits = np.ones_like(closes)
    dividends, removed = np.zeros_like(closes), np.zeros_like(closes, dtype=bool)
    payable = {}
    days = np.array([str(t.tz_convert('America/New_York').date()) for t in grid])
    first = {d: int(np.flatnonzero(days == d)[0]) for d in np.unique(days)}
    supported = {'forward_split', 'reverse_split', 'cash_dividend', 'worthless_removal'}
    for a in actions.to_dict('records'):
        if a['symbol'] not in symbols:
            continue
        if a['kind'] not in supported:
            raise ValueError(f"Unsupported corporate action {a['kind']} for {a['symbol']}; dataset blocked.")
        i, j = first.get(str(a['date'])[:10]), symbols.index(a['symbol'])
        if i is None:
            continue
        if a['kind'] in ('forward_split', 'reverse_split'):
            if not np.isfinite(a['ratio']) or a['ratio'] <= 0:
                raise ValueError('Invalid split ratio.')
            splits[i, j] *= a['ratio']
        elif a['kind'] == 'cash_dividend':
            if not np.isfinite(a['cash']) or a['cash'] < 0 or not a['payable']:
                raise ValueError('Incomplete dividend action.')
            dividends[i, j] += a['cash']
            payable.setdefault(i, []).append((j, float(a['cash']), str(a['payable'])[:10]))
        else:
            removed[i, j] = True
    # Carry marks only for valuation, never to fabricate tradable bars.
    marks = np.full_like(closes, np.nan)
    previous = np.full(len(symbols), np.nan)
    retired = np.zeros(len(symbols), dtype=bool)
    rets = np.zeros_like(closes)
    for i in range(len(grid)):
        carry = previous / splits[i]
        retired |= removed[i]
        current = np.where(np.isfinite(closes[i]), closes[i], carry)
        current[retired] = 0
        opens[i, retired] = np.nan
        closes[i, retired] = np.nan
        volumes[i, retired] = 0
        rets[i] = np.divide((current + dividends[i]) * splits[i], previous,
                            out=np.ones_like(previous), where=np.isfinite(previous) & (previous > 0)) - 1
        marks[i] = current
        previous = current
    # All features use only completed bars. No backwards filling or full-history price scaler.
    r = pd.DataFrame(rets, index=grid).clip(-1, 10)
    total = (1 + r).clip(lower=1e-8).cumprod()
    vol = pd.DataFrame(volumes, index=grid)
    feature_parts = [r, total.pct_change(12, fill_method=None), total.pct_change(78, fill_method=None),
                     r.rolling(12, min_periods=2).std(), total / total.rolling(12).mean() - 1,
                     vol / vol.rolling(12, min_periods=2).mean() - 1]
    raw = np.stack([p.to_numpy() for p in feature_parts], axis=2)
    raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0).reshape(len(grid), -1)
    training = grid < pd.Timestamp(train_end, tz='UTC')
    if not training.any() and 'scaler_mean' not in meta:
        raise ValueError('Training period missing; cannot fit scaler.')
    mean = np.array(meta.get('scaler_mean', raw[training].mean(axis=0) if training.any() else []))
    std = np.array(meta.get('scaler_std', raw[training].std(axis=0) if training.any() else []))
    std = np.maximum(std, 1e-5)
    features = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    valid = np.isfinite(opens) & np.isfinite(closes) & (volumes > 0)
    features = np.concatenate([features, valid.astype(np.float32)], axis=1)
    meta = dict(meta, scaler_mean=mean.tolist(), scaler_std=std.tolist(),
                coverage={s: float(valid[:, j].mean()) for j, s in enumerate(symbols)})
    return Market(grid, symbols, opens, closes, volumes, marks, features, splits,
                  dividends, payable, removed, meta)


def save_dataset(folder, bars, actions, meta):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    bars = bars.sort_values(['time', 'symbol']).reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    digest.update(actions.to_json().encode())
    meta = dict(meta, fingerprint=digest.hexdigest())
    bars.to_parquet(folder / 'bars.parquet', index=False)
    actions.to_parquet(folder / 'actions.parquet', index=False)
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return folder


def load_dataset(folder, symbols, train_end='2024-01-01', scaler=None):
    folder = Path(folder)
    meta = json.loads((folder / 'metadata.json').read_text())
    if scaler:
        meta.update(scaler)
    bars, actions = pd.read_parquet(folder / 'bars.parquet'), pd.read_parquet(folder / 'actions.parquet')
    # Catch accidental modification of a checkpoint's underlying dataset.
    digest = hashlib.sha256(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    digest.update(actions.to_json().encode())
    if digest.hexdigest() != meta['fingerprint']:
        raise ValueError('Dataset fingerprint mismatch. Import as a new dataset.')
    return prepare(bars, actions, symbols, meta, train_end)


def demo_dataset(folder, symbols, seed=42):
    rng = np.random.default_rng(seed)
    grids = []
    for year in range(2020, 2026):
        g = session_grid(f'{year}-01-02', f'{year}-02-01')
        days = g.tz_convert('America/New_York').normalize().unique()[:12]
        grids.append(g[g.tz_convert('America/New_York').normalize().isin(days)])
    times = grids[0].append(grids[1:])
    rows = []
    common = rng.normal(0.000015, 0.0014, len(times))
    for j, symbol in enumerate(symbols):
        returns = common + rng.normal(0.000005, 0.001, len(times))
        price = (80 + j * 30) * np.exp(np.cumsum(returns))
        opening = np.r_[price[0], price[:-1]]
        for i, t in enumerate(times):
            rows.append((t, symbol, opening[i], max(opening[i], price[i]) * 1.0005,
                         min(opening[i], price[i]) * .9995, price[i], float(rng.integers(20000, 100000))))
    bars = pd.DataFrame(rows, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    return save_dataset(folder, bars, pd.DataFrame(columns=ACTION_COLUMNS),
                        {'synthetic': True, 'feed': 'SYNTHETIC DEMO', 'symbols': list(symbols),
                         'description': 'Generated prices; never eligible for live promotion.'})


class Alpaca:
    """Read-only HTTP client; credentials stay in environment/memory, never in checkpoints."""
    def __init__(self, cancelled=None):
        key, secret = os.environ.get('APCA_API_KEY_ID'), os.environ.get('APCA_API_SECRET_KEY')
        if not key or not secret:
            raise ValueError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY before connecting Alpaca.')
        self.session = requests.Session()
        self.cancelled = cancelled
        self.session.headers.update({'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret})

    def get(self, path, params):
        for attempt in range(5):
            if self.cancelled and self.cancelled():
                raise RuntimeError('Data request cancelled.')
            response = self.session.get('https://data.alpaca.markets' + path, params=params, timeout=(5, 15))
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 16))
                continue
            if response.status_code != 200:
                raise RuntimeError(f'Alpaca returned HTTP {response.status_code}. Check account access and data entitlement.')
            return response.json()
        raise RuntimeError('Alpaca unavailable or rate-limited after retries.')

    def bars(self, symbols, start, end):
        params = dict(symbols=','.join(symbols), timeframe='5Min', start=str(start), end=str(end),
                      adjustment='raw', feed='iex', limit=10000, sort='asc')
        rows = []
        while True:
            page = self.get('/v2/stocks/bars', params)
            for symbol, values in (page.get('bars') or {}).items():
                rows.extend((v['t'], symbol, v['o'], v['h'], v['l'], v['c'], v['v']) for v in values)
            token = page.get('next_page_token')
            if not token:
                break
            params['page_token'] = token
            time.sleep(.32)  # Stay below Basic's historical-request rate limit.
        result = pd.DataFrame(rows, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        if not result.empty:
            result['time'] = pd.to_datetime(result.time, utc=True)
        return result

    def actions(self, symbols, start, end):
        params = dict(symbols=','.join(symbols), start=str(start)[:10], end=str(end)[:10], limit=1000)
        rows = []
        while True:
            page = self.get('/v1/corporate-actions', params)
            for group, values in (page.get('corporate_actions') or {}).items():
                kind = {'forward_splits': 'forward_split', 'reverse_splits': 'reverse_split',
                        'cash_dividends': 'cash_dividend', 'worthless_removals': 'worthless_removal'}.get(group, group)
                for a in values:
                    ratio = float(a.get('new_rate', 1)) / float(a.get('old_rate', 1))
                    rows.append((a['id'], a.get('symbol', a.get('initiating_symbol', '')),
                                 kind, a.get('ex_date', a.get('process_date')), ratio,
                                 float(a.get('cash_rate', 0)), a.get('payable_date', '')))
            token = page.get('next_page_token')
            if not token:
                break
            params['page_token'] = token
            time.sleep(.32)
        return pd.DataFrame(rows, columns=ACTION_COLUMNS).drop_duplicates('id')

    def download(self, folder, symbols, progress=None):
        parts = []
        for year in range(2020, 2026):
            if progress:
                progress(f'Downloading IEX bars for {year}…')
            parts.append(self.bars(symbols, f'{year}-01-01T00:00:00Z', f'{year}-12-31T23:59:59Z'))
        actions = self.actions(symbols, '2020-01-01', '2026-01-31')
        bars = pd.concat(parts, ignore_index=True)
        meta = {'synthetic': False, 'feed': 'IEX', 'symbols': list(symbols),
                'downloaded': pd.Timestamp.now(tz='UTC').isoformat(), 'adjustment': 'raw'}
        # Validate before declaring the download usable.
        market = prepare(bars, actions, symbols, meta)
        if min(market.meta['coverage'].values()) < .70:
            raise ValueError('Less than 70% regular-session coverage for a symbol. Dataset is not eligible.')
        return save_dataset(folder, bars, actions, meta)
