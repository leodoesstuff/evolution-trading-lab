"""One accounting/execution engine shared by training, evaluation, and paper trading."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .day_rules import enabled, day_context, summarize, clock_features


@dataclass
class Book:
    cash: float
    shares: np.ndarray
    peak: float
    alive: bool = True
    max_drawdown: float = 0.0
    receivables: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    curve: list = field(default_factory=list)
    applied: set = field(default_factory=set)
    version: int = 0
    analytics: dict = field(default_factory=dict)
    rules_version: int = 1

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.__dict__.setdefault('analytics',{})
        self.__dict__.setdefault('rules_version',1)

    @classmethod
    def fresh(cls, config):
        return cls(config.initial_cash, np.zeros(len(config.symbols)), config.initial_cash,
                   rules_version=getattr(config,'rules_version',1))

    def equity(self, prices):
        if np.any((self.shares > 1e-10) & ~np.isfinite(prices)):
            raise ValueError('Held asset has no usable valuation; trading halted.')
        return float(self.cash + np.dot(self.shares, np.nan_to_num(prices)) +
                     sum(v['amount'] for v in self.receivables))

    def drawdown(self, prices):
        return max(0.0, 1 - self.equity(prices) / self.peak)

    def observe(self, prices, timestamp, config):
        self.last_prices = np.asarray(prices).copy()
        equity = self.equity(prices)
        self.peak = max(self.peak, equity)
        dd = max(0.0, 1 - equity / self.peak)
        self.max_drawdown = max(self.max_drawdown, dd)
        if dd >= config.death_drawdown - 1e-10:
            self.alive = False
        self.curve.append((str(timestamp), equity, self.cash, dd, self.version))
        return equity

    def actions(self, splits, dividends, removed, day, key):
        if key in self.applied:
            return
        self.applied.add(key)
        self.shares *= splits
        for j, cash, payable in dividends:
            amount = float(self.shares[j] * cash)
            if amount > 0:
                self.receivables.append({'amount': amount, 'payable': payable})
        self.shares[removed] = 0
        unpaid = []
        for item in self.receivables:
            if item['payable'] <= day:
                self.cash += item['amount']
            else:
                unpaid.append(item)
        self.receivables = unpaid

    def execute(self, targets, prices, liquidity, timestamp, config, liquidate=False):
        if not self.alive:
            return
        prices = np.asarray(prices, dtype=float)
        usable = np.isfinite(prices) & (prices > 0) & (liquidity > 0)
        equity = self.equity(prices)
        weights = np.clip(np.asarray(targets), 0, config.max_weight)
        if weights.shape != self.shares.shape or not np.isfinite(weights).all():
            raise ValueError('Invalid allocation.')
        desired = np.divide(equity * weights, prices, out=self.shares.copy(), where=usable)
        delta = desired - self.shares
        # Avoid repeatedly paying costs for tiny price drift.
        if not liquidate:
            delta[np.abs(delta * np.nan_to_num(prices)) < equity * .002] = 0
        capacity = np.maximum(np.nan_to_num(liquidity), 0) * config.participation
        delta = np.clip(delta, -capacity, capacity)
        delta[~usable] = 0
        for j in list(np.flatnonzero(delta < 0)) + list(np.flatnonzero(delta > 0)):
            qty = float(delta[j])
            fill = prices[j] * (1 + np.sign(qty) * config.slippage_bps / 10000)
            if qty > 0:
                qty = min(qty, self.cash / (fill * (1 + config.fee_bps / 10000)))
            else:
                qty = max(qty, -self.shares[j])
            if abs(qty) < 1e-10:
                continue
            fee = abs(qty) * fill * config.fee_bps / 10000
            self.cash -= qty * fill + fee
            before_shares=self.shares[j]
            self.shares[j] += qty
            if enabled(config):
                a=self.analytics
                a['notional']=a.get('notional',0)+abs(qty)*prices[j]
                a['fees']=a.get('fees',0)+fee
                a['slippage']=a.get('slippage',0)+abs(qty)*abs(fill-prices[j])
                if before_shares>1e-8 and self.shares[j]<=1e-8:a['roundtrips']=a.get('roundtrips',0)+1
            self.trades.append((str(timestamp), config.symbols[j], qty, float(fill), fee, self.version))
        if self.cash < -1e-6 or (self.shares < -1e-8).any():
            raise AssertionError('Execution violated cash/short constraints.')
        self.cash = max(0.0, float(self.cash))


def observation(market, i, book):
    prices = market.marks[i]
    equity = max(book.equity(prices), 1e-8)
    weights = book.shares * np.nan_to_num(prices) / equity
    return np.concatenate([market.features[i],clock_features(market.times[i]) if getattr(book,'rules_version',1)>=2 else [], weights, [book.cash / equity, book.drawdown(prices)]]).astype(np.float32)


def allocation(actions, book, prices, config):
    current = book.shares * np.nan_to_num(prices) / max(book.equity(prices), 1e-8)
    return np.where(np.asarray(actions) == 3, current, np.asarray(actions) * config.max_weight / 2).clip(0, config.max_weight)


def step(market, i, book, targets, config, force_close=False, hold=False):
    """Decision was made at bar i-1 close, execution is at bar i open."""
    day = str(market.times[i].tz_convert('America/New_York').date())
    book.actions(market.splits[i], market.payable.get(i, []), market.removed[i], day, str(market.times[i]))
    # Observe overnight gaps before allowing any new exposure.
    prices = np.where(np.isfinite(market.opens[i]), market.opens[i], market.marks[i - 1] / market.splits[i])
    prices[market.removed[i]] = 0
    book.observe(prices, market.times[i], config)
    if enabled(config):
        _,stamp,closing=day_context(market.times[i])
        record=book.analytics.setdefault('days',{}).setdefault(day,dict(start=book.equity(prices),end=book.equity(prices),active=False,halted=False))
        if not record['halted'] and book.equity(prices)/max(record['start'],1e-8)-1<=-config.max_daily_loss:
            record['halted']=True
            book.analytics['loss_stops']=book.analytics.get('loss_stops',0)+1
        exiting=force_close or record['halted'] or (closing is not None and stamp>=closing-pd.Timedelta(minutes=10))
        if book.alive and (exiting or not hold):
            book.execute(np.zeros(len(config.symbols)) if exiting else targets,prices,market.volumes[i],market.times[i],config,liquidate=exiting)
        after=book.observe(market.marks[i],market.times[i]+pd.Timedelta(minutes=5),config)
        exposure=float(np.dot(book.shares,np.nan_to_num(market.marks[i]))/max(after,1e-8))
        record['end']=after
        record['active']=record['active'] or exposure>=config.min_mean_exposure
        a=book.analytics;a['bars']=a.get('bars',0)+1;a['exposure_sum']=a.get('exposure_sum',0)+exposure
        a['active_bars']=a.get('active_bars',0)+int(exposure>=config.min_mean_exposure)
        if (force_close or (closing is not None and stamp+pd.Timedelta(minutes=5)>=closing)) and np.any(book.shares>1e-8):
            a['flat_failed']=True;book.alive=False
        return after
    if book.alive and not hold:
        book.execute(targets, prices, market.volumes[i], market.times[i], config)
    return book.observe(market.marks[i], market.times[i] + pd.Timedelta(minutes=5), config)


def results(book, config):
    values = np.array([r[1] for r in book.curve])
    net = (float(values[-1]) if len(values) else config.initial_cash) / config.initial_cash - 1
    # Four equally sized chronological segments penalize profits concentrated in one segment.
    chunks = np.array_split(values, min(4, len(values))) if len(values) else []
    previous = config.initial_cash
    returns = []
    for chunk in chunks:
        returns.append(float(chunk[-1]) / previous - 1)
        previous = float(chunk[-1])
    score = net - 2 * book.max_drawdown - (float(np.std(returns)) if returns else 0)
    result=dict(net_return=net,drawdown=book.max_drawdown,
                score=score if book.alive else -1000+score,alive=book.alive,
                trades=len(book.trades),final_equity=float(values[-1]) if len(values) else config.initial_cash)
    if enabled(config):
        result.update(summarize(book,config,net))
        parts=result['score_parts']
        result['score']=parts['profit']-parts['drawdown_penalty']-parts['consistency_penalty']
        if not book.alive:result['score']-=1000
    return result


def benchmark(market, start, end, config, interrupted=None):
    """Equal allocation on the first executable bar, then hold, including drawdowns."""
    import copy
    config=copy.copy(config);config.rules_version=1
    book = Book.fresh(config)
    book.observe(market.marks[start], market.times[start], config)
    step(market, start + 1, book, np.full(len(config.symbols), config.max_weight), config)
    for i in range(start + 2, end + 1):
        if interrupted and i % 64 == 0 and interrupted():
            return None, book
        day = str(market.times[i].tz_convert('America/New_York').date())
        book.actions(market.splits[i], market.payable.get(i, []), market.removed[i], day, str(market.times[i]))
        book.observe(market.marks[i], market.times[i] + pd.Timedelta(minutes=5), config)
    return results(book, config), book
