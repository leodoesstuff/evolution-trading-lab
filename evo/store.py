"""SQLite audit log and atomic local checkpoints (never load untrusted checkpoints)."""
import json
import os
import sqlite3
import tempfile
from pathlib import Path
import torch


class Store:
    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.db = self.home / 'experiments.sqlite'
        with self.connect() as c:
            c.executescript('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run TEXT, time TEXT DEFAULT CURRENT_TIMESTAMP,
                    kind TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS diagnostics (
                    run TEXT, generation INTEGER, bot TEXT, payload TEXT,
                    PRIMARY KEY(run,generation,bot));
                CREATE TABLE IF NOT EXISTS metrics (
                    run TEXT, generation INTEGER, bot TEXT, parent TEXT, birth INTEGER,
                    score REAL, net_return REAL, drawdown REAL, alive INTEGER,
                    PRIMARY KEY(run, generation, bot));
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY, run TEXT, bot TEXT, mode TEXT, version INTEGER,
                    time TEXT, symbol TEXT, shares REAL, price REAL, fee REAL);
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY, run TEXT, bot TEXT, mode TEXT, version INTEGER,
                    time TEXT, equity REAL, cash REAL, drawdown REAL);
                CREATE TABLE IF NOT EXISTS holdouts (
                    scope TEXT PRIMARY KEY, run TEXT, used TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS portfolios (
                    run TEXT, bot TEXT, mode TEXT, payload TEXT,
                    PRIMARY KEY(run,bot,mode));
                CREATE UNIQUE INDEX IF NOT EXISTS unique_trade
                    ON trades(run,bot,mode,version,time,symbol);
                CREATE UNIQUE INDEX IF NOT EXISTS unique_equity
                    ON equity(run,bot,mode,version,time);
            ''')

    def connect(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.execute('PRAGMA journal_mode=WAL')
        return c

    def event(self, run, kind, **payload):
        with self.connect() as c:
            c.execute('INSERT INTO events(run,kind,payload) VALUES(?,?,?)',
                      (run, kind, json.dumps(payload, default=str)))

    def metric(self, run, generation, bot, result):
        with self.connect() as c:
            c.execute('INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?,?,?)',
                      (run, generation, bot.id, bot.parent, bot.birth, result['score'],
                       result['net_return'], result['drawdown'], int(result['alive'])))
            if result.get('rules_version',1)>=2:
                c.execute('INSERT OR REPLACE INTO diagnostics VALUES(?,?,?,?)',
                          (run,generation,bot.id,json.dumps(result,default=str)))

    def log_trace(self, run, bot, mode, version, trades, curve, book=None, prices=None):
        if prices is None and book is not None:
            prices = getattr(book, 'last_prices', None)
        with self.connect() as c:
            if book is not None and prices is not None and curve:
                payload = dict(cash=float(book.cash), shares=book.shares.tolist(),
                               prices=prices.tolist(), equity=float(curve[-1][1]), asof=str(curve[-1][0]),
                               receivables=book.receivables, alive=book.alive, version=book.version,
                               analytics=getattr(book,'analytics',{}),rules_version=getattr(book,'rules_version',1))
                c.execute('INSERT OR REPLACE INTO portfolios VALUES(?,?,?,?)',
                          (run,bot,mode,json.dumps(payload)))
            c.executemany('''INSERT OR REPLACE INTO trades(run,bot,mode,version,time,symbol,shares,price,fee)
                VALUES(?,?,?,?,?,?,?,?,?)''',
                [(run, bot, mode, t[5] if len(t) > 5 else version, *t[:5]) for t in trades])
            c.executemany('''INSERT OR REPLACE INTO equity(run,bot,mode,version,time,equity,cash,drawdown)
                VALUES(?,?,?,?,?,?,?,?)''',
                [(run, bot, mode, p[4] if len(p) > 4 else version, *p[:4]) for p in curve])

    def claim_holdout(self, scope, run):
        """A period can be claimed once, including across different datasets/universes."""
        with self.connect() as c:
            c.execute('INSERT OR IGNORE INTO holdouts(scope,run) VALUES(?,?)', (scope, run))
            return c.execute('SELECT run FROM holdouts WHERE scope=?', (scope,)).fetchone()[0] == run

    def save(self, run, state):
        folder = self.home / run
        folder.mkdir(exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=folder, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, folder / 'checkpoint.pt')
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def load(self, run):
        return torch.load(self.home / run / 'checkpoint.pt', map_location='cpu', weights_only=False)

    def save_model(self, run, bot):
        """Archive each live policy version so its exact weights remain inspectable."""
        folder = self.home / run / 'models' / bot.id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'v{bot.version:08}.pt'
        if path.exists():
            return
        fd, temp = tempfile.mkstemp(dir=folder, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(bot.pack(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def request(self, run, action):
        # Separate small control file means UI never overwrites a worker checkpoint.
        folder = self.home / run
        folder.mkdir(exist_ok=True)
        path = folder / 'control.json'
        temp = folder / 'control.tmp'
        temp.write_text(json.dumps({'action': action}))
        os.replace(temp, path)

    def control(self, run):
        path = self.home / run / 'control.json'
        return json.loads(path.read_text())['action'] if path.exists() else 'run'

    def rows(self, sql, params=()):
        import pandas as pd
        with self.connect() as c:
            return pd.read_sql_query(sql, c, params=params)
