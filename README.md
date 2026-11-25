# Evolution Trading Lab

A local trading research app: small neural-network bots learn during their lives,
compete on historical data, and reproduce through mutated weights. Drawdown kills
weaker bots. Qualified champions automatically start local paper trading against
live IEX data, where learning continues. No brokerage orders are submitted.

## Open the app

Double-click **Launch Trading Lab.command** on your Mac. Dependencies are already
installed in this workspace. Open http://localhost:8501 if a browser does not open.

1. Click **Create demo data**.
2. For a first look, select **5 or 10 bots** and **1 or 2 generations**.
3. Click **Start experiment**. The screen updates every ten seconds.
4. Choose a completed bot/session to view its equity, stock prices, and trades.

Demo prices are generated, visibly labeled, and cannot promote to live paper
trading. The full default is 100 bots × 100 generations and will require multiple
sessions. Session budgets are wall-clock limits, not a promise that an experiment
finishes within that time. Pause/resume preserves weights, optimizer state, lineage,
random-generator state, and the experiment cursor. An interrupted bot evaluation
is discarded and rerun from its preceding checkpoint.

On another machine, install Python 3.11 or 3.12, create a virtual environment, then:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

## Connect real data

Create a free Alpaca account at https://app.alpaca.markets/signup and obtain paper
API keys. Enter them in the sidebar, or set `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` in the environment before launching. Keys entered through
the app remain in memory and are inherited by its worker processes. They are
never saved in reports, checkpoints, or logs. Restarting the app clears them.

Click **Download 2020–2025 IEX history**. This can take several minutes and requires
internet access and the relevant free account entitlements. The app uses only GET
requests to Alpaca's market-data service. It does not use the trading/order API.

Free IEX prices cover one exchange, not the consolidated market. Missing bars
are kept as missing; only valuation marks carry forward. Corporate-action access
is required. Unhandled mergers, spin-offs, or symbol changes block a dataset
instead of silently inventing its accounting. Historical download success and live
account access must be checked using your own account; they cannot be verified
without credentials.

## Experiment rules

| Setting | Default |
|---|---|
| Universe | AAPL, MSFT, AMZN, JPM, XOM |
| Starting virtual money | $10,000 per life / paper portfolio |
| Actions, per stock | 0%, 10%, 20%, or hold current allocation |
| Trading | Five-minute regular-session bars; exit before close (v2) |
| Borrowing / shorting | Disabled |
| Death | At least 15% below that portfolio's highest observed value |
| Lifetime learning | One-step advantage actor-critic, small 32×32 network |
| Reproduction | 20% elite slots, 60% mutated children, remaining slots fresh; eligible parents only (v2) |
| Mutation | Gaussian noise on 10% of weights; children reset optimizer state |
| Training | Random 780-bar episodes, 2020–2023 |
| Ranking | Three common 390-bar windows; return minus drawdown and inconsistency |
| Validation | Top five training snapshots, full 2024, learning disabled |
| Final test | One validation champion, full 2025, learning disabled |
| Promotion | Positive validation/test returns; activity and day-exit checks (v2); drawdown below 15%; adequate coverage; unused final period |
| Fees / slippage | 1 / 5 basis points per fill, configurable in `Config` |

Training-window returns are averaged for the leaderboard; they are not annual
returns. Ranking is the median window score. Losses are measured at observed
opens and closes; a five-minute bar does not establish the exact intrabar drawdown.
The 20% cap constrains requested allocations; price drift can move an existing
position above it until the next tradable rebalance.

Historical signals use completed bars and fill no earlier than the next bar's
open. Fills are capped at 1% of that bar's realized volume, an approximate
participation model rather than a reconstruction of an exchange order book.
Fractional shares are permitted. Split ratios change share counts on the ex-date;
cash-dividend entitlements enter equity on the ex-date and become spendable on
the payable date. Worthless removals zero holdings. Features use causal total-return
changes and a scaler fitted only on training data.

The fixed list contains companies selected today; it is an educational prototype
with survivor-selection bias. Results are not a test of the entire stock market.
Reports compare the champion with holding cash and buying the same stock basket,
using the same dividends, splits, and initial execution costs.

## Live paper behavior

Passing all checks automatically starts a **new** $10,000 local portfolio, not
the historical test portfolio. Decisions use completed five-minute bars; queued
orders execute on a subsequent poll using fresh present-time bid/ask quotes plus
slippage. Quote size limits simulated fills. Missing/stale quotes block all fills.
Orders expire after 60 seconds and are discarded after outages or worker restarts.
Historical prices are never used to replay missed live fills.

The bot learns from the next observed live transition. Every fill records the
model version responsible for it; each learning update creates a new version.
The paper bot's model and optimizer are saved separately from the frozen champion.
Training replacements need their own qualification and paper portfolio.
Each live model version is archived under the run's `models/` folder, including
its exact weights and optimizer state.

Your Mac and the worker must stay awake and connected. Session expiry pauses the
worker; Resume starts another bounded session. Emergency Stop discards orders and
stops the experiment; it does not invent liquidation prices. Eliminated portfolios
stay eliminated. Corporate-action providers can publish late or incomplete data;
local paper simulation is not a broker's authoritative account ledger.

## Reports, tests, and command line

Run these from this folder with the installed virtual-environment Python:

```sh
python run.py demo
python run.py new --dataset demo --population 10 --generations 2
python run.py run RUN_ID --minutes 15
python run.py control RUN_ID pause
python run.py export RUN_ID
python -m pytest -q
```

The `data/` directory holds Parquet bars/actions and fingerprinted metadata.
`runs/experiments.sqlite` stores lineage, scores, versioned trades, equity, events,
and final-test usage. Each run saves an atomic `checkpoint.pt`, `report.json`, and
worker log. Only open checkpoints created by this app; PyTorch checkpoints are
not an interchange format for untrusted files. `EVO_HOME` can relocate runtime data.

A final-test period is marked used before evaluation. Reusing it in another real
experiment blocks automatic promotion even if you change the stocks or settings.
Synthetic experiments have a separate usage ledger. Deleting the runtime database
also deletes this protection. A completed experiment may produce no qualifying bot.

Design references: [Alpaca bars](https://alpaca.markets/sdks/python/api_reference/data/stock/requests.html),
[corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1),
[PyTorch policy gradients](https://docs.pytorch.org/docs/stable/distributions.html),
[backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## Trading desk and bot pages

Use **Market** for a candlestick workspace with a stock watchlist, volume, and
buy/sell markers. Scroll over the chart to zoom, drag to select a region, use the
pan tool to move around, and double-click to reset. The bottom range slider and
5-minute / 15-minute / 1-hour candles provide additional ways to explore.
Chart times use New York time. Turn off **Live updates** to inspect a chart
without refreshes; **Refresh results** fetches the latest saved bot state.

**Bots** lists the traders, including ones still waiting for their first session.
Each **Open portfolio** link opens a personal bot page with its own URL. The
portfolio shows shares, prices, allocation, cash, dividends awaiting payment,
balance history, and downloadable trades. Choose a session explicitly: each
learning life and practice test starts a separate $10,000 account. Old experiments
reconstruct positions from their trade records and split/removal history; new
experiments save exact portfolio snapshots. These are end-of-session holdings,
not a continuously updating view of an unfinished historical episode.

**Learning speed** is available in the sidebar for new experiments. For an existing
experiment, pause it, expand **Training speed explained**, choose a speed, and
click **Apply speed**, then Resume. Standard updates after each bar, Fast batches
8 transitions, and Turbo batches 32. Every five-minute trading step and risk check
still runs; batching changes the learning trajectory and does not guarantee the
same results. Historical training has no real-time playback delay. Live paper
trading remains tied to the real market clock and continues updating per live bar.

A local 390-bar synthetic benchmark took 0.271 s in Standard, 0.102 s in Fast,
and 0.080 s in Turbo. These are training-episode measurements, not promises for
whole experiments or other machines. Old checkpoints load with Standard speed
and can be changed while paused. Final-test usage protections remain unchanged.

## Follow the current leader

Turn on **Follow best bot** above the workspace to automatically update every ten
seconds. Market and Bot profile follow the leader's saved portfolio and trades.
**Highest return** compares surviving bots in the latest generation with results,
using their average return after costs over the same practice windows. Choose
**Risk-adjusted score** to include drawdown and consistency in the ranking.
A qualified paper bot takes precedence once live practice starts.

Historical portfolios update after a bot's test completes, not during an unfinished
life. No completed survivor means the view waits for one. An eliminated paper bot
is not replaced with an old training result. Turn follow off to resume manual bot
and session selection. This is a read-only view: it does not change training,
promote a bot, reset an experiment, or rewrite saved data.

## Live room: independent practice and competitions

Open **Live room** in the Workspace selector. Pick one or up to ten saved bot
snapshots from the selected experiment, choose whether learning continues, then
click **Start live practice**. Population weights, the best saved snapshots, the
frozen champion, and the latest automatic-paper model are offered when available.
An older bot with no saved weights cannot be launched from its ID alone.

Every manual session is labeled **Experimental**, even if a snapshot failed
qualification or learned on synthetic data. This never promotes or re-qualifies
it, and never touches the final-test ledger. All selected bots receive fresh
$10,000 portfolios, one shared quote/bar feed, identical timestamps, and the same
fee, spread, slippage and liquidity assumptions. Learning changes only practice
copies. A buy-and-hold basket and cash benchmark start with the same balance.
The basket makes one quote-size-limited initial purchase, then holds and receives
the same corporate actions. Cash earns no interest. Results are after execution
costs; displayed fees alone exclude spread and slippage.

The room shows market hours, quote age, connection health, live prices, competition
rankings, holdings, versioned trades, candlesticks, and comparative return charts.
Refresh runs every ten seconds and can be disabled to inspect chart zoom. Saved
trades, comparison history, and a session report can be downloaded. Quote polling
is every five seconds; decisions use completed five-minute bars. No future or
unfinished bars enter the models. Current free data is IEX-only, not consolidated
US-market data.

Pause, Resume, Stop, and Emergency Stop affect only that practice session. Resume
keeps its balances, holdings, model versions, benchmark, and start time. The Mac
must remain awake. Session budgets pause the worker; they do not reset it. Markets
closed means waiting with no new orders. Stale data/outages discard queued orders;
no past fills are replayed on recovery. Orders expire after sixty seconds.
Drawdown-eliminated entrants stop trading and freeze their recorded competition
value and portfolio snapshot; the buy-and-hold benchmark continues through its
own drawdowns. Unsupported corporate actions block new decisions for reconciliation.

Practice runs live in **`practice/`**, with their own SQLite database, checkpoints,
logs, reports, model versions, and recent chart bars. Training `runs/` and original
`data/` are read-only inputs to this feature. Creating another practice session
always uses a new ID; it cannot overwrite an earlier practice account.

This remains a local simulator. **Alpaca paper-order submission is deferred**;
there are no order, cancellation, live-account, or other brokerage writes in the
practice engine. Only read-only market-data requests are made.

Dashboard performance: bot/session selectors read the small portfolio index instead
of scanning the raw trade/equity archive. The bot directory shows 25 cards per page.
Historical records remain intact; legacy runs use their metrics to build selectors.

## Day-trader v2 and transparent scoring

New experiments started from the app or CLI use **day-trader v2**. Old saved runs
keep **legacy overnight rules**; their checkpoints, scores, and history are not
rewritten or silently upgraded. Starting a new experiment uses fresh networks with
two additional clock inputs (session progress and time remaining), so old 42-input
models and new 44-input day-trader models remain distinct.

V2 starts exiting ten minutes before each actual exchange close, including early
closes, and stops opening positions in that exit window. Historical episode endings
also request liquidation. Exits pay fees and slippage, respect available liquidity,
and ignore the small-order deadband. A position that cannot be exited by the close
makes that evaluation ineligible; no closing price or liquidity is invented.
Live v2 bots use fresh quotes for exits and are marked failed if an outage leaves
them holding overnight. Legacy bots and the buy-and-hold comparison retain their
original holding behavior.

A daily loss of 3% from that day's opening portfolio value triggers liquidation and
blocks further entries for the day. This is a trigger, not a guaranteed maximum
loss: price jumps, missing quotes, or insufficient liquidity can produce larger
losses. The original 15% overall drawdown elimination rule still applies.

Each practice window requires at least 5% mean equity exposure, meaningful exposure
on at least 40% of its trading days, and at least three completed position round
trips (reduced to the number of days for shorter windows). Every common window must
pass. Tiny purchases and repetitive low-exposure trades do not satisfy these gates.
Activity is never added to the reward as a trade-count bonus. New experiments use
`active_survivors` selection: active bots within the risk and exit limits can keep
learning and reproduce while still unprofitable. The strongest 20% (up to available
survivors) retain their learned weights and optimizer; mutated descendants and fresh
bots fill the other slots. Profit remains part of the ranking, so smaller losses
can be preferred without declaring them winners. The hall of fame and champion
qualification still require positive profit. Saved experiments without this setting
retain the original positive-profit reproduction gate after three warmup generations.

The v2 score is **net return − 0.5 × maximum drawdown − daily-return standard
deviation**, averaged over the common evaluation windows. Activity and successful
liquidation are separate eligibility gates, so a cash-only zero score cannot earn
reproduction rights or become a v2 winner. These configurable research thresholds
are initial defaults, not an assurance of future profitability.

Bot pages show net profit, current exposure, mean exposure, completed round trips,
active days, daily profit/loss, fees, modeled slippage, daily loss stops, and a score
breakdown. The portfolio's figures describe its selected session; the evaluation
score report combines all practice windows. Older runs show “Not recorded” for
statistics they never collected. New diagnostic rows are stored in a separate,
small indexed table; the dashboard does not scan the raw historical trade archive.

## Clear bot statuses

**Champion** means the saved winning snapshot passed validation and an untouched
final test. A selected finalist that failed qualification is never labeled Champion.
The completed run shows a champion banner and a direct portfolio link.

**Training** marks the bot currently being processed; **Waiting** marks queued
bots. **Testing** marks a validation/final exam in progress, and **Paused** means
the bot's worker is not currently processing it. **Survived** means a completed
life stayed within the risk limits; it does not mean the bot won or is still
training. **Inactive** identifies a risk survivor that failed the v2 participation
gates. **Eliminated** means a risk or exit rule failed in its latest evaluated life.
All these labels are computed for display without rewriting saved bot states.

Live-practice status labels follow the session and worker state: **Learning live**,
**Practicing live**, **Waiting**, **Paused**, **Stopped**, **Finished**, or
**Eliminated**. An old stored `active` flag cannot make a paused session look live.
# Historical practice coverage

**Follow best bot** displays the strongest surviving bot that meets the activity
and risk rules in the most recent generation with an eligible completed evaluation,
even if its return is negative. While newer tests have no eligible survivor, the
previous saved survivor stays visible with its generation explicitly labeled.
A losing leader is shown with a warning and its actual average evaluation return;
viewing it does not admit it to reproduction or qualification. The view reports
completed evaluations, survivors, bots meeting activity/risk rules, and profitable
candidates separately. Empty views distinguish running training, paused training,
and finished experiments. Viewing a bot does not change its training policy; the
hall of fame and champion qualification always require positive profit.

New experiments use hourly strategy decisions to reduce excessive turnover during
exploration. Advanced settings allow 5-, 15-, 30-, or 60-minute decisions. Holdings
remain unchanged between decisions, while valuation, daily loss stops, and closing
exits still run at every historical five-minute bar (and each fresh live quote).
Learning assigns the full holding-period reward to the decision that produced it,
with discounting for its actual duration. Historical evaluation and live practice
use the same decision cadence. Costs and activity/profit qualification rules are
unchanged. New practice periods start and end at exchange-session boundaries to
avoid artificial forced exits in the middle of a session. Early closes count as
whole sessions. Missing data or insufficient exit liquidity can still eliminate a
bot; no fill is fabricated. Existing checkpoints without a cadence setting retain five-minute behavior;
continuations create separate checkpoints. New experiments stop with an explicit
extinction result if every bot is eliminated and there is no saved candidate,
instead of continually replacing the whole population. Survival does not imply profit.

New experiments using real history default to **Broad**: each bot learns on a random
4,680-bar period (about 60 regular trading days), then is evaluated on six
1,560-bar periods (about 20 days each) spread across the available 2020–2023
training history. Quick retains 10 learning days and three 5-day evaluations;
Deep uses 120 learning days and eight 40-day evaluations. Days are 78-bar
regular-session equivalents; early closes change the actual session count.
Bots that hit the risk limit stop early. Longer coverage requires more compute
and does not guarantee better results. Validation stays in 2024 and the final
historical test stays in 2025; the longer practice periods never use these years.

Choose **History per generation** when starting an experiment. The short synthetic
demo defaults to Quick. CLI: `python run.py new --dataset alpaca-iex --coverage Broad`.
Saved experiments retain their original coverage. To retain existing learned bots,
pause a training run and use **Give these bots more historical practice → Continue
bots with more history**. This creates a separate experiment with the current
weights, optimizer state, lineage IDs, RNG, and generation retained. Its shorter-period
scores and hall of fame are reset for fair reevaluation; the source checkpoint,
portfolios, trade history, and results stay intact. The session time budget remains
independent of historical days: Resume continues a paused session.

## Champion seeds and readable names

Choose **Starting bots → Last champion** to create a separate experiment from the
most recent qualified champion. The default population contains one transferred
copy, 79 mutated descendants, and 20 fresh bots. The original checkpoint and its
results stay intact. A legacy model's portfolio inputs are moved to their new
positions when session-clock inputs are inserted; their initial weights and Adam
moments are zero. The other weights, optimizer state, and root model version are
retained. New rules and evaluation scores start fresh at generation one.

A champion selected using the old validation and test years brings prior selection
information into this research. Those years are not a fresh holdout. The existing
holdout-use registry continues to block automatic promotion on a reused period.

Bot names are now readable names such as **Vega Juniper Stone**. Names are stable
display labels; raw bot IDs stay in profile details, search, exports, and storage.
No saved identifiers, portfolios, or trade histories are renamed.

