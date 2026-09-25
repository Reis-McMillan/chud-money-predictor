# chud-money-predictor

Multivariate [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch) forecasts of the
**Kalshi BTC 15-minute contract price** (KXBTC15M), from the contract's own candles and the BRTI
(CME CF Bitcoin Real-Time Index) that settles it. Data comes from the chud-money API, which
fronts the production QuestDB.

Pipeline: `download` (two raw tables → one Parquet per UTC day) → `resample` (1-minute BRTI bars) →
`frame` (bars joined with the active contract's candle) → `backtest-contract` (at every minute of
every window, forecast the price at every remaining minute; score against persistence) →
`finetune` (post-train TimesFM on the same samples).

## Data

Two QuestDB tables, served by the chud-money API (`https://api.chud-money.mcmlln.dev`, market
`btc-15m`) as Server-Sent Events, one request per UTC day:

| source | endpoint | table | what |
|---|---|---|---|
| `brti` | `GET /btc-15m/data/index-hist` | `index_values_hist` (`index_id='BRTI'`) | index ticks from 2025-09-17. One per second until about May 2026, five per second after. |
| `contracts` | `GET /btc-15m/data/candles` | `contract_candles_hist` (`series_ticker='KXBTC15M'`) | 1-minute candles from 2025-12-10: YES bid/ask OHLC, trade OHLC + mean, volume, open interest, `floor_strike`. One market per 15-minute window, 15 candles each. **Candle `ts` is the END of its minute.** |

The market filter is applied server-side from the tag; the client asserts the stream's `meta`
event names the expected table, key and columns. `GET /btc-15m` (public) reports each table's row
count and time bounds, which is where `chudp download` takes its snapshot from. The API allows two
concurrent exports in total (the SPA may hold one), so `download` runs one stream at a time by
default (`--jobs 2` is safe: a 429 is retried with backoff).

A day is recorded complete when the stream ends with its `done` event and the server's row count
matches what arrived; that proves the transfer, not QuestDB's contents. There are no per-day counts
any more, so a sealed day rewritten server-side is not noticed: `download --force` (with
`--start/--end`) refetches. A day of BRTI is about 78 MB on the wire (uncompressed), the whole
history about 32 GB, so copying `data/raw/` between machines is still the fast way to move history.

## Auth

The data stream only accepts a five-minute Verys access token exchanged for the API's own
audience, from an identity holding the `chud-money` role. `chudp` is the same public PKCE client
as the SPA, driven headlessly:

```bash
uv run chudp auth login      # email, then the 6-digit code Verys mails you (valid 5 minutes)
uv run chudp auth status     # identity, refresh-token age, and a live token exchange
uv run chudp auth token      # print a token, e.g. for curl -H "Authorization: Bearer $(...)"
uv run chudp auth logout     # revoke our refresh token and delete the session file
```

`login` writes `.auth/session.json` (chmod 600, git-ignored): the rotating refresh token **and** the
60-day Verys browser cookie. **Treat that file as a password.** From then on every request takes a
token from a thread-safe provider that re-exchanges 30 s before expiry and persists each rotated
refresh token atomically. Signing out of the SPA in your browser revokes every refresh token of the
client, ours included; `chudp` then mints a new one from the saved cookie without asking for a code.
Only when the cookie is gone too does it say `run chudp auth login`. A 403 from the API means the
identity lacks the role (an admin adds it with `POST {verys}/roles/chud-money/identities/<email>`).
Run `auth login` separately on each machine rather than copying the file: sessions are independent.

## Setup

```bash
uv sync --extra cpu --extra viz --group dev     # laptop (macOS: CPU, MPS works too)
uv sync --extra rocm --extra viz --group dev    # MI300X: torch from the ROCm index (see pyproject)
```

`cpu` and `rocm` are mutually exclusive extras. The ROCm index version in `pyproject.toml` must
match the box's driver stack; change it and run `uv lock`. The first model load downloads ~1.3 GB
into `models/` (`HF_HOME`).

## Run

```bash
uv run chudp api info
uv run chudp download --source both --start 2026-09-13 --end 2026-09-16   # no range = all history
uv run chudp resample --freq 1m
uv run chudp frame                                   # prints the alignment / strike audit
uv run chudp backtest-contract --config configs/contract_smoke.toml      # 1 day, context 512, CPU
uv run chudp backtest-contract --config configs/contract_full.toml       # all history, GPU
uv run chudp rescore <run_id> --require-quote-ok                          # no model needed
uv run chudp finetune --config configs/finetune.toml
```

Any flag overrides the config file. A backtest writes `data/backtests/<run_id>/`: `summary.txt`,
`forecasts.parquet` (one row per window × origin minute `m` × step `h`, with median, deciles,
realised price and both baselines), `metrics_*.parquet`, `bootstrap.parquet`, `baselines.parquet`
(the persistence fan), `meta.json` and `plots/`.

## How it is defined

**Bars.** Closed-left, labelled by their START. Every statistic is computed on the on-the-second
ticks only: that is what Kalshi samples and what the older history already is, so the bars do not
change when the tick density does. `n_raw_ticks` is a diagnostic, never a model input. Three
statistics look inside the minute: `rv_1s`, the realized variance (sum of squared one-second log
returns, scaled to a full bar and null with fewer than half the bar's returns, so a sparse minute is
never a calm one), and `ret_l10` / `ret_l30`, the log return over the bar's last 10 / 30 seconds. A
return belongs to the bar of its later tick; one across more than a minute or a UTC midnight is dropped.

**One grid, one rule.** Contract candles are re-labelled to `ts − 1m`, so for both sources *the
row labelled L is fully observable at wall clock L + 1m*. A window opening at T0 owns rows
T0 .. T0+14m (`k = 0..14`); row `k = 14` holds the settlement candle and the BRTI bar whose `mean`
settles the market. `chudp frame` audits this: the price's outcome and the index's outcome must
agree in every window, and Kalshi's `floor_strike` must equal the BRTI mean of the minute before
the window (it does, to half a cent). Null strikes are imputed from that identity and flagged.

**Origins.** Every bar is the context end of exactly one origin. With `k` the row's minute in its
window: `m = (k+1) mod 15` minutes since the window opened, context = the `context` rows ending
there, `15 − m` steps remain, step `h` is row `i + h`, and the last step is the settlement candle.
`m = 0` is an origin too: at T0 the strike (the mean of the minute that just ended) and all of BRTI
are known. The model always emits a 64-step patch; steps past expiry belong to the next contract and
are dropped everywhere.

**Inputs (25 variates).** Target: `mid_close = (yes_bid_close + yes_ask_close) / 2`, in dollars.
17 past-only covariates: BRTI log close, 1-minute return, range, trailing vol, mean-vs-close,
moneyness `log(brti/strike)`, its vol- and time-scaled z and fair value `Φ(z)`; contract bid, ask,
spread, mid high/low, within-contract mid change, last trade (mid where none), log volume, log open
interest. 7 known-future covariates: the window clock (`k/14`, sin/cos), time to expiry, time of
day, and the fair-value path "if BRTI stays where it is" (from origin-time information only; the
`no_fair_path` preset removes it). Missing values are interpolated as TimesFM would, and any 32-point
input patch that is exactly flat (a one-cent spread for half an hour) gets a fixed jitter of 1% of a
price tick, because TimesFM's running statistics divide by a sigma that is 0 for a flat series.

**The one-second preset, `full_rv` (29 variates).** The model stays on the 1-minute grid (the contract
only has 1-minute candles); what the ticks add is summarised per minute. `sigma_rv` is the per-minute
sigma from the mean `rv_1s` of the last 30 bars (`chudp frame --rv-lookback`): one-second returns make
a window that short precise, and a short window follows the vol regime, which the 240-bar `sigma_1m`
cannot. `full_rv` swaps trailing vol, z and `Φ(z)` for their `sigma_rv` versions and adds the regime
ratio `log(sigma_rv15 / sigma_rv240)`, the mispricing `Φ(z) − mid`, and the last 10 / 30 seconds'
index return in sigmas (21 past-only covariates); its fair-value path is built on `sigma_rv` too.
`full`, the origin set (still gated on `sigma_1m`) and the index-only baseline are unchanged, so the
two presets are scored on identical rows. `scripts/analysis/signal_check.py` is the small-model
diagnostic behind this preset (vol forecast quality, gap → next mid change, trade P&L after costs).

**Scoring.** Errors are in cents. The mid is close to a martingale, so absolute error means little;
read **`skill_mse = 1 − MSE / MSE(persistence)`** and **`skill_pinball = 1 − pinball /
pinball(persistence fan)`** on identical rows, with their day-block bootstrap intervals.
Persistence is the last observed mid, except at the window open, where the last mid belongs to the
contract that just settled at 0 or 1 and the reference is 0.50. Squared error (of the mean of the
model's deciles) is the headline because a price that ends at 0 or 1 is bimodal: absolute error is
minimised by shouting the likelier extreme, which rewards overconfidence rather than information.
`skill_mae` is still reported, median against median. The persistence fan is the empirical
distribution of the change in the mid by (minute, step, 10-cent price level), fitted before the
evaluation period (the summary says so when it cannot be). A second baseline is the index-only
fair value. The settlement candle is also scored as a probability (Brier / log loss
against the 0/1 outcome, versus the market's own last price). Origins whose quote is degenerate
(spread > 10 cents, common in the first illiquid weeks) are flagged, not dropped: `chudp rescore
<run_id> --require-quote-ok` gives the filtered numbers.

## Fine-tuning

`chudp finetune` trains on exactly the backtest's samples and input arrays
([finetune.py](src/chud_predictor/finetune.py)).

- **Split.** Chronological on UTC midnights: train, validation, test, 70 / 15 / 15 of the contract
  era by default (`--val-start` / `--test-start` pin the dates). An origin belongs to a split only if
  all of its target rows do, and each split ends `embargo_bars` (one day) before the next begins.
  Train drives the gradient steps, validation picks the checkpoint and stops early, test is scored
  once at the end.
- **Loss.** Pinball loss on the nine deciles in price space, over the steps to expiry only
  (`--loss-scale fan` divides each step by the persistence fan's width instead).
- **Same path as inference.** Training calls the function `predict` runs, with the same covariates
  and gradients enabled; evaluation is fp32 with clipped, sorted quantiles.
- **Outputs** in `data/finetune/<run_id>/`: `best/` and `last/` checkpoints, `baselines.parquet`
  (persistence fan fitted on the train split only), `splits.json`, `config.json`, `train_log.jsonl`,
  `summary.txt`. The summary prints the two `backtest-contract` commands that score the tuned and
  the zero-shot model on the held-out dates with that same fan. The backtest warns if a checkpoint
  is used with a different context or covariate preset than it was trained with.

Memory scales with tokens per pass, `micro_batch × variates × (context/32 + 2)`. On ROCm 7.2 /
torch 2.14 the fp32 backward pass segfaulted above roughly 12.5k tokens, so at 25 variates keep
`micro_batch ≤ 8` at context 1024 (≤ 4 at 2048) and grow `batch_size` instead; the same limits hold
for the 29 variates of `full_rv` (986 tokens per sample at context 1024).

## Tests

```bash
uv run pytest -q                      # offline: fake chud-money API (SSE) + fake Verys, synthetic ticks + candles, stub / tiny models
CHUDP_SLOW=1 uv run pytest -m slow    # loads the real weights
uv run ruff check .
```

## Caveats

- Persistence is hard to beat. If the skill intervals straddle zero, the honest reading is that the
  market is efficient at this horizon.
- About 26k contracts over nine months is little data for fine-tuning, and origins inside a window
  are near-duplicates: try `--trainable last:4` before `all`, and trust only the day-block intervals.
- TimesFM 3.0 weights are under the **TimesFM Non-Commercial License v1.0**. The Apache-2.0 2.0
  checkpoint has no variate attention, so this covariate design does not port to it.
- Context is capped at 15,360 rows by the model.
