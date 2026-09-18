# chud-money-predictor

Zero-shot [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch) forecasts of the BRTI
(CME CF Bitcoin Real-Time Index), pulled from the production QuestDB and backtested as a trader of
the Kalshi 15-minute BTC market (KXBTC15M).

Pipeline: `download` (raw 1-second ticks → one Parquet per UTC day) → `resample` (1-minute bars) →
`backtest-kalshi` (a forecast at every minute of every 15-minute window, scored against the real
settlement) → `rescore-kalshi` (re-score the same forecasts under other strike / trading rules).

## Data

QuestDB table `index_values_hist` (there is no `historical` table): `index_id='BRTI'`, one row per
second, designated timestamp `ts`, partitioned by day, dedup on `(ts, index_id)`. It starts on
2025-09-17 and a backfill is still appending; the downloader is incremental and idempotent, so just
re-run it.

The DB is cluster-internal (namespace `questdb`, no ingress). Access from a laptop:

```bash
make creds          # kubectl secret questdb-secrets/DB_PASS -> .env (git-ignored, chmod 600)
make pf             # kubectl port-forward svc/questdb 19000:9000 (pidfile in .run/)
```

On a box without kubeconfig (the MI300X), copy `.env.example` to `.env` and paste the password
(Oracle Vault key `qdb-password`), then either `ssh -L 19000:localhost:19000` from a machine with the
port-forward, or copy `data/raw/` over and skip the download entirely.

## Setup

```bash
uv sync --extra cpu --extra viz --group dev     # laptop (macOS: CPU, MPS works too)
uv sync --extra rocm --extra viz --group dev    # MI300X: torch from the ROCm index (see pyproject)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`cpu` and `rocm` are mutually exclusive extras. The ROCm index version in `pyproject.toml` must
match the box's driver stack; change it and run `uv lock`. The first model load downloads ~1.3 GB
into `models/` (`HF_HOME`).

## Run

```bash
uv run chudp qdb info
uv run chudp download --start 2025-09-18 --end 2025-09-20   # or no range for all history
uv run chudp resample --freq 1m
uv run chudp backtest-kalshi --config configs/kalshi_smoke.toml   # 1 day, ctx 512, CPU, ~10 s
uv run chudp backtest-kalshi --config configs/kalshi_full.toml    # all history, ctx 4096, GPU
uv run chudp rescore-kalshi <run_id> --strike-mode open_avg60 --tau 0.55 --price 0.5
uv run chudp forecast --context 4096 --plot                       # one generic 64-step forecast
```

Any flag overrides the config file. Outputs land in `data/backtests/<run_id>/`: `summary.txt`,
`forecasts.parquet` (one row per window × minute with median, deciles, strike, settlement, P(up)),
`trades.parquet`, `metrics_*.parquet`, `calibration.parquet`, `bootstrap.parquet`, `meta.json`,
and `plots/`.

## How the backtest is defined

- Windows open at HH:00/15/30/45 UTC. **Settlement** is Kalshi's rule: the mean of the once-per-second
  index over the final minute, i.e. the `mean` of the 1-minute bar labelled T0+14m. The raw data is
  that index, so settlement is exact.
- **Strike** defaults to the BRTI tick at window open (`open_tick`); `open_avg60` (the previous
  window's settlement) is the alternative and is reported as a sensitivity row.
- At minute `m = 0..14` the model sees 1-minute bars ending strictly before T0+m and forecasts the
  bar labelled T0+14m (step `14-m` of one 64-step forecast; TimesFM rounds every horizon up to 64).
  The model forecasts `log(mean)`, not `close`, because the settlement is a mean.
- **P(up)** comes from the nine deciles via a piecewise-linear CDF with exponential tails.
  Baselines: naive (flat at last close), RW+vol (`Φ((log last − log K)/(σ√(14−m+1/3)))` from
  trailing 1-minute vol), constant 0.5.
- **Trading**: buy YES at price `c` when P(up) ≥ τ, NO at `1−c` when P(up) ≤ 1−τ; Kalshi fee
  `0.07·p·(1−p)` on by default. There are no historical Kalshi quotes, so fixed-price P&L is
  optimistic; **read the model-vs-RW+vol difference and `bss_rw`, not absolute P&L.** A
  `--quotes` file with `(t0, m, yes_bid, yes_ask)` replaces the fixed price when you have one.
- Confidence intervals are a day-block bootstrap (origins inside a day overlap heavily).

## Tests

```bash
uv run pytest -q                      # offline: fake QuestDB, synthetic ticks, stub model
CHUDP_SLOW=1 uv run pytest -m slow    # loads the real weights
uv run ruff check .
```

## Caveats

- TimesFM 3.0 weights are under the **TimesFM Non-Commercial License v1.0**. `model.py` keeps a
  backend seam; `google/timesfm-2.0-500m-pytorch` (Apache-2.0) is the clean swap if this ever
  drives real orders.
- The up/down market's exact strike rule is not documented by Kalshi; the two strike modes
  bracket it. If they disagree at m ≥ 12, get the real strike from the Kalshi API before trusting
  anything.
- Context is capped at 15,360 bars by the model (10.7 days of 1-minute bars).
