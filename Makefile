.PHONY: creds pf pf-stop pf-status sync-cpu sync-rocm download bars backtest-kalshi finetune smoke test test-slow lint

# --- credentials & connectivity ------------------------------------------------
creds:            ## fetch the QuestDB password from the cluster into .env
	./scripts/fetch-creds.sh

pf:               ## start kubectl port-forward (localhost:19000 -> questdb:9000)
	./scripts/port-forward.sh start

pf-stop:
	./scripts/port-forward.sh stop

pf-status:
	./scripts/port-forward.sh status

# --- environments --------------------------------------------------------------
sync-cpu:         ## laptop (macOS / CPU / MPS)
	uv sync --extra cpu --extra viz --group dev

sync-rocm:        ## MI300X box
	uv sync --extra rocm --extra viz --group dev

# --- pipeline ------------------------------------------------------------------
download:
	uv run chudp download $(ARGS)

bars:
	uv run chudp resample --freq 1m $(ARGS)

backtest-kalshi:
	uv run chudp backtest-kalshi --config configs/kalshi_full.toml $(ARGS)

finetune:         ## fine-tune on train, select on validation, score test once
	uv run chudp finetune --config configs/finetune.toml $(ARGS)

smoke:            ## 1-day CPU smoke run (needs data for 2025-09-19 downloaded)
	uv run chudp backtest-kalshi --config configs/kalshi_smoke.toml $(ARGS)

# --- quality -------------------------------------------------------------------
test:
	uv run pytest -q

test-slow:        ## also runs the tests that load the real model
	CHUDP_SLOW=1 uv run pytest -q -m "slow or not slow"

lint:
	uv run ruff check . && uv run ruff format --check .
