.PHONY: creds pf pf-stop pf-status sync-cpu sync-rocm download bars frame backtest finetune smoke test test-slow lint

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
download:         ## BRTI ticks + Kalshi contract candles
	uv run chudp download --source both $(ARGS)

bars:
	uv run chudp resample --freq 1m $(ARGS)

frame:            ## join bars and candles into the model frame
	uv run chudp frame $(ARGS)

backtest:
	uv run chudp backtest-contract --config configs/contract_full.toml $(ARGS)

finetune:         ## fine-tune on train, select on validation, score test once
	uv run chudp finetune --config configs/finetune.toml $(ARGS)

smoke:            ## 1-day CPU smoke run (see configs/contract_smoke.toml for the data it needs)
	uv run chudp backtest-contract --config configs/contract_smoke.toml $(ARGS)

# --- quality -------------------------------------------------------------------
test:
	uv run pytest -q

test-slow:        ## also runs the tests that load the real model
	CHUDP_SLOW=1 uv run pytest -q -m "slow or not slow"

lint:
	uv run ruff check .
