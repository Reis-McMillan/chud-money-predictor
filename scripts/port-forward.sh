#!/usr/bin/env bash
# Manage a kubectl port-forward to the production QuestDB HTTP port.
#   scripts/port-forward.sh start|stop|status [LOCAL_PORT]
# Default local port 19000 -> svc/questdb:9000 in namespace questdb.
set -euo pipefail

CMD="${1:-status}"
LOCAL_PORT="${2:-${QDB_PORT:-19000}}"
RUN_DIR="$(cd "$(dirname "$0")/.." && pwd)/.run"
PIDFILE="$RUN_DIR/port-forward-$LOCAL_PORT.pid"
LOGFILE="$RUN_DIR/port-forward-$LOCAL_PORT.log"
mkdir -p "$RUN_DIR"

is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "$CMD" in
  start)
    if is_running; then
      echo "already running (pid $(cat "$PIDFILE")) on 127.0.0.1:$LOCAL_PORT"
      exit 0
    fi
    nohup kubectl port-forward -n questdb svc/questdb "$LOCAL_PORT:9000" >"$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 40); do
      if grep -q "Forwarding from" "$LOGFILE" 2>/dev/null; then
        echo "port-forward up: 127.0.0.1:$LOCAL_PORT -> questdb:9000 (pid $(cat "$PIDFILE"))"
        exit 0
      fi
      if ! is_running; then break; fi
      sleep 0.25
    done
    echo "port-forward failed to start; log follows:" >&2
    cat "$LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PIDFILE")" && echo "stopped port-forward (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if is_running; then
      echo "running (pid $(cat "$PIDFILE")) on 127.0.0.1:$LOCAL_PORT"
      if curl -s --max-time 5 "http://127.0.0.1:$LOCAL_PORT/exec?query=select%201" >/dev/null; then
        echo "questdb reachable"
      else
        echo "questdb NOT reachable through the forward" >&2
        exit 1
      fi
    else
      echo "not running"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 start|stop|status [LOCAL_PORT]" >&2
    exit 2
    ;;
esac
