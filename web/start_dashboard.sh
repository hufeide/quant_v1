#!/usr/bin/env bash
# 启动 dashboard HTTP 服务。由 Windows 批处理调用，也可在 WSL 内直接执行。
set -uo pipefail

HOST="0.0.0.0"
PORT="${PORT:-10000}"
DIR="/home/administator/workspace/quant_v1"
LOG="/tmp/dashboard_server.log"

cd "$DIR" || { echo "FAIL cannot cd to $DIR"; exit 1; }

# 清理旧进程，避免端口占用
pkill -f "http.server ${PORT}" 2>/dev/null
sleep 1

if [ ! -f "$DIR/dashboard.html" ]; then
  echo "FAIL dashboard.html not found - run: python3 web/build_dashboard.py"
  exit 1
fi

nohup python3 -m http.server "$PORT" --bind "$HOST" >"$LOG" 2>&1 &
SERVER_PID=$!

# 等待端口就绪
for i in $(seq 1 20); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL server exited early, log:"
    cat "$LOG"
    exit 1
  fi
  if curl -s -o /dev/null "http://localhost:${PORT}/dashboard.html"; then
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/dashboard.html")
    echo "OK http://localhost:${PORT}/dashboard.html  HTTP $CODE  pid=$SERVER_PID"
    exit 0
  fi
  sleep 0.5
done

echo "FAIL server did not become ready, log:"
cat "$LOG"
exit 1
