#!/usr/bin/env bash
# Darker Market 一键启动（macOS / Linux）
#   ./start.sh            安装所需环境（仅首次或有更新时）、启动服务并打开浏览器
#   ./start.sh --no-open  启动但不打开浏览器
#   PORT=5050 ./start.sh  换一个端口
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-5001}"
OPEN=1; [ "${1:-}" = "--no-open" ] && OPEN=0
URL="http://localhost:${PORT}"

step() { printf "\033[1;34m▶ %s\033[0m\n" "$1"; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$1"; }
die()  { printf "\033[1;31m✗ %s\033[0m\n" "$1"; exit 1; }
hash_of() { if command -v shasum >/dev/null; then shasum | cut -c1-16; else sha1sum | cut -c1-16; fi; }
open_browser() { [ "$OPEN" = 1 ] || return 0; (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true; }

# 1. 检查 Python 与 Node
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "需要 Python 3.10 或更高版本，请先安装：https://www.python.org/downloads/"
command -v npm >/dev/null 2>&1 || die "需要 Node.js 20 或更高版本，请先安装：https://nodejs.org/"
ok "Python $("$PY" -c 'import platform; print(platform.python_version())')，Node $(node --version)"

# 2. 已经在运行就直接打开
if curl -fs "$URL/api/status" >/dev/null 2>&1; then ok "Darker Market 已在运行：$URL"; open_browser; exit 0; fi

# 3. Python 环境（.venv），依赖有变化时才重新安装
if [ ! -x .venv/bin/python ]; then step "首次运行：创建独立的 Python 环境（.venv）"; "$PY" -m venv .venv; fi
VPY=.venv/bin/python
REQ_HASH=$(cat requirements.txt | hash_of)
if [ "$(cat .venv/.requirements-hash 2>/dev/null || true)" != "$REQ_HASH" ]; then
  step "安装 Python 依赖（首次约 1~3 分钟）"
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -r requirements.txt
  echo "$REQ_HASH" > .venv/.requirements-hash
fi
ok "Python 依赖已就绪"

# 4. 数据库：没有 Postgres 也没关系，自动建好内置数据库（data/market.db）
"$VPY" -m backend.init_db | sed 's/^/  /'
ok "数据库已就绪"

# 5. 前端，代码有变化时才重新构建
if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules/.package-lock.json ]; then
  step "安装前端依赖（首次约 1 分钟）"; (cd frontend && npm ci --silent --no-audit --no-fund)
fi
FE_HASH=$( (cat frontend/package-lock.json frontend/index.html frontend/vite.config.ts; find frontend/src -type f -print0 | sort -z | xargs -0 cat) | hash_of)
if [ "$(cat frontend/dist/.build-hash 2>/dev/null || true)" != "$FE_HASH" ]; then
  step "构建页面"
  if ! (cd frontend && npm run build --silent > ../.build.log 2>&1); then cat .build.log; die "页面构建失败，上面是原因"; fi
  rm -f .build.log; echo "$FE_HASH" > frontend/dist/.build-hash
fi
ok "页面已就绪"

# 6. 启动服务，就绪后打开浏览器
step "启动 Darker Market：$URL （按 Ctrl+C 停止）"
( for _ in $(seq 1 60); do if curl -fs "$URL/api/status" >/dev/null 2>&1; then open_browser; exit 0; fi; sleep 0.5; done ) &
export PORT
exec "$VPY" -m backend
