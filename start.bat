@echo off
rem Darker Market 一键启动（Windows）：双击运行，或在命令行执行 start.bat
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if "%PORT%"=="" set PORT=5001
set URL=http://localhost:%PORT%

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (echo 需要 Python 3.10 或更高版本，请先安装：https://www.python.org/downloads/ & pause & exit /b 1)
where npm >nul 2>nul
if errorlevel 1 (echo 需要 Node.js 20 或更高版本，请先安装：https://nodejs.org/ & pause & exit /b 1)

if not exist .venv\Scripts\python.exe (
  echo 首次运行：创建独立的 Python 环境（.venv）
  %PY% -m venv .venv || (pause & exit /b 1)
)
echo 检查 Python 依赖（首次约 1~3 分钟）
.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements.txt || (pause & exit /b 1)

echo 准备数据库（没有 Postgres 时自动建好内置数据库 data\market.db）
.venv\Scripts\python.exe -m backend.init_db || (pause & exit /b 1)

if not exist frontend\node_modules (
  echo 安装前端依赖（首次约 1 分钟）
  pushd frontend & call npm ci --silent --no-audit --no-fund & popd
)
echo 构建页面
pushd frontend & call npm run build --silent >nul & popd

echo 启动 Darker Market：%URL% （关闭此窗口即停止）
start "" cmd /c "timeout /t 4 >nul & start %URL%"
.venv\Scripts\python.exe -m backend
pause
