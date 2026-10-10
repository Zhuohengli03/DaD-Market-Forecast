import uvicorn

from .settings import Settings

if __name__ == "__main__":
    s = Settings()
    print(f"Darker Market 已启动：http://localhost:{s.port}  （按 Ctrl+C 停止）", flush=True)
    # warnings and errors only: per-request access lines mean nothing to people using the app
    uvicorn.run("backend.main:create_app", factory=True, host=s.host, port=s.port, log_level="warning", access_log=False)
