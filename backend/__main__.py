import uvicorn

from .settings import Settings

if __name__ == "__main__":
    s = Settings()
    uvicorn.run("backend.main:create_app", factory=True, host="0.0.0.0", port=s.port)
