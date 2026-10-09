import uvicorn

from .settings import Settings

if __name__ == "__main__":
    s = Settings()
    uvicorn.run("backend.main:create_app", factory=True, host=s.host, port=s.port)
