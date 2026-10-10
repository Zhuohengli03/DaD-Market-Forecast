"""LAN access and port are set in the app (Settings), saved to .env, and applied by restarting from the UI."""
import socket

H = {"X-Darker-Market": "1"}


def test_network_settings_round_trip(client, settings):
    d = client.get("/api/setup/network").json()["data"]
    assert d["lan"] is False and d["port"] == settings.port and d["restart_required"] is False
    r = client.put("/api/setup/network", json={"lan": True, "port": 5077}, headers=H)
    d = r.json()["data"]
    assert r.status_code == 200 and d["lan"] is True and d["port"] == 5077 and d["restart_required"] is True
    env = settings.env_path.read_text()
    assert "HOST=0.0.0.0" in env and "PORT=5077" in env
    assert all(":5077" in u for u in d["lan_urls"])


def test_rejects_bad_or_busy_ports(client):
    assert client.put("/api/setup/network", json={"lan": False, "port": 80}, headers=H).status_code == 422
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); s.listen(1)
        busy = s.getsockname()[1]
        r = client.put("/api/setup/network", json={"lan": False, "port": busy}, headers=H)
        assert r.status_code == 409 and str(busy) in r.json()["error"]


def test_restart_from_the_app(client):
    calls = []
    client.app.state.restarter = lambda: calls.append("restart")
    r = client.post("/api/setup/restart", headers=H)
    assert r.status_code == 202
    import time
    for _ in range(50):
        if calls:
            break
        time.sleep(0.05)
    assert calls == ["restart"]
