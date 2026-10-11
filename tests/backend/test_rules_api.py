"""Settings → rules: where the rules come from, a rules service to sync from (paid), and syncing now."""
from src import rules

H = {"X-Darker-Market": "1"}


def test_rules_status_starts_from_the_bundled_file(client):
    d = client.get("/api/rules").json()["data"]
    assert d["source"] == "bundled" and d["version"] == rules.bundled()["version"] and d["url"] == "" and d["key_saved"] is False


def test_set_a_rules_service_and_sync(client, settings):
    newer = {**rules.bundled(), "version": rules.bundled()["version"] + 1, "updated": "2026-12-01"}
    seen = {}
    client.app.state.rules_get = lambda url, headers: (seen.update(url=url, key=headers["X-License-Key"]), newer)[1]
    r = client.put("/api/rules/source", json={"url": "https://rules.example.com/v1/rules", "key": "LIC-12345678"}, headers=H)
    assert r.status_code == 200 and r.json()["data"]["key_hint"] == "5678"
    assert "LIC-12345678" in settings.env_path.read_text()                       # the key lives in .env, never in the reply
    assert "LIC-12345678" not in r.text
    d = client.post("/api/rules/sync", headers=H).json()["data"]
    assert d["error"] is None and d["source"] == "synced" and d["version"] == newer["version"] and d["synced_at"]
    assert seen == {"url": "https://rules.example.com/v1/rules", "key": "LIC-12345678"}


def test_sync_needs_a_service_and_only_https(client):
    assert client.post("/api/rules/sync", headers=H).status_code == 409
    assert client.put("/api/rules/source", json={"url": "http://rules.example.com"}, headers=H).status_code == 422


def test_a_failed_sync_is_reported_and_changes_nothing(client):
    client.app.state.rules_get = lambda url, headers: (_ for _ in ()).throw(OSError("offline"))
    client.put("/api/rules/source", json={"url": "https://rules.example.com/v1/rules", "key": "K"}, headers=H)
    d = client.post("/api/rules/sync", headers=H).json()["data"]
    assert d["error"] and d["source"] == "bundled"
