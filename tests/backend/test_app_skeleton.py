def test_status_envelope(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["error"] is None
    assert body["data"]["source"] == "sqlite"
    assert body["data"]["items"][0]["name"] == "Test Ore"


def test_unknown_route_is_enveloped_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["ok"] is False
