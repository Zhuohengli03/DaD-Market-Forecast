def test_list_seasons(client):
    data = client.get("/api/seasons").json()["data"]
    assert [s["id"] for s in data] == ["s1", "s2"]


def test_add_update_delete(client):
    r = client.post("/api/seasons", json={"name": "S0", "start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"})
    assert r.status_code == 201
    sid = r.json()["data"]["id"]
    r = client.put(f"/api/seasons/{sid}", json={"name": "S0b", "start": "2025-01-01T00:00:00Z", "end": "2025-03-01T00:00:00Z"})
    assert r.status_code == 200 and r.json()["data"]["name"] == "S0b"
    assert client.delete(f"/api/seasons/{sid}").status_code == 200
    assert client.delete(f"/api/seasons/{sid}").status_code == 404


def test_overlap_rejected(client):
    r = client.post("/api/seasons", json={"name": "X", "start": "2025-12-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"})
    assert r.status_code == 422 and "overlap" in r.json()["error"]
    assert len(client.get("/api/seasons").json()["data"]) == 2


def test_end_before_start_rejected(client):
    r = client.post("/api/seasons", json={"name": "X", "start": "2027-02-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"})
    assert r.status_code == 422


def test_open_ended_season_blocks_later_start(client):
    r = client.post("/api/seasons", json={"name": "X", "start": "2026-06-01T00:00:00Z", "end": None})
    assert r.status_code == 422  # s2 is open-ended from 2026-01-20


def test_bad_timestamp_rejected(client):
    r = client.post("/api/seasons", json={"name": "X", "start": "yesterday", "end": None})
    assert r.status_code == 422
