def test_listings_default_page(client):
    r = client.get("/api/listings/Test Ore")
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 100 and body["meta"] == {"total": 1000, "page": 1, "page_size": 100}
    assert body["data"][0]["created_at"] >= body["data"][-1]["created_at"]


def test_listings_filters(client):
    body = client.get("/api/listings/Test Ore", params={"sold": "true", "min_qty": 6, "page_size": 500}).json()
    assert all(r["has_sold"] and r["quantity"] >= 6 for r in body["data"])
    body2 = client.get("/api/listings/Test Ore", params={"season_id": "s1"}).json()
    assert body2["meta"]["total"] == 0


def test_listings_pagination_validation(client):
    assert client.get("/api/listings/Test Ore", params={"page_size": 501}).status_code == 422
    assert client.get("/api/listings/Test Ore", params={"page": 0}).status_code == 422
    assert client.get("/api/listings/Ghost").status_code == 404
