"""Rules that need upkeep (season openings, hand-over times, which posts are patches) live in a rules file,
not in code: the bundled one by default, a newer synced one when present and valid."""
import json

import pytest

from src import rules


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rules, "_data_dir", lambda: tmp_path)
    rules.clear_cache()
    yield tmp_path
    rules.clear_cache()


def _write(data_dir, **change):
    data = {**rules.bundled(), **change}
    (data_dir / "rules.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_bundled_rules_are_used_by_default(data_dir):
    r = rules.current()
    assert r["source"] == "bundled" and r["version"] >= 1
    assert r["seasons"]["market_openings"]["early-access-season-11"] == ["2026-10-09", "2026-10-23"]


def test_a_newer_synced_file_wins_an_older_or_broken_one_does_not(data_dir):
    newer = {**rules.bundled()["seasons"], "market_openings": {**rules.bundled()["seasons"]["market_openings"],
             "early-access-season-12": ["2027-01-15", "2027-01-29"]}}
    _write(data_dir, version=rules.bundled()["version"] + 1, seasons=newer)
    assert rules.current()["source"] == "synced"
    assert rules.current()["seasons"]["market_openings"]["early-access-season-12"] == ["2027-01-15", "2027-01-29"]
    _write(data_dir, version=0)
    rules.clear_cache()
    assert rules.current()["source"] == "bundled"                      # older than what ships: ignored
    (data_dir / "rules.json").write_text("{broken")
    rules.clear_cache()
    assert rules.current()["source"] == "bundled"                      # unreadable: ignored


def test_season_dates_and_patch_words_come_from_the_rules(data_dir):
    from backend.services.announcements import is_patch
    from src.season_calendar import with_market_opening
    seasons = {**rules.bundled()["seasons"], "market_openings": {"early-access-season-12": ["2027-01-15", "2027-01-29"]}}
    words = {"patch": ["hotfix"], "not_patch": ["hotfix #999"]}
    _write(data_dir, version=rules.bundled()["version"] + 1, seasons=seasons, announcements={**rules.bundled()["announcements"], **words})
    s12 = with_market_opening({"id": "early-access-season-12", "start": "2027-01-15T00:00:00+00:00", "end": None})
    assert s12["market_open"] == "2027-01-29"
    assert is_patch("Hotfix #130") and not is_patch("Hotfix #999") and not is_patch("Early Access Season #12")


def test_validation_names_what_is_wrong():
    with pytest.raises(ValueError, match="version"):
        rules.validate({"seasons": {}, "announcements": {}})
    bad = {**rules.bundled(), "seasons": {**rules.bundled()["seasons"], "market_openings": {"s": ["2027-01-15"]}}}
    with pytest.raises(ValueError, match="market_openings"):
        rules.validate(bad)


def test_sync_saves_a_valid_newer_file_and_keeps_the_old_one_on_failure(data_dir):
    newer = {**rules.bundled(), "version": rules.bundled()["version"] + 5}
    seen = {}

    def get(url, headers):
        seen.update(url=url, headers=headers)
        return newer
    assert rules.sync("https://rules.example/v1/rules", "KEY123", get) is None
    assert seen["headers"]["X-License-Key"] == "KEY123"
    assert rules.current()["version"] == newer["version"] and rules.current()["source"] == "synced"
    err = rules.sync("https://rules.example/v1/rules", "KEY123", lambda url, headers: {"nonsense": True})
    assert err and rules.current()["version"] == newer["version"]          # a bad answer changes nothing
    err = rules.sync("https://rules.example/v1/rules", "KEY123", lambda url, headers: (_ for _ in ()).throw(OSError("offline")))
    assert err and rules.current()["version"] == newer["version"]
