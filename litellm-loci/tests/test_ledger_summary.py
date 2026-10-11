"""ledger_summary + the ops_logs collector: aggregation, honesty about organic vs internal, health logic."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

from loci_litellm import ledger_summary as ls

PKG_ROOT = Path(__file__).resolve().parents[1]
NOW = time.mktime((2026, 10, 11, 18, 30, 0, 0, 0, -1))


def row(i=0, **kw):
    base = {"ts": NOW - 60 * i, "group": "medium-tier", "deployment": "dep-local", "local": True, "ok": True,
            "empty_response": False, "internal": False, "latency_s": 1.0, "cost_usd": 0.0,
            "prompt_tokens": 100, "completion_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "error_class": ""}
    base.update(kw)
    return base


def test_hourly_summary_counts_and_splits_organic_from_internal() -> None:
    rows = [row(i) for i in range(6)] + [row(7, internal=True), row(8, internal=True)]
    rows += [row(9, ok=False, error_class="BadGatewayError"), row(10, empty_response=True)]
    (s,) = ls.hourly_summaries(rows)
    assert (s["calls"], s["ok"], s["good"], s["empty"]) == (10, 9, 8, 1)
    assert (s["organic_calls"], s["internal_calls"]) == (8, 2)
    assert s["errors"] == {"BadGatewayError": 1}
    assert s["hour_word"] == "eighteen" and s["local"] is True


def test_latency_percentiles_use_only_answered_calls() -> None:
    rows = [row(i, latency_s=1.0) for i in range(8)] + [row(9, latency_s=99.0, empty_response=True),
                                                         row(10, latency_s=77.0, ok=False)]
    (s,) = ls.hourly_summaries(rows)
    assert s["p95_s"] == 1.0


def test_cache_share_and_old_rows() -> None:
    rows = [row(0, prompt_tokens=1000, cache_read_tokens=900), {"ts": NOW, "group": "medium-tier",
            "deployment": "dep-local", "local": True, "ok": True, "prompt_tokens": 1000}]
    (s,) = ls.hourly_summaries(rows)
    assert s["cache_read_share"] == 0.45 and s["empty"] == 0


def test_the_hour_is_spelled_so_the_shippers_digit_stripping_cannot_merge_hours() -> None:
    lines = []
    for hour_ts in (NOW, NOW - 3600):
        (s,) = ls.hourly_summaries([row(0, ts=hour_ts)])
        lines.append(ls.format_line(s))
    import re
    stripped = {re.sub(r"\d+", "<n>", ln) for ln in lines}
    assert len(stripped) == 2, "two different hours collapse to one signature"


def test_severity() -> None:
    (clean,) = ls.hourly_summaries([row(i) for i in range(5)])
    assert ls.severity(clean) == "info"
    (empty,) = ls.hourly_summaries([row(0), row(1, empty_response=True)])
    assert ls.severity(empty) == "warning"
    (bad,) = ls.hourly_summaries([row(0, ok=False), row(1, ok=False), row(2)])
    assert ls.severity(bad) == "error"


def test_no_private_fields_leak_into_the_line() -> None:
    (s,) = ls.hourly_summaries([row(0, agent="OpenAI/Python 2.24", key_alias="secret-alias", trace_id="t")])
    text = ls.format_line(s) + json.dumps(s)
    assert "secret-alias" not in text and "OpenAI/Python" not in text


# --- health ------------------------------------------------------------------------------------

def test_health_uses_organic_rows_when_there_are_enough_and_says_so() -> None:
    rows = [row(i) for i in range(25)] + [row(30 + i, internal=True, ok=False) for i in range(10)]
    g = ls.health(rows, now=NOW)["groups"]["medium-tier"]
    assert g["basis"] == "organic" and g["n"] == 25 and g["good_rate"] == 1.0


def test_health_falls_back_to_all_rows_and_labels_it() -> None:
    rows = [row(i) for i in range(5)] + [row(10 + i, internal=True) for i in range(30)]
    g = ls.health(rows, now=NOW)["groups"]["medium-tier"]
    assert g["basis"] == "all" and g["n"] == 35 and g["n_organic"] == 5


def test_health_ignores_cloud_rows_and_old_rows() -> None:
    rows = [row(0, local=False), row(1, ts=NOW - 7200)]
    assert ls.health(rows, now=NOW)["groups"] == {}


def _snap(**g):
    base = {"n": 40, "n_organic": 40, "basis": "organic", "ok_rate": 1.0, "good_rate": 1.0, "empty_rate": 0.0, "p95_s": 1.0}
    base.update(g)
    return {"version": 1, "as_of": NOW, "window_s": 3600, "min_organic": 20, "groups": {"medium-tier": base}}


@pytest.mark.parametrize("snap,now,expected_degraded,fragment", [
    (_snap(), NOW, False, "meets"),
    (_snap(good_rate=0.5), NOW, True, "good rate"),
    (_snap(empty_rate=0.4), NOW, True, "empty rate"),
    (_snap(good_rate=0.5), NOW + 7200, False, "old"),                      # stale: not evidence
    (_snap(good_rate=0.5, n=5), NOW, False, "only 5"),                     # thin: not evidence
    (_snap(good_rate=0.5, basis="all", n_organic=2), NOW, False, "synthetic"),   # eval traffic: not evidence
    ({"version": 2}, NOW, False, "no usable"),
    (None, NOW, False, "no usable"),
])
def test_local_degraded_is_conservative(snap, now, expected_degraded, fragment) -> None:
    degraded, reason = ls.local_degraded(snap, "medium-tier", now=now)
    assert degraded is expected_degraded and fragment in reason


def test_unknown_group_is_not_degraded() -> None:
    assert ls.local_degraded(_snap(), "nope", now=NOW)[0] is False


def test_health_file_is_atomic_and_tolerates_a_bad_path(tmp_path: Path) -> None:
    good = tmp_path / "sub" / "h.json"
    bad = tmp_path / "file-not-dir" / "h.json"
    (tmp_path / "file-not-dir").write_text("x")             # a file where a directory is needed
    written = ls.write_health_file({"version": 1}, str(good), str(bad))
    assert written == [str(good)] and json.loads(good.read_text())["version"] == 1
    assert not [p for p in good.parent.iterdir() if p.suffix == ".tmp"]


# --- collector ---------------------------------------------------------------------------------

def _collector():
    spec = importlib.util.spec_from_file_location("ops_logs_ledger_collector",
                                                  PKG_ROOT / "deploy" / "ops_logs_ledger_collector.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ops_logs_ledger_collector"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeAgg:
    def __init__(self):
        self.calls = []

    def add(self, source, host, path, day, tstr, line, sev=None):
        self.calls.append({"source": source, "day": day, "tstr": tstr, "line": line, "sev": sev})


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_collector_ships_one_line_per_hour_group_deployment_and_the_health_file(tmp_path: Path) -> None:
    now = time.time()
    rows = [row(i, ts=now - 60 * i) for i in range(30)] + [row(0, ts=now, group="simple-tier", deployment="cloud",
                                                                 local=False, cost_usd=0.02)]
    ledger = _write(tmp_path / "l.jsonl", rows)
    agg, health_path = FakeAgg(), tmp_path / "health.json"
    n = _collector().ship_ledger(agg, {}, {}, "win-litellm-ledger-test", ledger, False, health_paths=[str(health_path)])
    assert n == len(agg.calls) >= 2
    assert {c["source"] for c in agg.calls} == {"win-litellm-ledger-test"}
    snap = json.loads(health_path.read_text())
    assert snap["version"] == 1 and "medium-tier" in snap["groups"] and "simple-tier" not in snap["groups"]


def test_collector_writes_no_health_file_when_told_not_to(tmp_path: Path) -> None:
    ledger = _write(tmp_path / "l.jsonl", [row(0, ts=time.time())])
    health_path = tmp_path / "health.json"
    _collector().ship_ledger(FakeAgg(), {}, {}, "src", ledger, False, health_paths=[str(health_path)],
                             write_health=False)
    assert not health_path.exists()


def test_collector_reports_an_unreadable_ledger_loudly(tmp_path: Path) -> None:
    agg = FakeAgg()
    n = _collector().ship_ledger(agg, {}, {}, "src", str(tmp_path / "missing.jsonl"), False)
    assert n == 0 and agg.calls[0]["sev"] == "error" and "unreadable" in agg.calls[0]["line"]
