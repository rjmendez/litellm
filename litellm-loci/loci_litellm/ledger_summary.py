"""Aggregate the cost ledger into per-hour summaries and a small health snapshot.

LiteLLM-free (stdlib only), so the offline report, the ops_logs collector and Loci can all use it.

What it produces
  hourly_summaries(rows)  one dict per (hour, group, deployment): calls, ok, empty answers, p50/p95 latency,
                          spend, prompt and cache tokens, error classes, and how many calls were internal
                          (eval harness, probes, memory-system calls) versus organic.
  health(rows)            per group, for LOCAL deployments only, over a recent window: attempts, ok rate,
                          empty-answer rate, p95. This is the signal a router can act on. It is computed from
                          ORGANIC rows when there are enough of them and says so in `basis`; it never mixes
                          the two silently.

Honesty notes
  - Rows are per attempt (a router retry is its own row), so `calls` is attempts, not client requests.
  - `ok` here means succeeded AND returned an answer: an empty response counts as not ok.
  - Latency percentiles are over calls that succeeded with an answer.
  - A row written before the cache/empty fields existed reads as 0 cache tokens and not empty.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

HOUR_WORDS = ("midnight", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "noon", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
              "nineteen", "twenty", "twentyone", "twentytwo", "twentythree")


def read_rows(path: str, since_ts: float = 0.0) -> Iterator[dict]:
    """Yield ledger rows with ts >= since_ts. Unreadable lines are skipped (the report counts them)."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict) and float(row["ts"]) >= since_ts:
                    yield row
            except (ValueError, KeyError, TypeError):
                continue


def _pct(values: list, q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 2)


def _good(row: dict) -> bool:
    """Succeeded and returned an answer."""
    return bool(row.get("ok")) and not row.get("empty_response")


def _hour_start(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts).replace(minute=0, second=0, microsecond=0)


def hourly_summaries(rows: Iterable[dict]) -> list[dict]:
    buckets: dict = defaultdict(list)
    for row in rows:
        try:
            key = (_hour_start(float(row["ts"])), str(row.get("group") or "?"), str(row.get("deployment") or "?"))
        except (KeyError, TypeError, ValueError):
            continue
        buckets[key].append(row)
    out = []
    for (hour, group, deployment), rs in sorted(buckets.items()):
        good = [r for r in rs if _good(r)]
        lat = [r["latency_s"] for r in good if isinstance(r.get("latency_s"), (int, float))]
        prompt = sum(int(r.get("prompt_tokens") or 0) for r in rs)
        cache = sum(int(r.get("cache_read_tokens") or 0) for r in rs)
        internal = sum(1 for r in rs if r.get("internal"))
        out.append({
            "hour": hour.isoformat(timespec="minutes"),
            "day": hour.date().isoformat(),
            "hour_word": HOUR_WORDS[hour.hour],
            "group": group,
            "deployment": deployment,
            "local": bool(rs[0].get("local")),
            "calls": len(rs),
            "ok": sum(1 for r in rs if r.get("ok")),
            "good": len(good),
            "empty": sum(1 for r in rs if r.get("ok") and r.get("empty_response")),
            "internal_calls": internal,
            "organic_calls": len(rs) - internal,
            "p50_s": _pct(lat, 0.5),
            "p95_s": _pct(lat, 0.95),
            "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rs), 6),
            "prompt_tokens": prompt,
            "cache_read_tokens": cache,
            "cache_write_tokens": sum(int(r.get("cache_write_tokens") or 0) for r in rs),
            "cache_read_share": None if not prompt else round(cache / prompt, 4),
            "errors": dict(Counter(str(r.get("error_class") or "?") for r in rs if not r.get("ok"))),
        })
    return out


def severity(summary: dict) -> str:
    """info, warning or error for one hourly summary, so a degraded hour stands out in the log index."""
    calls = summary["calls"]
    if calls and summary["ok"] / calls < 0.5:
        return "error"
    if (calls and summary["good"] / calls < 0.9) or summary["empty"] or summary["errors"]:
        return "warning"
    return "info"


def format_line(summary: dict) -> str:
    s = summary
    cache = "n/a" if s["cache_read_share"] is None else f"{s['cache_read_share']:.2f}"
    errors = ",".join(f"{k}:{v}" for k, v in sorted(s["errors"].items())) or "none"
    return (f"litellm ledger hour={s['hour_word']} group={s['group']} deployment={s['deployment']} "
            f"local={str(s['local']).lower()} calls={s['calls']} ok={s['ok']} good={s['good']} empty={s['empty']} "
            f"organic={s['organic_calls']} internal={s['internal_calls']} p50_s={s['p50_s']} p95_s={s['p95_s']} "
            f"cost_usd={s['cost_usd']:.4f} prompt_tokens={s['prompt_tokens']} cache_read_share={cache} "
            f"errors={errors}")


def health(rows: Iterable[dict], window_s: float = 3600.0, now: Optional[float] = None,
           min_organic: int = 20) -> dict:
    """Per-group health of LOCAL deployments over the last `window_s` seconds.

    basis is "organic" when the group has at least `min_organic` non-internal local attempts in the window,
    else "all" (internal rows included) so the snapshot is still produced but is not mistaken for real use.
    """
    now = time.time() if now is None else now
    per: dict = defaultdict(list)
    for row in rows:
        try:
            if float(row["ts"]) >= now - window_s and row.get("local"):
                per[str(row.get("group") or "?")].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    groups = {}
    for group, rs in per.items():
        organic = [r for r in rs if not r.get("internal")]
        use, basis = (organic, "organic") if len(organic) >= min_organic else (rs, "all")
        good = [r for r in use if _good(r)]
        lat = [r["latency_s"] for r in good if isinstance(r.get("latency_s"), (int, float))]
        groups[group] = {
            "n": len(use),
            "n_organic": len(organic),
            "basis": basis,
            "ok_rate": round(sum(1 for r in use if r.get("ok")) / len(use), 4),
            "good_rate": round(len(good) / len(use), 4),
            "empty_rate": round(sum(1 for r in use if r.get("ok") and r.get("empty_response")) / len(use), 4),
            "p95_s": _pct(lat, 0.95),
        }
    return {"version": 1, "as_of": round(now, 3), "window_s": window_s, "min_organic": min_organic,
            "groups": groups}


def write_health_file(snapshot: dict, *paths: str) -> list[str]:
    """Write the snapshot atomically to each path (a failure on one path does not stop the others).
    Returns the paths actually written."""
    written = []
    for path in paths:
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, sort_keys=True)
            os.replace(tmp, target)
            written.append(str(target))
        except OSError:
            continue
    return written


def local_degraded(snapshot: dict, group: str, now: Optional[float] = None, max_age_s: float = 1800.0,
                   min_n: int = 20, min_good_rate: float = 0.9, max_empty_rate: float = 0.15,
                   require_organic: bool = True) -> tuple[bool, str]:
    """(degraded, reason) for a group from a snapshot. Conservative: a stale, thin, non-organic or missing
    snapshot is NOT evidence of degradation (returns False with the reason), so this can only ever move
    traffic away from local on measured, recent, organic failure."""
    now = time.time() if now is None else now
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        return False, "no usable health snapshot"
    age = now - float(snapshot.get("as_of") or 0)
    if age > max_age_s:
        return False, f"snapshot is {int(age)}s old (max {int(max_age_s)}s)"
    g = (snapshot.get("groups") or {}).get(group)
    if not g:
        return False, f"no local data for group {group!r}"
    if require_organic and g.get("basis") != "organic":
        return False, f"only synthetic traffic for {group!r} (organic n={g.get('n_organic')})"
    if g["n"] < min_n:
        return False, f"only {g['n']} attempts (need {min_n})"
    if g["good_rate"] < min_good_rate:
        return True, f"good rate {g['good_rate']:.2f} below {min_good_rate:.2f} over {g['n']} attempts"
    if g["empty_rate"] > max_empty_rate:
        return True, f"empty rate {g['empty_rate']:.2f} above {max_empty_rate:.2f} over {g['n']} attempts"
    return False, "local meets the thresholds"
