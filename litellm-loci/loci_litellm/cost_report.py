"""Report on the cost ledger: spend per model group, what the local tier saved, and whether local meets
the reliability bar.

    python -m loci_litellm.cost_report --config <litellm config.yaml> [--ledger FILE] [--since-hours 24]
                                       [--exclude-internal] [--json]

What the numbers are:
  - Counts are per ATTEMPT (one ledger row per LiteLLM call, retries and fallbacks included), not per
    client request.
  - "local" is the ledger's `local` field, an address-class heuristic on the deployment host (see
    cost_ledger). A local GPU host reached by a plain hostname counts as cloud unless it is named in
    LOCI_LOCAL_HOSTS when the ledger is written.
  - Savings are counterfactual: every token served successfully by a local deployment is priced at the
    group's priced deployment with the lowest (input + output) per-token price sum, taken from
    `model_info.input_cost_per_token` / `output_cost_per_token` in the config, local deployments included:
    a local deployment that declares a price (or an explicit 0) is a candidate, and a 0 makes the group's
    savings 0. It ignores the real input/output mix. A group with no priced deployment reports savings as null ("n/a"), not zero.
  - Latency percentiles (local_p50_s / local_p95_s) are over SUCCESSFUL local calls only; failed calls
    are excluded from them. Rows whose latency is null are skipped.
  - By default internal calls (judges, classifiers, embeddings flagged by the memory system) are
    included in every figure; --exclude-internal drops them. The mode is printed.
  - The SLO is LOCI_LOCAL_SLO_SUCCESS (default 0.98), LOCI_LOCAL_SLO_P95_S (default 20) and
    LOCI_LOCAL_SLO_MIN_CALLS (default 50). A malformed value falls back to the default with a warning.
  - The verdict says only whether local met the SLO and which criterion failed. The report does not read
    deployment order from the config, so it cannot say whether the order already matches.
Unparseable ledger lines are counted and reported on stderr. It only reports; it never edits config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def _pctl(values: list, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def _env_number(name: str, default: float, cast=float):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"warning: {name}={raw!r} is not a number; using the default {default}", file=sys.stderr)
        return default


def cloud_prices(config_path: str) -> dict:
    """{group: (input_per_token, output_per_token)} of each group's lowest-priced deployment
    (lowest input + output per-token sum)."""
    import yaml
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    best: dict = {}
    for entry in data.get("model_list") or []:
        group = entry.get("model_name")
        info = entry.get("model_info") or {}
        p_in, p_out = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if group and p_in is not None and p_out is not None:
            price = (float(p_in), float(p_out))
            if group not in best or sum(price) < sum(best[group]):
                best[group] = price
    return best


def read_rows(path: str, since_ts: float, stats: Optional[dict] = None) -> Iterable[dict]:
    """Yield ledger rows newer than since_ts. If `stats` is given it receives `read` and `skipped`
    (unparseable, non-object or timestamp-less lines)."""
    if stats is not None:
        stats.setdefault("read", 0)
        stats.setdefault("skipped", 0)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
                ts = float(row["ts"])
            except (ValueError, KeyError, TypeError):
                if stats is not None:
                    stats["skipped"] += 1
                continue
            if stats is not None:
                stats["read"] += 1
            if ts >= since_ts:
                yield row


def summarise(rows: Iterable[dict], prices: dict, include_internal: bool = True) -> dict:
    groups: dict = defaultdict(lambda: {"local": defaultdict(list), "cloud": defaultdict(list)})
    for row in rows:
        if not include_internal and row.get("internal"):
            continue
        groups[row.get("group") or "?"]["local" if row.get("local") else "cloud"][row.get("deployment") or "?"].append(row)
    out: dict = {}
    slo_ok = _env_number("LOCI_LOCAL_SLO_SUCCESS", 0.98)
    slo_p95 = _env_number("LOCI_LOCAL_SLO_P95_S", 20.0)
    slo_min = _env_number("LOCI_LOCAL_SLO_MIN_CALLS", 50, int)
    for group, sides in groups.items():
        local_rows = [r for rs in sides["local"].values() for r in rs]
        cloud_rows = [r for rs in sides["cloud"].values() for r in rs]
        local_ok = [r for r in local_rows if r.get("ok")]
        price = prices.get(group)
        if price is None:
            counterfactual: Optional[float] = None
        else:
            counterfactual = sum(r.get("prompt_tokens", 0) * price[0] + r.get("completion_tokens", 0) * price[1]
                                 for r in local_ok)
        success = (len(local_ok) / len(local_rows)) if local_rows else None
        lat = [r["latency_s"] for r in local_ok if isinstance(r.get("latency_s"), (int, float))]
        p95 = _pctl(lat, 0.95)
        failing = []
        if not local_rows:
            failing.append("no local calls")
        else:
            if len(local_rows) < slo_min:
                failing.append(f"only {len(local_rows)} local calls, need {slo_min}")
            if success < slo_ok:
                failing.append(f"success {success:.1%} below {slo_ok:.1%}")
            if p95 > slo_p95:
                failing.append(f"p95 {p95:.1f}s above {slo_p95:g}s")
        meets = not failing
        out[group] = {
            "calls": len(local_rows) + len(cloud_rows),
            "local_calls": len(local_rows),
            "cloud_calls": len(cloud_rows),
            "local_success": None if success is None else round(success, 4),
            "local_p50_s": round(_pctl(lat, 0.5), 2),
            "local_p95_s": round(p95, 2),
            "cloud_spend_usd": round(sum(r.get("cost_usd", 0.0) for r in cloud_rows), 6),
            "local_savings_usd": None if counterfactual is None else round(counterfactual, 6),
            "local_failures": sorted({r.get("error_class") or "?" for r in local_rows if not r.get("ok")}),
            "meets_slo": meets,
            "recommendation": "local meets the SLO" if meets else "local does not meet the SLO: " + "; ".join(failing),
        }
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the LiteLLM config.yaml (for per-deployment prices)")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--since-hours", type=float, default=24.0)
    ap.add_argument("--exclude-internal", action="store_true",
                    help="leave out calls the memory system flags as internal (judges, classifiers, embeddings)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from loci_litellm.cost_ledger import ledger_path
    path = args.ledger or ledger_path()
    if not os.path.exists(path):
        print(f"no ledger at {path}", file=sys.stderr)
        return 2
    stats: dict = {}
    prices = cloud_prices(args.config)
    summary = summarise(read_rows(path, time.time() - args.since_hours * 3600, stats), prices,
                        include_internal=not args.exclude_internal)
    if stats.get("skipped"):
        print(f"warning: skipped {stats['skipped']} unreadable ledger line(s) of "
              f"{stats['skipped'] + stats['read']} in {path}", file=sys.stderr)
    unpriced = sorted(g for g in summary if g not in prices)
    if unpriced:
        print("warning: no priced deployment in the config for group(s): " + ", ".join(unpriced)
              + " (savings shown as n/a)", file=sys.stderr)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    mode = "internal calls excluded" if args.exclude_internal else "internal calls included"
    print(f"ledger {path}: {stats.get('read', 0)} rows read, {mode}; latency over successful local calls only; "
          f"savings priced at the lowest-priced deployment per group")
    print(f"{'group':16s} {'calls':>6s} {'local':>6s} {'ok%':>6s} {'p95s':>6s} {'cloud$':>9s} {'saved$':>9s}  verdict")
    for group, s in sorted(summary.items()):
        ok = "-" if s["local_success"] is None else f"{100 * s['local_success']:.1f}"
        saved = "n/a" if s["local_savings_usd"] is None else f"{s['local_savings_usd']:.4f}"
        print(f"{group:16s} {s['calls']:6d} {s['local_calls']:6d} {ok:>6s} {s['local_p95_s']:6.1f} "
              f"{s['cloud_spend_usd']:9.4f} {saved:>9s}  {s['recommendation']}")
    total_saved = sum(s["local_savings_usd"] or 0.0 for s in summary.values())
    total_spend = sum(s["cloud_spend_usd"] for s in summary.values())
    note = " (groups without a price excluded)" if unpriced else ""
    print(f"\ncloud spend ${total_spend:.4f}, saved by local ${total_saved:.4f}{note} over the last "
          f"{args.since_hours:g} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
