#!/usr/bin/env python3
"""Check a LiteLLM deployment (config.yaml and optional .env) against the invariants
this integration depends on. Exit 0 = clean, 1 = violations.

    python check_deployment.py --config config.yaml [--env .env] [--embedding-group embedding]

Invariants (each exists because it regressed or was found missing in production):

  embedding-group     a model group with the embedding name exists, does not appear in
                      router_settings.fallbacks / context_window_fallbacks /
                      content_policy_fallbacks, and is not a tier of a deployment's
                      complexity_router_config. Not checked: litellm_settings.fallbacks,
                      default_fallbacks, or the embedding group as an alias target.
  priced              every deployment not exempted below is priced: it declares
                      input_cost_per_token and output_cost_per_token, or LiteLLM's built-in
                      cost map knows the model. Exempt: the embedding group, auto-router
                      deployments, and any deployment whose api_base is `os.environ/NAME`
                      with NAME starting OLLAMA, VLLM or LLAMA (the exemption is decided by
                      that variable NAME only; the checker does not look at where it points).
  no-ip-literal       no literal api_base is an IP address, and (only when --env is given)
                      no env variable named *_URL/_BASE/_BASE_URL/_ENDPOINT/_HOST, and no
                      variable referenced as an api_base through os.environ/NAME, holds an
                      IP address. Variables whose name contains BIND or LISTEN are skipped.
  alias-targets       every model_group_alias target is a real model group.
  trusted-proxies     general_settings.trusted_proxy_ranges is set (warning only).

Without --env the environment half of no-ip-literal is not evaluated, and a NOTE line
says so. The checker validates generic proxy hygiene only; it does not check that any
callback is registered or that any LOCI_* setting exists.

Stdlib plus PyYAML.
"""
from __future__ import annotations

import argparse
import ipaddress
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ENV_REF = re.compile(r"^os\.environ/(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
URL_VAR = re.compile(r"(_URL|_BASE|_BASE_URL|_ENDPOINT|_HOST)$")
BIND_VAR = re.compile(r"(BIND|LISTEN)", re.I)
# Deployments are treated as local (exempt from `priced`) when their api_base is
# `os.environ/NAME` and NAME matches this pattern. This is a naming convention, nothing more.
LOCAL_BASE_ENV = re.compile(r"^(OLLAMA|VLLM|LLAMA)[A-Z_]*$")


def _host_is_ip(value: str) -> bool:
    """True when the URL's host (or a bare host) is an IP literal."""
    value = value.strip().strip("'\"")
    if not value:
        return False
    host = urlsplit(value if "://" in value else f"//{value}").hostname or ""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _cost_map_prices(model: str) -> bool:
    """True when LiteLLM's built-in cost map has both per-token prices for this model.

    The deployment model is provider-prefixed ("openai/openai/o3"); try the full name and each
    suffix. Returns False when litellm cannot be imported, so an unprovable price is a failure."""
    try:
        import litellm
        table = litellm.model_cost
    except Exception:
        return False
    parts = model.split("/")
    for i in range(len(parts)):
        entry = table.get("/".join(parts[i:]))
        if entry and entry.get("input_cost_per_token") is not None and entry.get("output_cost_per_token") is not None:
            return True
    return False


def load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


def check(config: dict, env: dict[str, str] | None = None, embedding_group: str = "embedding"):
    """Return (violations, warnings) as lists of strings."""
    violations: list[str] = []
    warnings: list[str] = []
    deployments = config.get("model_list") or []
    groups = {d.get("model_name") for d in deployments if d.get("model_name")}

    # embedding-group
    emb = [d for d in deployments if d.get("model_name") == embedding_group]
    if not emb:
        violations.append(f"embedding-group: no model group named {embedding_group!r}")
    router = config.get("router_settings") or {}
    for key in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
        for entry in router.get(key) or []:
            for src, dsts in (entry or {}).items():
                if embedding_group == src or embedding_group in (dsts or []):
                    violations.append(f"embedding-group: {embedding_group!r} appears in router_settings.{key}")
    for d in deployments:
        params = d.get("litellm_params") or {}
        tiers = (params.get("complexity_router_config") or {}).get("tiers") or {}
        if embedding_group in tiers.values():
            violations.append(f"embedding-group: {embedding_group!r} is a tier of the auto-router")

    # priced
    for d in deployments:
        params = d.get("litellm_params") or {}
        info = d.get("model_info") or {}
        if "complexity_router_config" in params or str(params.get("model", "")).startswith("auto_router/"):
            continue
        if d.get("model_name") == embedding_group:
            continue
        base = str(params.get("api_base", ""))
        m = ENV_REF.match(base)
        if m and LOCAL_BASE_ENV.match(m.group("name")):
            continue
        name = f"{d.get('model_name')} ({info.get('id') or params.get('model')})"
        missing = [f for f in ("input_cost_per_token", "output_cost_per_token") if f not in info and f not in params]
        if missing and not _cost_map_prices(str(params.get("model", ""))):
            for field in missing:
                violations.append(f"priced: {name} has no {field} and LiteLLM's cost map has no entry")

    # no-ip-literal (config)
    for d in deployments:
        base = (d.get("litellm_params") or {}).get("api_base")
        if isinstance(base, str) and not ENV_REF.match(base) and _host_is_ip(base):
            violations.append(f"no-ip-literal: {d.get('model_name')} api_base is an IP address: {base}")
    # no-ip-literal (env)
    referenced: set[str] = set()
    for d in deployments:
        base = (d.get("litellm_params") or {}).get("api_base")
        m = ENV_REF.match(base) if isinstance(base, str) else None
        if m:
            referenced.add(m.group("name"))
    for key, value in (env or {}).items():
        by_name = URL_VAR.search(key) and not BIND_VAR.search(key)
        if (by_name or key in referenced) and _host_is_ip(value):
            violations.append(f"no-ip-literal: {key} points at an IP address ({value}); use the DNS name")

    # alias-targets
    for alias, target in (router.get("model_group_alias") or {}).items():
        t = target if isinstance(target, str) else (target or {}).get("model")
        if t not in groups:
            violations.append(f"alias-targets: {alias!r} -> {t!r} is not a model group")

    # trusted-proxies
    if not (config.get("general_settings") or {}).get("trusted_proxy_ranges"):
        warnings.append("trusted-proxies: general_settings.trusted_proxy_ranges is not set")
    return violations, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--env", type=Path)
    ap.add_argument("--embedding-group", default="embedding")
    args = ap.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    env = load_env(args.env) if args.env else None
    violations, warnings = check(config, env, args.embedding_group)
    if env is None:
        print("NOTE  --env not given: environment variables were not checked for IP addresses")
    for w in warnings:
        print(f"WARN  {w}")
    for v in violations:
        print(f"FAIL  {v}")
    print(f"{len(violations)} violation(s), {len(warnings)} warning(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
