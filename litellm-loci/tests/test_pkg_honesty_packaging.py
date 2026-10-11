"""Packaging honesty: one implementation, an identifiable version, a README that matches the code."""
from __future__ import annotations

import ast
import importlib
import logging
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO = PKG_ROOT.parent
SHIM_DIR = REPO / "litellm" / "integrations" / "loci"
PKG_DIR = PKG_ROOT / "loci_litellm"

SHIM_NAMES = {
    "backend_resolver": ["BackendResolver"],
    "cost_ledger": ["LociCostLedger", "loci_cost_ledger"],
    "grounding_hook": ["LociGroundingHook", "loci_grounding_hook"],
    "memory_writer": ["LociMemoryWriter", "loci_memory_writer"],
    "tier_router": ["LociTierRouter", "loci_tier_router"],
    "dedup_gate": ["LociDedupGate", "loci_dedup_gate"],
}


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=180)


# --- one source of truth -------------------------------------------------------------------------

@pytest.mark.parametrize("module,names", sorted(SHIM_NAMES.items()))
def test_shim_modules_are_the_same_objects_as_the_package(module, names) -> None:
    shim = importlib.import_module(f"litellm.integrations.loci.{module}")
    pkg = importlib.import_module(f"loci_litellm.{module}")
    for name in names:
        assert getattr(shim, name) is getattr(pkg, name), f"{module}.{name} is a different object in the shim"


def test_every_package_callback_module_has_a_shim_and_vice_versa() -> None:
    pkg = {p.stem for p in PKG_DIR.glob("*.py")} - {"__init__", "cost_report"}
    shim = {p.stem for p in SHIM_DIR.glob("*.py")} - {"__init__"}
    assert pkg == shim == set(SHIM_NAMES)


@pytest.mark.parametrize("path", sorted(SHIM_DIR.glob("*.py")), ids=lambda p: p.name)
def test_shim_files_hold_no_logic(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    defs = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert defs == [], f"{path.name} defines {[d.name for d in defs]}; the implementation lives only in loci_litellm"
    assert len(source.splitlines()) < 70


def test_no_second_copy_of_the_callback_config_file() -> None:
    assert not list(SHIM_DIR.glob("*.yaml"))
    assert not list(PKG_DIR.glob("*.yaml"))


def test_shim_import_fails_loudly_when_the_package_cannot_be_found(tmp_path: Path) -> None:
    shim_copy = tmp_path / "litellm" / "integrations" / "loci"
    shim_copy.mkdir(parents=True)
    target = shim_copy / "__init__.py"
    target.write_text((SHIM_DIR / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('shimcopy', {str(target)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "try:\n"
        "    spec.loader.exec_module(m)\n"
        "except ImportError as e:\n"
        "    print('IMPORTERROR', e)\n"
        "else:\n"
        "    print('IMPORTED')\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert "IMPORTERROR" in out.stdout, out.stdout + out.stderr
    assert "loci_litellm" in out.stdout and "litellm-loci" in out.stdout


# --- identifiable version -------------------------------------------------------------------------

def test_runtime_version_matches_pyproject() -> None:
    import loci_litellm

    declared = tomllib.loads((PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert loci_litellm.__version__ == declared
    assert loci_litellm.__version__ != "unknown"


def test_shim_exposes_the_same_version() -> None:
    import litellm.integrations.loci as shim
    import loci_litellm

    assert shim.__version__ == loci_litellm.__version__


def test_import_logs_version_and_path(caplog) -> None:
    import loci_litellm

    with caplog.at_level(logging.INFO, logger="loci_litellm"):
        importlib.reload(loci_litellm)
    messages = [r.getMessage() for r in caplog.records if r.name == "loci_litellm"]
    assert any(loci_litellm.__version__ in m and str(PKG_DIR) in m for m in messages), messages


def test_pyproject_declares_every_third_party_import() -> None:
    deps = tomllib.loads((PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    names = {re.split(r"[<>=!~ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert {"httpx", "litellm", "pyyaml"} <= names


def test_dedup_gate_is_no_longer_exported() -> None:
    import litellm.integrations.loci as shim
    import loci_litellm

    assert not hasattr(loci_litellm, "LociDedupGate") and "loci_dedup_gate" not in loci_litellm.__all__
    assert not hasattr(shim, "loci_dedup_gate")


# --- numeric env at import is loud ----------------------------------------------------------------

def test_non_numeric_recall_env_fails_the_import_instead_of_being_ignored() -> None:
    code = f"import sys; sys.path.insert(0, {str(PKG_ROOT)!r}); import loci_litellm.grounding_hook"
    out = _run(code, {"LOCI_RECALL_TOP_K": "abc"})
    assert out.returncode != 0 and "ValueError" in out.stderr


# --- README matches the code ----------------------------------------------------------------------

_ENV = r"(?:LOCI|MNEMOSYNE|QDRANT|OLLAMA|EMBED)_[A-Z0-9_]+"


def _code_env_names() -> set[str]:
    found: set[str] = set()
    for path in PKG_DIR.glob("*.py"):
        found |= set(re.findall(rf"""["']({_ENV})["']""", path.read_text(encoding="utf-8")))
    return found - {"LOCI_GROUNDING_STATUS"}  # a label in the injected notice, not an env var


_CHANGES_HEADING = "## Behavior changes since 0.1.0"


def _readme_text(include_changes: bool = False) -> str:
    """The README without its behavior-changes section, which names removed settings on purpose (a
    separate test pins that those names really are gone)."""
    text = (PKG_ROOT / "README.md").read_text(encoding="utf-8")
    if include_changes or _CHANGES_HEADING not in text:
        return text
    head, _, tail = text.partition(_CHANGES_HEADING)
    nxt = re.search(r"(?m)^## ", tail)
    return head + (tail[nxt.start():] if nxt else "")


def _readme_names() -> set[str]:
    text = _readme_text()
    names = set(re.findall(rf"`({_ENV})`", text))
    # suffix shorthand in one table row: `LOCI_RANKER_W_RELEVANCE`, `_RECENCY`, ...
    for prefix in ("LOCI_RANKER_W_", "LOCI_LOCAL_SLO_"):
        for line in text.splitlines():
            if f"`{prefix}" in line:
                names |= {prefix + s for s in re.findall(r"`_([A-Z0-9_]+)`", line.split("|")[1])}
    return names


def test_every_env_var_the_code_reads_is_in_the_readme() -> None:
    missing = sorted(_code_env_names() - _readme_names())
    assert missing == [], f"read by the code but not documented: {missing}"


def test_readme_documents_no_env_var_the_code_does_not_read() -> None:
    phantom = sorted(_readme_names() - _code_env_names())
    assert phantom == [], f"documented but never read: {phantom}"


def test_readme_does_not_repeat_removed_or_false_claims() -> None:
    text = _readme_text().replace("is not a no-op", "")
    for needle in ("uv add", "gitignored", "no-op", "proceeds unchanged", "LOCI_CB_RECOVERY_WINDOW",
                   "LOCI_QDRANT_WORKERS", "gen_url", "gen_model", "LOCI_DEDUP_"):
        assert needle not in text, needle


def test_the_removed_settings_listed_in_the_readme_are_really_gone() -> None:
    """The 'Removed settings' bullet is a promise: none of those env vars or resolver methods may still be
    read, otherwise the README tells operators a live setting does nothing."""
    from loci_litellm.backend_resolver import BackendResolver

    section = _readme_text(include_changes=True).partition(_CHANGES_HEADING)[2]
    lines = section.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("- **Removed settings")), None)
    assert start is not None, "README lost its 'Removed settings' bullet"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("- **")), len(lines))
    tokens = re.findall(r"`([A-Za-z0-9_*/ ]+)`", "\n".join(lines[start:end]))
    code = _code_env_names()
    checked = 0
    for token in tokens:
        for name in (t.strip() for t in token.split("/")):
            if name.endswith("*"):
                assert not any(c.startswith(name[:-1]) for c in code), f"{name} is still read by the code"
            elif re.fullmatch(_ENV, name):
                assert name not in code, f"{name} is still read by the code"
            elif re.fullmatch(r"[a-z_]+", name):
                assert not hasattr(BackendResolver, name), f"BackendResolver.{name} still exists"
            else:
                continue
            checked += 1
    assert checked >= 8, f"only {checked} names were checkable; the bullet or the parser changed"

