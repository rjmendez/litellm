"""Honesty tests for the deployment checker output and for backend resolution precedence."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from loci_litellm import backend_resolver as br
from loci_litellm.backend_resolver import BackendResolver

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
_spec = importlib.util.spec_from_file_location("check_deployment_honesty", DEPLOY / "check_deployment.py")
cd = importlib.util.module_from_spec(_spec)
sys.modules["check_deployment_honesty"] = cd
_spec.loader.exec_module(cd)


@pytest.fixture
def cfg():
    return yaml.safe_load((DEPLOY / "config.example.yaml").read_text(encoding="utf-8"))


# --- deploy checker ---------------------------------------------------------------------------------

def test_a_clean_run_without_env_says_the_env_half_was_not_checked(tmp_path: Path, cfg, capsys) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cd.main(["--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "NOTE" in out and "--env" in out and "not checked" in out
    assert "0 violation(s)" in out


def test_a_run_with_env_does_not_print_the_note(tmp_path: Path, cfg, capsys) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("SOMETHING=1\n", encoding="utf-8")
    assert cd.main(["--config", str(path), "--env", str(env)]) == 0
    assert "NOTE" not in capsys.readouterr().out


def test_ip_behind_an_api_base_variable_is_caught_whatever_the_variable_is_called(cfg) -> None:
    deployment = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-simple")
    deployment["litellm_params"]["api_base"] = "os.environ/PROVIDER_ENDPOINT_ADDR"
    violations, _ = cd.check(cfg, {"PROVIDER_ENDPOINT_ADDR": "http://203.0.113.9:8080"})
    assert any(v.startswith("no-ip-literal") and "PROVIDER_ENDPOINT_ADDR" in v for v in violations)


def test_a_variable_that_is_neither_named_like_a_url_nor_referenced_is_not_flagged(cfg) -> None:
    violations, _ = cd.check(cfg, {"WORKER_ADDR": "203.0.113.9"})
    assert violations == []


def test_exemption_from_pricing_is_by_variable_name_only(cfg) -> None:
    """The README says so; this pins it so the README cannot silently become untrue."""
    deployment = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-simple")
    deployment["litellm_params"]["model"] = "openai/not-in-any-cost-map"
    deployment["model_info"].pop("input_cost_per_token", None)
    deployment["model_info"].pop("output_cost_per_token", None)
    deployment["litellm_params"].pop("input_cost_per_token", None)
    deployment["litellm_params"].pop("output_cost_per_token", None)
    deployment["litellm_params"]["api_base"] = "https://api.cloud.example.test/v1"
    violations, _ = cd.check(cfg)
    assert any(v.startswith("priced") for v in violations)
    deployment["litellm_params"]["api_base"] = "os.environ/OLLAMA_REMOTE_API"  # name alone grants the exemption
    violations, _ = cd.check(cfg)
    assert not any(v.startswith("priced") and "cloud-simple" in v for v in violations)


def test_checker_makes_no_claim_about_callbacks(cfg) -> None:
    cfg["litellm_settings"]["callbacks"] = []
    violations, warnings = cd.check(cfg)
    assert violations == [] and warnings == []


# --- backend resolver ---------------------------------------------------------------------------------

@pytest.fixture
def no_probe(monkeypatch):
    monkeypatch.setattr(br, "_alive", lambda url, timeout=1.0: False)
    for name in ("LOCI_OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "MNEMOSYNE_EMBEDDING_API_URL", "LOCI_QDRANT_URL",
                 "QDRANT_URL", "LOCI_QDRANT_API_KEY", "QDRANT_API_KEY", "LOCI_EMBED_MODEL", "MNEMOSYNE_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    BackendResolver.ollama_url.cache_clear()


def test_embed_api_base_appends_v1_and_strips_a_trailing_slash(monkeypatch, no_probe) -> None:
    monkeypatch.setenv("LOCI_OLLAMA_BASE_URL", "https://ollama.example.test:11434/")
    assert BackendResolver.embed_api_base() == "https://ollama.example.test:11434/v1"


def test_ollama_url_wins_over_the_mnemosyne_endpoint(monkeypatch, no_probe) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.test")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://embed.example.test/v1/")
    assert BackendResolver.embed_api_base() == "https://ollama.example.test/v1"


def test_mnemosyne_endpoint_is_used_only_when_no_ollama_url_resolves(monkeypatch, no_probe) -> None:
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://embed.example.test/v1/")
    assert BackendResolver.embed_api_base() == "https://embed.example.test/v1"


def test_nothing_configured_resolves_to_nothing_not_a_guess(no_probe) -> None:
    assert BackendResolver.embed_api_base() is None
    assert BackendResolver.qdrant() == ("", "")


def test_loci_prefixed_env_beats_unprefixed_and_both_beat_the_config_file(monkeypatch, no_probe, tmp_path: Path) -> None:
    cfg_file = tmp_path / "b.toml"
    cfg_file.write_text('[qdrant]\nurl = "https://cfg.example.test"\napi_key = "from-config"\n', encoding="utf-8")
    monkeypatch.setattr(br, "_CONFIG_PATH", str(cfg_file))
    br._config.cache_clear()
    assert BackendResolver.qdrant() == ("https://cfg.example.test", "from-config")
    monkeypatch.setenv("QDRANT_URL", "https://plain.example.test")
    assert BackendResolver.qdrant_url() == "https://plain.example.test"
    monkeypatch.setenv("LOCI_QDRANT_URL", "https://loci.example.test")
    assert BackendResolver.qdrant_url() == "https://loci.example.test"


def test_a_failed_probe_is_remembered_until_restart(monkeypatch, no_probe) -> None:
    """Documented behaviour: the first answer, including 'nothing reachable', is memoised."""
    assert BackendResolver.ollama_url() == ""
    monkeypatch.setenv("LOCI_OLLAMA_BASE_URL", "https://late.example.test")
    assert BackendResolver.ollama_url() == ""  # env set after first call is not seen
    BackendResolver.ollama_url.cache_clear()
    assert BackendResolver.ollama_url() == "https://late.example.test"


@pytest.mark.parametrize("url", ["", "not a url", "http://", "http://127.0.0.1:1"])
def test_alive_is_false_for_unusable_or_unreachable_urls(url: str) -> None:
    assert br._alive(url, timeout=0.2) is False


def test_alive_is_true_for_a_listening_socket() -> None:
    import socket

    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        assert br._alive(f"http://127.0.0.1:{srv.getsockname()[1]}", timeout=1.0) is True


def test_the_conftest_isolates_the_private_config() -> None:
    assert not Path(br._CONFIG_PATH).exists()
    assert br.config_load_error() == ""
