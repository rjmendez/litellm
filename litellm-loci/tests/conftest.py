"""Isolate every test from the machine's private Loci config and from process-wide resolver caches."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_backend_resolver(monkeypatch, tmp_path_factory):
    from loci_litellm import backend_resolver as br

    none = tmp_path_factory.mktemp("loci_cfg") / "none.toml"
    monkeypatch.setattr(br, "_CONFIG_PATH", str(none))
    monkeypatch.delenv("LOCI_CONFIG", raising=False)  # a set-but-missing file would warn in every test
    br._config.cache_clear()
    br.BackendResolver.ollama_url.cache_clear()
    monkeypatch.setattr(br, "_config_error", "")
    yield
    br._config.cache_clear()
    br.BackendResolver.ollama_url.cache_clear()
