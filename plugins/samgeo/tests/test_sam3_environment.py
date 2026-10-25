import pytest

from ppg_samgeo import environment


def test_sam3_requires_compatible_local_runtime(monkeypatch):
    monkeypatch.setattr(environment.sys, "frozen", False, raising=False)
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(environment.importlib.metadata, "version",
                        lambda name: {"segment-geospatial": "1.4.2", "transformers": "5.0.0"}[name])
    assert environment.local_available("sam3")
    monkeypatch.setattr(environment.importlib.metadata, "version",
                        lambda name: {"segment-geospatial": "1.3.9", "transformers": "5.0.0"}[name])
    assert not environment.local_available("sam3")
    monkeypatch.setattr(environment.importlib.metadata, "version",
                        lambda name: {"segment-geospatial": "1.4.2", "transformers": "4.57.0"}[name])
    assert not environment.local_available("sam3")


def test_compatible_managed_sam3_environment_precedes_incompatible_host(monkeypatch):
    monkeypatch.setattr(environment, "installed_info", lambda env=None: {
        "features": ["sam3"], "versions": {"segment-geospatial": "1.4.2", "transformers": "5.20.0"}})
    monkeypatch.setattr(environment, "local_available", lambda feature="samgeo": feature == "sam3")
    assert environment.where("sam3") == "environment"
    monkeypatch.setattr(environment, "local_available", lambda feature="samgeo": False)
    monkeypatch.setattr(environment, "installed_info", lambda env=None: {
        "features": ["sam3"], "versions": {"segment-geospatial": "1.2.0", "transformers": "5.20.0"}})
    assert environment.where("sam3") is None


def test_sam3_help_and_status_do_not_reuse_open_model_estimate(monkeypatch):
    monkeypatch.setattr(environment, "where", lambda feature="samgeo": None)
    monkeypatch.setattr(environment, "installed_info", lambda env=None: {})
    assert "--sam3" in environment.status_text("sam3") or "facebook/sam3" in environment.status_text("sam3")
    with pytest.raises(environment.EnvironmentMissing, match="--sam3"):
        environment.worker_python("sam3")

