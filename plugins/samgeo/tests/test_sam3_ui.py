"""SAM 3 selection, onboarding and credentials kept outside project metadata."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ppg_samgeo import SamgeoPlugin, environment, run_info, run_job


@pytest.fixture
def pane():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ppg_samgeo.pane import SamgeoPane

    qapp = QApplication.instance() or QApplication([])
    widget = SamgeoPane()
    widget.show()
    qapp.processEvents()
    yield widget
    widget.e_token.clear()
    widget.close()


def test_engine_selection_and_session_token_are_separate(pane):
    from PySide6.QtWidgets import QLineEdit

    assert pane.engine() == "open"
    assert pane.c_models.isEnabled()
    assert not pane.sam3_holder.isVisibleTo(pane)
    pane.c_engine.setCurrentIndex(pane.c_engine.findData("sam3"))
    assert pane.sam3_holder.isVisibleTo(pane)
    assert not pane.c_models.isEnabled()
    assert pane.e_token.echoMode() == QLineEdit.EchoMode.Password
    pane.e_token.setText("hf_private_test_credential")
    assert pane.session_token() == "hf_private_test_credential"
    assert pane.values()["engine"] == "sam3"
    assert "token" not in pane.values()
    assert "hf_private_test_credential" not in json.dumps(pane.values())
    assert "Meta" in pane.l_access.text() and "Hugging Face" in pane.l_access.text()
    pane.c_engine.setCurrentIndex(0)
    assert pane.session_token() is None
    assert pane.c_models.isEnabled()


def test_run_metadata_ignores_credentials_and_identifies_model():
    params = {"engine": "sam3", "text": "muri", "prompt": "stone wall.", "token": "hf_private_test_credential"}
    info = run_info(params, {"name": "orthomosaic"})
    assert info["engine"] == "sam3"
    assert info["backend"] == "transformers"
    assert info["model_id"] == "facebook/sam3"
    assert "detector" not in info and "models" not in info
    assert "token" not in info and "hf_private_test_credential" not in json.dumps(info)
    with pytest.raises(ValueError, match="unknown"):
        run_info({"engine": "unrecognized"}, {"name": "orthomosaic"})


def test_worker_receives_token_but_run_writer_does_not(monkeypatch):
    from ppg_samgeo import client, runs, segment

    seen = {}

    class Worker:
        device = "cpu"
        versions = {"torch": "test", "transformers": "test"}

        def __init__(self, *args, **kwargs):
            seen["credentials"] = kwargs
            assert args[0] == "sam3"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def write(chunk, plan, objects, info):
        seen["stored"] = info
        return dict(info)

    monkeypatch.setattr(client, "Segmenter", Worker)
    monkeypatch.setattr(segment, "find", lambda *args, **kwargs: [])
    monkeypatch.setattr(runs, "write", write)
    info = run_info({"engine": "sam3", "prompt": "stone wall."}, {"name": "orthomosaic"})
    result = run_job(None, None, info, token="hf_private_test_credential", device="cpu")
    assert seen["credentials"]["token"] == "hf_private_test_credential"
    assert seen["credentials"]["device"] == "cpu"
    assert "hf_private_test_credential" not in json.dumps(seen["stored"])
    assert result["device"] == "cpu" and result["versions"]["torch"] == "test"


def test_cli_access_does_not_echo_environment_token(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_private_test_credential")
    assert SamgeoPlugin().command("access") == 0
    output = capsys.readouterr().out
    assert "https://huggingface.co/facebook/sam3" in output
    assert "Meta" in output and "Hugging Face" in output
    assert "hf_private_test_credential" not in output


def test_install_and_status_select_sam3_feature(monkeypatch, capsys):
    wanted = []
    monkeypatch.setattr(environment, "install", lambda progress, features: wanted.append(features) or "test-environment")
    monkeypatch.setattr(environment, "status_text", lambda feature="samgeo": f"status {feature}")
    monkeypatch.setattr(environment, "where", lambda feature="samgeo": "environment" if feature == "sam3" else None)
    plugin = SamgeoPlugin()
    assert plugin.command("install", "--sam3") == 0
    assert wanted == [("sam3",)]
    assert "status sam3" in capsys.readouterr().out
    assert plugin.command("status", "--engine", "sam3") == 0
    assert "status sam3" in capsys.readouterr().out
    assert plugin.command("status") == 1


def test_find_checks_selected_engine_dependencies(monkeypatch):
    plugin = SamgeoPlugin()
    chunk = object()
    plugin.context = SimpleNamespace(app=SimpleNamespace(document=SimpleNamespace(chunk=chunk)))
    monkeypatch.setattr(plugin, "_rasters", lambda chunk: [{"name": "orthomosaic"}])
    wanted = []
    monkeypatch.setattr(plugin, "_ready", lambda then, feature: wanted.append(feature) or False)
    plugin.find({"engine": "sam3", "prompt": "stone.", "raster": "orthomosaic"})
    assert wanted == ["sam3"]
