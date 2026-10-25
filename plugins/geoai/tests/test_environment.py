"""Standalone activation and preservation of the shared AI environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ppg_geoai import environment


def test_import_and_activate_without_samgeo(tmp_path):
    code = r'''
import importlib.abc, sys
class NoSamgeo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ppg_samgeo" or fullname.startswith("ppg_samgeo."):
            raise ImportError("SamGeo is not installed")
sys.meta_path.insert(0, NoSamgeo())
from ppg_geoai import GeoaiPlugin
from ppg_geoai.pane import GeoaiPane
from pyarchinit_photogrammetry.plugins.api import Context
plugin = GeoaiPlugin()
ctx = Context("geoai", __import__("pathlib").Path(sys.argv[1]))
plugin.activate(ctx)
assert "geoai" in ctx.contributions.commands
assert not any(m == "ppg_samgeo" or m.startswith("ppg_samgeo.") for m in sys.modules)
assert "torch" not in sys.modules and "geoai" not in sys.modules
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_install_geoai_preserves_samgeo_and_sam3_features(tmp_path, monkeypatch):
    env = tmp_path / "geoai-env"
    python = environment.env_python(env)
    python.parent.mkdir(parents=True)
    python.touch()
    (env / environment.MARKER).write_text(json.dumps({"features": ["samgeo", "sam3"], "device": "mps"}))
    commands = []
    monkeypatch.setattr(environment, "_run", lambda cmd, *args: commands.append(cmd))
    monkeypatch.setattr(environment, "probe", lambda *args: {
        "python": "3.12.12", "versions": {"torch": "2.14.1", "transformers": "5.20.2",
                                          "samgeo.samgeo3": "1.4.2", "geoai": "0.25.0"}, "device": "mps"})
    monkeypatch.setattr("pyarchinit_photogrammetry.ai.sam_env.torch_backend_args", lambda *args: [])
    environment.install(features=("geoai",), env=env, uv=Path("/test/uv"))
    info = environment.installed_info(env)
    assert info["features"] == ["samgeo", "sam3", "geoai"]
    packages = commands[1]
    assert all(p in packages for p in ("sam2", "segment-geospatial>=1.4.2,<2", "transformers>=5,<6",
                                      "geoai-py", "segmentation-models-pytorch"))
    assert info["device"] == "mps"


def test_missing_geoai_has_its_own_install_command(monkeypatch):
    monkeypatch.setattr(environment, "where", lambda *args: None)
    assert "ppg plugin geoai install" in environment.status_text("geoai")
    with pytest.raises(environment.EnvironmentMissing, match="ppg plugin geoai install"):
        environment.worker_python("geoai")
