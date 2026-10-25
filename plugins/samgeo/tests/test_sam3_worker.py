"""SAM 3 adapter, environment selection, access failures and secret-safe transport."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_samgeo import access, client, environment, runs, segment
from ppg_samgeo.worker_sam3 import Sam3


def test_sam3_concepts_masks_and_unknown_quality():
    calls = []

    class Model:
        def set_image(self, image):
            calls.append(image.shape)

        def set_confidence_threshold(self, value):
            calls.append(value)

        def generate_masks(self, text, quiet):
            calls.append(text)
            self.masks = np.zeros((3, 1, 20, 30), dtype=bool)
            self.masks[0, 0, 4:9, 8:16] = True
            self.masks[1, 0, 10:13, 3:7] = True
            self.scores = np.array([0.8, 0.1, 0.9])

    adapter = Sam3.__new__(Sam3)
    adapter.sam = Model()
    result = adapter.segment(np.zeros((20, 30, 3), np.uint8), "stone wall. brick. stone wall.", 0.3, 0.25)
    assert calls == [(20, 30, 3), 0.3, "stone wall", "brick"]
    assert len(result) == 2 and {o["label"] for o in result} == {"stone wall", "brick"}
    assert result[0]["box"] == [8, 4, 16, 9]
    assert result[0]["quality"] is None
    r, c, mask = client.unpack(result[0])
    assert (r, c) == (4, 8) and mask.shape == (5, 8) and mask.all()
    adapter.sam.scores = []
    with pytest.raises(ValueError, match="threshold"):
        adapter.segment(np.zeros((20, 30, 3)), "stone", 2, 0.25)


def test_missing_access_before_loading_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: None))
    monkeypatch.setitem(sys.modules, "samgeo", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "samgeo.samgeo3", SimpleNamespace(SamGeo3=lambda **kw: pytest.fail("loaded weights")))
    with pytest.raises(RuntimeError, match="SAM3_ACCESS_REQUIRED"):
        Sam3()


def test_sam3_uses_feature_environment(monkeypatch):
    features = []
    monkeypatch.setattr(environment, "worker_python", lambda feature: features.append(feature) or Path(sys.executable))
    monkeypatch.setenv("PPG_SAMGEO_FAKE", "1")
    with client.Segmenter(engine="sam3") as model:
        assert model.device == "fake"
    assert features == ["sam3"]


def test_gated_error_is_not_reported_as_connectivity_problem():
    hint = client._hint("GatedRepoError: 403 https://huggingface.co/facebook/sam3")
    assert "Meta" in hint and "token" in hint and "internet" not in hint


def test_redact_before_stderr_truncation(monkeypatch):
    secret = "not-a-standard-token-format"
    monkeypatch.setenv("HF_TOKEN", "hf_exampleForUnitTestOnly1234")
    lines = []
    proc = SimpleNamespace(stderr=io.BytesIO((secret + " hf_exampleForUnitTestOnly1234 " + "x" * 500 + "\n").encode()))
    client.Segmenter._drain(proc, lines, secret)
    assert secret not in "".join(lines) and "hf_example" not in "".join(lines)
    assert access.redact(f"Bearer {secret}", secret) == "Bearer [REDACTED]"
    assert "hf_" not in access.redact("traceback: hf_anotherTokenForTesting")


def test_token_redacted_from_worker_error_and_traceback(tmp_path):
    # Test the process boundary, not just the redaction helper.
    script = tmp_path / "bad_worker.py"
    script.write_text('''import sys, pickle, struct
n = struct.unpack('>Q', sys.stdin.buffer.read(8))[0]
r = pickle.loads(sys.stdin.buffer.read(n))
secret = r['token']
print('library error: ' + secret, file=sys.stderr, flush=True)
b = pickle.dumps({'ok': False, 'error': '401 ' + secret, 'traceback': secret})
sys.stdout.buffer.write(struct.pack('>Q', len(b)) + b)
sys.stdout.buffer.flush()
''')
    secret = "test-credential-never-persist"
    with pytest.raises(client.WorkerError) as caught:
        client.Segmenter(engine="sam3", token=secret, python=Path(sys.executable), script=script)
    assert secret not in str(caught.value) + caught.value.details
    assert "[REDACTED]" in caught.value.details


def test_unknown_mask_quality_survives_pipeline(tmp_path):
    import samples

    _, chunk = samples.project(tmp_path / "project")
    plan = segment.plan(chunk.orthomosaic_path)

    class Predictor:
        def segment(self, image, *args, **kw):
            from ppg_samgeo.worker import _cut
            mask = np.zeros(image.shape[:2], bool)
            mask[40:80, 40:80] = True
            return [{"box": [40, 40, 80, 80], "score": 0.8, "quality": None,
                     "label": "stone", "mask": _cut(mask)}]

    objects = segment.find(Predictor(), plan, "stone")
    assert objects and objects[0]["quality"] is None
    run = runs.write(chunk, plan, objects, {"text": "stone", "prompt": "stone", "engine": "sam3"})
    assert runs.features(chunk, run)[0][1]["quality"] is None
    assert "token" not in json.dumps(run)


def test_probe_checks_sam3_symbols_and_distribution_names(monkeypatch):
    seen = []
    def invoke(cmd, **kwargs):
        seen.append(cmd[-1])
        return SimpleNamespace(stdout='{"versions": {}, "device": "cpu"}', stderr="", returncode=0)
    monkeypatch.setattr(environment.subprocess, "run", invoke)
    environment.probe(Path(sys.executable), ["samgeo.samgeo3", "geoai"])
    assert "from transformers import Sam3Model, Sam3Processor" in seen[0]
    assert '"geoai": "geoai-py"' in seen[0]


def test_unknown_feature_does_not_touch_environment(tmp_path):
    with pytest.raises(ValueError, match="Unknown AI features"):
        environment.install(features=("typo",), env=tmp_path / "missing", uv=Path("uv"))
    assert not (tmp_path / "missing").exists()
