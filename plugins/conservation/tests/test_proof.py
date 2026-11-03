"""The proof script works on a copy of the project it is given: the original is never written."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

import conservation_samples as S

HERE = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("run_proof", HERE / "proof" / "run_proof.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(doc_path: Path) -> dict:
    files = [doc_path] + sorted(p for p in Path(str(doc_path) + ".files").rglob("*") if p.is_file())
    return {str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}


def test_the_proof_runs_on_a_copy(tmp_path):
    doc, ch, entry = S.face(tmp_path / "progetto_vero")
    path = Path(doc.path)
    before = _state(path)
    out = tmp_path / "prova"
    proof = _script()
    assert proof.main([str(path), "--elevation", entry["name"], "--window", "0", str(S.Z0), "6", str(S.Z0 + 3),
                       "--out", str(out)]) == 0
    assert _state(path) == before  # the original untouched
    copy = out / "progetto" / path.name
    assert copy.exists() and (out / "progetto" / (path.name + ".files")).is_dir()
    import pyarchinit_photogrammetry as ppg

    copied = ppg.Document.open(copy)
    assert [c["name"] for c in copied.chunk.meta["conservation"]["campaigns"]] == ["Prova del plugin"]
    text = (out / "LEGGIMI.md").read_text(encoding="utf-8")
    assert "una copia di" in text and "l'originale non è stato toccato" in text and "Lesioni: 1" in text
    with pytest.raises(SystemExit, match="already there"):  # never over a copy made before
        proof.main([str(path), "--elevation", entry["name"], "--window", "0", str(S.Z0), "6", str(S.Z0 + 3),
                    "--out", str(out)])
    with pytest.raises(SystemExit, match="inside the project's data folder"):
        proof.main([str(path), "--elevation", entry["name"], "--window", "0", str(S.Z0), "6", str(S.Z0 + 3),
                    "--out", str(Path(str(path) + ".files") / "prova")])
    assert _state(path) == before
