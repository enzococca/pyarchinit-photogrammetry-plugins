"""The proxies modelled in Blender back into the chunk: measured as the program measures its own,
paired with their units, the files of the inbox taken away."""
from __future__ import annotations

import json

import numpy as np
import pytest
from samples import ORIGIN, emjson, project

from ppg_blender_em import emjson as em
from ppg_blender_em import inbox, scene


def _box(x0, y0, z0, x1, y1, z1):
    import trimesh

    return trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])


def _send(folders, name, mesh, shift, node_id=None):
    """A proxy as the live script in Blender sends it: Y-up, shifted, with its description."""
    import trimesh

    folder = inbox.inbox(folders)
    folder.mkdir(parents=True, exist_ok=True)
    local = scene.to_gltf(np.asarray(mesh.vertices) - np.asarray(shift))
    trimesh.Trimesh(local, mesh.faces, process=False).export(str(folder / f"{name}.glb"))
    (folder / f"{name}.json").write_text(json.dumps({"name": name, "node_id": node_id, "shift": list(shift)}),
                                         encoding="utf-8")


def test_measures_of_a_solid_and_of_an_open_mesh():
    box = _box(0, 0, 10, 2, 3, 10.5)
    m = inbox.measure(box.vertices, box.faces)
    assert m["closed"] and m["volume_m3"] == pytest.approx(3.0) and m["area_m2"] == pytest.approx(6.0)
    assert m["z_top"] == pytest.approx([10.5, 10.5]) and m["z_bottom"] == pytest.approx([10.0, 10.0])
    # the top taken away (a hole of two triangles) is closed first
    top = np.asarray(box.face_normals)[:, 2] > 0.5
    m = inbox.measure(box.vertices, box.faces[~top])
    assert m["closed"] and m["volume_m3"] == pytest.approx(3.0) and m["z_top"] == pytest.approx([10.5, 10.5])
    # a sheet has no volume
    sheet = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    m = inbox.measure(sheet, [[0, 1, 2], [0, 2, 3]])
    assert not m["closed"] and m["volume_m3"] == 0.0 and m["area_m2"] == pytest.approx(1.0)


def test_take_the_proxies_sent_from_blender(tmp_path):
    doc = project(tmp_path, model=False)
    chunk = doc.chunk
    chunk.meta["proxies"][1]["id_media"] = 7  # USM 2 published in PyArchInit
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    ids = emjson(em_path, units=[("1", "US", "1"), ("1", "USM", "2"), ("1", "US", "3"), ("1", "USVs", "101")])
    matrix = em.load(em_path)
    folders = scene.project_folders(emjson=em_path)
    shift = [ORIGIN[0], ORIGIN[1], ORIGIN[2] - 1.0]
    x0, y0, z0 = ORIGIN
    # USM 2 modelled again in Blender, under the code of the graph EMtools puts before the names
    _send(folders, "SCAVO.1.USM2", _box(x0 + 1.8, y0 + 0.5, z0 - 0.4, x0 + 2.9, y0 + 2.5, z0 + 0.2), shift)
    # a virtual unit modelled in Blender, by the id of its node
    usv = next(k for k, v in ids.items() if v[2] == "101")
    _send(folders, "Cube.001", _box(x0 + 4.0, y0 + 0.5, z0, x0 + 5.0, y0 + 1.5, z0 + 2.0), shift, node_id=usv)
    # a mesh that does not read stays in the inbox
    (inbox.inbox(folders) / "broken.glb").write_bytes(b"not a glb")
    (inbox.inbox(folders) / "broken.json").write_text("{}", encoding="utf-8")
    assert [p.stem for p in inbox.pending(folders)] == ["Cube.001", "SCAVO.1.USM2", "broken"]
    report = inbox.take(chunk, folders, matrix)
    assert report["taken"] == ["USVs101", "USM2"] and report["replaced"] == ["USM2"]
    assert report["republish"] == ["USM2"] and list(report["errors"]) == ["broken.glb"]
    assert [p.stem for p in inbox.pending(folders)] == ["broken"]
    entries = {e["name"]: e for e in chunk.meta["proxies"]}
    assert list(entries) == ["US1", "USM2", "US3", "USVs101"]
    usm2 = entries["USM2"]
    assert usm2["base"] == "blender" and usm2["closed"] and usm2["id_media"] == 7 and usm2["sito"] == "Scavo"
    assert usm2["volume_m3"] == pytest.approx(1.1 * 2.0 * 0.6) and usm2["area_m2"] == pytest.approx(2.2)
    assert usm2["z_top"] == pytest.approx([z0 + 0.2] * 2) and usm2["z_bottom"] == pytest.approx([z0 - 0.4] * 2)
    assert entries["USVs101"]["tipo"] == "USVs" and entries["USVs101"]["us"] == "101"
    # in the world, beside the other proxies
    import trimesh

    mesh = trimesh.load(str(chunk.data_dir / usm2["file"]), process=False, force="mesh")
    assert np.asarray(mesh.vertices).min(0) == pytest.approx([x0 + 1.8, y0 + 0.5, z0 - 0.4], abs=1e-4)
    # the next export writes them with the names of their nodes
    out = scene.export_project(chunk, folders, matrix, survey=False)
    assert "1.USVs101" in out["written"] and "1.USM2" in out["written"]


def test_without_a_matrix_the_name_says_the_unit(tmp_path):
    doc = project(tmp_path, model=False)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    x0, y0, z0 = ORIGIN
    _send(folders, "2.US15", _box(x0, y0, z0, x0 + 1, y0 + 1, z0 + 1), [x0, y0, z0])
    report = inbox.take(doc.chunk, folders, None)
    assert report["taken"] == ["US15"]
    entry = doc.chunk.meta["proxies"][-1]
    assert (entry["area"], entry["tipo"], entry["us"]) == ("2", "US", "15")
    assert entry["volume_m3"] == pytest.approx(1.0)


def test_measured_without_networkx():
    """The program carries no networkx (trimesh's own repair needs it): the measures do without."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['networkx'] = None\n"
            "import numpy as np, trimesh\n"
            "from ppg_blender_em.inbox import measure\n"
            "box = trimesh.creation.box(bounds=[[0, 0, 0], [2, 3, 0.5]])\n"
            "top = np.asarray(box.face_normals)[:, 2] > 0.5\n"
            "m = measure(box.vertices, box.faces[~top])\n"
            "print(m['closed'], round(m['volume_m3'], 6))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["True", "3.0"]
