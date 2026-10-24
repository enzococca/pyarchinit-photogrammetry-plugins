"""The EM project written from a chunk: the proxies named as the nodes of the matrix, the survey,
the shift, the manifest, and the files another program wrote left alone."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from samples import ORIGIN, emjson, project

from ppg_blender_em import emjson as em
from ppg_blender_em import scene


def _glb(path):
    """The JSON chunk and the positions of the first mesh of a GLB."""
    import trimesh

    raw = path.read_bytes()
    length = struct.unpack("<I", raw[12:16])[0]
    doc = json.loads(raw[20:20 + length])
    loaded = trimesh.load(str(path), force="scene")
    verts = np.vstack([g.apply_transform(t).vertices for g, t in
                       ((loaded.geometry[loaded.graph[n][1]].copy(), loaded.graph[n][0])
                        for n in loaded.graph.nodes_geometry)])
    return doc, verts


def test_the_standard_tree_of_the_em_json(tmp_path):
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path)
    folders = scene.project_folders(emjson=em_path)
    assert folders.standard and folders.base == (tmp_path / "Scavo").resolve()
    assert folders.proxies == folders.base / "EM" / "proxies" and folders.survey == folders.base / "RB"
    loose = tmp_path / "loose.em.json"
    emjson(loose)
    folders = scene.project_folders(emjson=loose)
    assert not folders.standard and folders.proxies == tmp_path.resolve() / "proxies"


def test_export_the_proxies_and_the_survey(tmp_path):
    doc = project(tmp_path)
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path)
    matrix = em.load(em_path)
    folders = scene.project_folders(emjson=em_path)
    report = scene.export_project(doc.chunk, folders, matrix)
    assert report["written"] == ["1.US1", "1.USM2", "1.US3", "Rilievo Chunk 1"]
    assert report["not_in_matrix"] == [] and report["foreign"] == []
    shift = report["shift"]
    # the lowest corner of what is written, rounded down to the metre: the bottoms of the proxies
    assert shift == [ORIGIN[0], ORIGIN[1], ORIGIN[2] - 1.0]
    files = sorted(p.name for p in folders.proxies.glob("*.glb"))
    assert files == ["1.US1.glb", "1.US3.glb", "1.USM2.glb"]
    gltf, verts = _glb(folders.proxies / "1.USM2.glb")
    assert gltf["nodes"][0]["name"] == "1.USM2" and gltf["meshes"][0]["name"] == "1.USM2"
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][:3] == pytest.approx(
        [0xCC / 255, 0x66 / 255, 0x33 / 255])  # the colour of its epoch
    # Y-up and shifted: back to Z-up and to the world, it is where the program built it
    import trimesh

    entry = doc.chunk.meta["proxies"][1]
    world = trimesh.load(str(doc.chunk.data_dir / entry["file"]), process=False, force="mesh").vertices
    back = scene.from_gltf(verts) + shift
    assert back.min(0) == pytest.approx(np.asarray(world).min(0), abs=1e-4)
    assert back.max(0) == pytest.approx(np.asarray(world).max(0), abs=1e-4)
    # the survey, shifted the same way
    survey = folders.survey / "Chunk 1" / "Chunk 1.glb"
    gltf, verts = _glb(survey)
    assert gltf["nodes"][0]["name"] == "Rilievo Chunk 1"
    back = scene.from_gltf(verts) + shift
    model = doc.chunk.to_world(np.asarray(doc.chunk.model.vertices))
    assert back.min(0) == pytest.approx(model.min(0), abs=1e-3)
    assert back.max(0) == pytest.approx(model.max(0), abs=1e-3)
    assert back.max(0)[:2] == pytest.approx([ORIGIN[0] + 5.0, ORIGIN[1] + 3.0], abs=1e-3)
    manifest = scene.read_manifest(folders.base)
    assert manifest["revision"] == 1 and manifest["shift"] == shift and manifest["epsg"] == "32633"
    assert [(o["name"], o["kind"], o["node_id"] is not None) for o in manifest["objects"][:3]] == [
        ("1.US1", "proxy", True), ("1.USM2", "proxy", True), ("1.US3", "proxy", True)]
    assert manifest["objects"][0]["epoch"] == "Tardoantico" and manifest["objects"][3]["kind"] == "survey"
    assert manifest["emjson"] == str(em_path.resolve())


def test_again_only_what_changed(tmp_path):
    doc = project(tmp_path)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    first = scene.export_project(doc.chunk, folders, survey=False)
    assert first["written"] == ["US1", "USM2", "US3"] and first["not_in_matrix"] == ["US1", "USM2", "US3"]
    again = scene.export_project(doc.chunk, folders, survey=False)
    assert again["written"] == [] and again["kept"] == ["US1", "USM2", "US3"] and again["revision"] == 1
    # the proxy of US 3 built again, deeper: only it is written, the manifest moves on
    import trimesh

    entry = doc.chunk.meta["proxies"][2]
    f = doc.chunk.data_dir / entry["file"]
    mesh = trimesh.load(str(f), process=False, force="mesh")
    mesh.vertices[:, 2] -= 0.1
    mesh.export(str(f))
    third = scene.export_project(doc.chunk, folders, survey=False)
    assert third["written"] == ["US3"] and third["revision"] == 2
    # US 1 taken out of the chunk: its file goes
    doc.chunk.meta["proxies"] = doc.chunk.meta["proxies"][1:]
    fourth = scene.export_project(doc.chunk, folders, survey=False)
    assert fourth["removed"] == ["US1"] and not (folders.proxies / "US1.glb").exists()
    assert [o["name"] for o in scene.read_manifest(folders.base)["objects"]] == ["USM2", "US3"]


def test_the_shift_of_the_matrix_and_the_shift_kept(tmp_path):
    doc = project(tmp_path)
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path, shift=(299000.0, 4499000.0, 0.0))
    folders = scene.project_folders(emjson=em_path)
    report = scene.export_project(doc.chunk, folders, em.load(em_path), survey=False)
    assert report["shift"] == [299000.0, 4499000.0, 0.0]
    # without the matrix the shift of the manifest stays: Blender's objects do not move
    report = scene.export_project(doc.chunk, folders, None, survey=False)
    assert report["shift"] == [299000.0, 4499000.0, 0.0]


def test_a_file_of_another_program_is_left_alone(tmp_path):
    doc = project(tmp_path)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    folders.proxies.mkdir(parents=True)
    (folders.proxies / "USM2.glb").write_bytes(b"glTF written by EMtools")
    report = scene.export_project(doc.chunk, folders, survey=False)
    assert report["foreign"] == ["USM2"] and report["written"] == ["US1", "US3"]
    assert (folders.proxies / "USM2.glb").read_bytes() == b"glTF written by EMtools"
    # written here, then changed by EMtools: left alone too, and never removed
    (folders.proxies / "US1.glb").write_bytes(b"modelled in Blender")
    doc.chunk.meta["proxies"] = []
    report = scene.export_project(doc.chunk, folders, survey=False)
    assert report["removed"] == ["US3"] and (folders.proxies / "US1.glb").exists()


def test_names_that_cannot_be_files(tmp_path):
    assert scene.safe_name("1.US12") == "1.US12"
    assert scene.safe_name("A/B.US1") is None and scene.safe_name("..") is None and scene.safe_name("") is None


def test_vertex_colours_are_written_linear():
    # glTF keeps vertex colours linear: the sRGB of a photo, written as it is, looks pale in Blender
    rgba = np.array([[0, 128, 255, 200], [10, 188, 64, 255]], np.uint8)
    lin = scene.linear_colours(rgba)
    assert lin.dtype == np.uint8 and lin[:, 3].tolist() == [200, 255]  # alpha is not a colour
    assert lin[0, :3].tolist() == [0, 55, 255] and lin[1, :3].tolist() == [1, 128, 13]
