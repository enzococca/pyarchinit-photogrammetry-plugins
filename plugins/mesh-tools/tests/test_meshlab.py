"""The operations with MeshLab itself (the real worker and pymeshlab), on synthetic models:
the texture coordinates kept by the decimation, the measures of a sphere, the borders kept by the
smoothing, the hole closed, a non-manifold model repaired, the edge of the remeshing, a filter that
refuses a model. Skipped when pymeshlab is not installed for this Python, or (not on CI, where it is
a failure) when it does not work there - on a Linux without libOpenGL, say: the reason says why."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mesh_samples import SCALE, bowtie_mesh, grid_mesh, project, sphere_mesh
from ppg_mesh_tools import environment, model, runner
from pyarchinit_photogrammetry.core import versions
from pyarchinit_photogrammetry.io.mesh_ply import MeshData, read_mesh


def _why_not() -> str | None:
    """Why the tests of MeshLab cannot run here, None when they can (the plugin's own check)."""
    if importlib.util.find_spec("pymeshlab") is None:
        return "pymeshlab is not installed for this Python"
    info = environment.probe(Path(sys.executable))
    if environment.usable(info) or os.environ.get("CI"):  # on CI a MeshLab that does not work is a failure
        return None
    return environment.problem_text(info)


WHY_NOT = _why_not()
pytestmark = pytest.mark.skipif(WHY_NOT is not None, reason=WHY_NOT or "")


def test_meshlab_works_here():
    """pymeshlab imports, reads a PLY and has every filter of the worker: else the message of the plugin
    (on Linux: the system packages to install) is the reason of the failure."""
    info = environment.probe(Path(sys.executable))
    assert environment.usable(info), environment.problem_text(info)


def _non_manifold_sphere():
    """A sphere with a fin on one of its edges (three faces on one edge) and three faces twice."""
    s = sphere_mesh()
    v, f = s.xyz, s.faces
    a, b = f[0][:2]
    extra = np.empty(1, s.vertices.dtype)
    extra["x"], extra["y"], extra["z"] = 3.0, 3.0, 3.0
    extra["red"], extra["green"], extra["blue"] = 0, 0, 0
    vertices = np.concatenate([s.vertices, extra])
    faces = np.vstack([f, [[a, b, len(v)]], f[:3]])
    return MeshData(vertices, faces)


def test_decimation_keeps_the_texture_coordinates(tmp_path):
    doc, ch = project(tmp_path)
    old = read_mesh(ch.product_path("model"))
    report = model.apply(ch, "decimate", {"target_faces": 600})
    assert report["texture_kept"] and abs(report["faces"] - 600) <= 2 and report["pymeshlab"]
    new = read_mesh(ch.product_path("model"))
    corners = new.xyz[new.faces][:, :, :2].reshape(len(new.faces), 6)
    assert np.abs(new.texcoords - corners).max() < 5e-3  # each corner keeps the coordinate of its place
    assert new.texture_files == ["model_texture.jpg"] and new.field("confidence") is not None
    assert new.colors is not None and set(new.vertices.dtype.names) == set(old.vertices.dtype.names)
    # the outline of the open surface is still where it was (borders kept)
    assert new.xyz[:, 0].min() == pytest.approx(0.0, abs=1e-6) and new.xyz[:, 0].max() == pytest.approx(1.0, abs=1e-6)


def test_measures_of_a_sphere(tmp_path):
    doc, ch = project(tmp_path, sphere_mesh(radius=1.0))
    m = model.measure(ch)
    assert m["area"] == pytest.approx(4 * np.pi * SCALE ** 2, rel=0.01)
    assert m["volume"] == pytest.approx(4 / 3 * np.pi * SCALE ** 3, rel=0.01) and not m["inverted"]
    assert m["watertight"] and m["two_manifold"] and m["components"] == 1 and m["boundary_loops"] == 0
    assert m["size"] == pytest.approx([2 * SCALE] * 3, rel=1e-3) and m["pymeshlab"]
    rows = dict(model.measure_rows(m))
    assert rows["Chiuso (watertight)"] == "sì" and rows["Volume"].endswith(" m³")


def test_smoothing_keeps_the_borders(tmp_path):
    doc, ch = project(tmp_path)
    old = read_mesh(ch.product_path("model"))
    report = model.apply(ch, "smooth", {"method": "taubin", "steps": 10})
    new = read_mesh(ch.product_path("model"))
    assert report["texture_kept"] and np.array_equal(new.faces, old.faces)
    border = model._boundary(old.faces, len(old.vertices))
    moved = np.linalg.norm(new.xyz - old.xyz, axis=1)
    assert moved[border].max() == 0.0 and moved[~border].max() > 1e-4
    assert np.array_equal(new.field("confidence"), old.field("confidence"))
    # Taubin does not shrink a closed model
    doc2, ch2 = project(tmp_path / "sphere", sphere_mesh())
    before = model.measure(ch2)["volume"]
    model.apply(ch2, "smooth", {"method": "taubin"})
    assert model.measure(ch2)["volume"] == pytest.approx(before, rel=0.005)


def test_the_hole_closed(tmp_path):
    doc, ch = project(tmp_path)
    before = model.measure(ch)
    assert before["boundary_loops"] == 2  # the outline and the hole
    report = model.apply(ch, "close_holes", {"max_hole_edges": 100})
    assert report["changed"] and report["details"]["closed_holes"] == 1 and report["details"]["new_faces"] > 10
    assert not report["texture_kept"] and ch.product_path("model_textured") is None
    assert report["measures"]["boundary_loops"] == 1 and report["measures"]["two_manifold"]
    # the outline is never offered: also with a limit it fits in, there is nothing more to close
    assert not model.apply(ch, "close_holes", {"max_hole_edges": 100000})["changed"]
    assert len(versions.versions(ch, "model")) == 1
    # asked for, the outline is closed too (MeshLab alone would have done it): the bottom of an object
    doc2, ch2 = project(tmp_path / "object", grid_mesh(n=20, hole=False, texture=False), texture=False)
    report = model.apply(ch2, "close_holes", {"max_hole_edges": 100000, "keep_outline": False})
    # a few edges stay where filling them would make faces cross (a wavy outline)
    assert report["changed"] and report["details"]["closed_holes"] == 1 and report["measures"]["boundary_edges"] < 10


def test_a_non_manifold_model_repaired(tmp_path):
    doc, ch = project(tmp_path, _non_manifold_sphere())
    m = model.measure(ch)
    assert not m["two_manifold"] and m["non_manifold_edges"] > 0
    # closing holes on it without the repair first: MeshLab refuses, the model stays
    before = (ch.data_dir / "model.ply").read_bytes()
    with pytest.raises(runner.WorkerError, match="manifold") as err:
        model.apply(ch, "close_holes", {"repair_first": False, "keep_outline": False})
    assert "Ripara la topologia" in str(err.value) and (ch.data_dir / "model.ply").read_bytes() == before
    report = model.apply(ch, "repair", {})
    assert report["changed"] and report["details"]["duplicate_faces"]["faces"] == -3
    assert report["measures"]["two_manifold"] and report["measures"]["non_manifold_edges"] == 0
    assert "facce doppie tolte 3" in model.report_text(report)


def test_remeshing_to_an_edge_length(tmp_path):
    doc, ch = project(tmp_path, grid_mesh(n=30, hole=False, texture=False), texture=False)
    report = model.apply(ch, "remesh", {"edge_length": 0.04})
    assert report["changed"] and report["name"] == "MeshLab: rimaglia isotropa, lato 0,04 m"
    assert report["measures"]["mean_edge"] == pytest.approx(0.04, rel=0.25)
    new = read_mesh(ch.product_path("model"))
    assert new.colors is not None and new.field("confidence") is not None


def test_decimation_keeps_a_texture_of_two_pages(tmp_path):
    import cv2

    from ppg_mesh_tools.model import texture_pages
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh

    mesh = grid_mesh(hole=False)
    page = (mesh.xyz[mesh.faces].mean(axis=1)[:, 0] > 0.5).astype(np.int32)
    mesh.texcoords[page == 1, 0::2] += 10.0  # the second page: u shifted, to tell the pages apart
    mesh.texnumber = page
    mesh.set_texture_files(["model_texture.jpg", "model_texture_1.jpg"])
    doc, ch = project(tmp_path, mesh)
    cv2.imwrite(str(ch.data_dir / "model_texture_1.jpg"), np.zeros((16, 16, 3), np.uint8))
    write_mesh(ch.product_path("model"), mesh)
    assert len(texture_pages(ch)) == 2
    report = model.apply(ch, "decimate", {"fraction": 0.3})
    assert report["texture_kept"]
    new = read_mesh(ch.product_path("model"))
    assert new.texture_files == ["model_texture.jpg", "model_texture_1.jpg"] and len(texture_pages(ch)) == 2
    assert set(np.unique(new.texnumber)) == {0, 1}
    corners = new.xyz[new.faces][:, :, :2].reshape(len(new.faces), 6).copy()
    corners[new.texnumber == 1, 0::2] += 10.0
    err = np.abs(new.texcoords - corners)
    # every face on its page (a face on the other one would be 10 off), every corner where it was: a
    # vertex that slides along the seam of the pages takes its coordinate along, within a cell (0.026)
    assert err.max() < 0.03 and np.median(err) < 1e-4


def test_the_repair_stays_repaired(tmp_path):
    """Two squares meeting at a corner: the repair splits the vertex they share and the next operations
    keep it split - the remeshing, which joins it again, is repaired after as well, and the measures
    after do not weld the copies back."""
    doc, ch = project(tmp_path, bowtie_mesh(), texture=False)
    assert model.measure(ch)["non_manifold_vertices"] == 1
    report = model.apply(ch, "remesh", {"edge_length": 0.1})
    assert "repaired_after" in report["details"] and report["measures"]["non_manifold_vertices"] == 0
    assert "non-manifold" not in model.report_text(report)
    assert model.measure(ch)["non_manifold_vertices"] == 0  # the copies stay apart
    report = model.apply(ch, "smooth", {"method": "taubin"})
    assert report["measures"]["non_manifold_vertices"] == 0
    # asked not to repair, the remeshing says what it left
    doc2, ch2 = project(tmp_path / "unrepaired", bowtie_mesh(), texture=False)
    report = model.apply(ch2, "remesh", {"edge_length": 0.1, "repair_first": False})
    assert report["measures"]["non_manifold_vertices"] == 1
    assert "; restano 1 vertice e 0 spigoli non-manifold" in model.report_text(report)
