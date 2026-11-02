"""The stones of a face against the generated pattern: sizes, courses and their heights, the mortar share, the type
the measures suggest; scattered stones; a face too coarse for its stones; the tables and the GeoPackages."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, stones, walls  # noqa: E402

Z0 = S.WORLD[2]


def _face(tmp_path, resolution=0.02):
    doc, ch = S.project(tmp_path, S.wall(lean=False, bulge=False))
    w = walls.new_wall(ch, "Muro")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.STONE_WALL, 0.0],
                             side=S.WORLD[:2] + [3.0, -5.0], resolution=resolution)
    return doc, ch, w, entry


def test_stones_courses_mortar_and_type(tmp_path):
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    polys, truths = S.stone_pattern(Z0)
    S.put_drawings(ch, entry["id"], polys)
    assert walls.face_info(ch, entry["id"])["drawings"]["stones"] == len(polys)
    region = box(0.0, Z0, S.STONE_WALL, Z0 + S.COURSES * S.PITCH)
    r = stones.measure_face(ch, w["id"], entry["id"], region=region)
    rows = r["rows"]
    assert len(rows) == len(polys) == r["stones"]
    for row, (width, height, centre) in zip(rows, truths):
        assert abs(row["width"] - width) < 0.01 and abs(row["height"] - height) < 0.01
        assert abs(row["major"] - max(width, height)) < 0.01 and abs(row["minor"] - min(width, height)) < 0.01
        lying = width >= height  # the last stone of a course is cut short: it stands upright
        assert abs(row["orientation"] if lying else abs(row["orientation"]) - 90.0) < 0.5
        assert row["rectangularity"] > 0.95
    assert r["courses"] == S.COURSES
    for k, c in enumerate(r["course_rows"]):
        assert abs(c["height"] - (Z0 + (k + 0.5) * S.PITCH)) < 0.01 and c["horizontal"]
        assert c["stones"] == sum(1 for t in truths if abs(t[2] - c["height"]) < 0.01)
    truth_mortar = 1.0 - sum(p.area for p in polys) / region.area
    assert abs(r["mortar"] - truth_mortar) < 0.03 and 0.10 < truth_mortar < 0.14
    typ = r["typology"]
    assert typ["coursed"] and typ["regular"] and typ["squared"]
    assert typ["line"].startswith("Le misure suggeriscono una muratura a filari, regolare, a conci squadrati (")
    text = stones.stones_text(walls.results(ch, w["id"], entry["id"])["stones"])
    assert "5 filari" in text and "filare 1: quota" in text and "Le misure suggeriscono" in text
    # the tables
    folder = walls.face_folder(ch, w["id"], entry["id"])
    assert (folder / "pietre.csv").read_text().splitlines()[0].startswith("n,filare,u_m,v_m,larghezza_m")
    assert len((folder / "filari.csv").read_text().splitlines()) == S.COURSES + 1
    from openpyxl import load_workbook

    wb = load_workbook(folder / "pietre.xlsx")
    assert wb.sheetnames == ["Pietre", "Filari", "Sintesi"] and wb["Pietre"].max_row == len(polys) + 1
    con = sqlite3.connect(folder / "pietre_misure.gpkg")
    assert con.execute("SELECT COUNT(*) FROM pietre_misure").fetchone()[0] == len(polys)
    con.close()
    # the region kept: a second run without it measures on the same area
    again = stones.measure_face(ch, w["id"], entry["id"])
    assert again["region_kind"] == "drawn" and abs(again["mortar"] - r["mortar"]) < 1e-9


def test_scattered_stones_are_not_coursed(tmp_path):
    doc, ch, w, entry = _face(tmp_path, resolution=0.01)
    S.put_drawings(ch, entry["id"], S.irregular_stones(Z0))
    r = stones.measure_face(ch, w["id"], entry["id"])
    typ = r["typology"]
    assert not typ["coursed"] and not typ["squared"] and r["region_kind"] == "bounds"
    assert "senza filari evidenti" in typ["line"] and "non squadrate" in typ["line"]


def test_a_face_too_coarse_for_its_stones_is_refused(tmp_path):
    doc, ch, w, entry = _face(tmp_path, resolution=0.1)
    with pytest.raises(walls.NotReady, match="Pietre"):
        stones.measure_face(ch, w["id"], entry["id"])
    S.put_drawings(ch, entry["id"], S.stone_pattern(Z0)[0])
    with pytest.raises(walls.NotReady, match="100,0 mm per pixel"):
        stones.measure_face(ch, w["id"], entry["id"])


def test_the_export_of_the_face_carries_the_measures(tmp_path):
    doc, ch, w, entry = _face(tmp_path)
    polys, _truths = S.stone_pattern(Z0)
    S.put_drawings(ch, entry["id"], polys)
    stones.measure_face(ch, w["id"], entry["id"])
    out = tmp_path / "export"
    paths = stones.export_face(ch, w["id"], entry["id"], out, stem="faccia")
    for key, z in (("gpkg", 0), ("gpkg_world", 1)):
        con = sqlite3.connect(paths[key])
        tables = {r[0] for r in con.execute("SELECT table_name FROM gpkg_contents")}
        assert {"pietre", "pietre_misure"} <= tables
        assert con.execute("SELECT COUNT(*) FROM pietre_misure").fetchone()[0] == len(polys)
        assert con.execute("SELECT z FROM gpkg_geometry_columns WHERE table_name = 'pietre_misure'").fetchone()[0] == z
        cols = {r[1] for r in con.execute("PRAGMA table_info(pietre_misure)")}
        assert {"larghezza_m", "altezza_m", "allungamento", "filare"} <= cols
        con.close()
    assert (out / "faccia_pietre.xlsx").exists() and (out / "faccia_disegni.dxf").exists()
    assert np.isfinite(walls.results(ch, w["id"], entry["id"])["stones"]["mortar"])
