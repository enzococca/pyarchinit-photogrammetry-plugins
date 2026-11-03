"""The comparison of two campaigns of a face: per term the area gained and lost, the cracks new, longer or wider, the
treated areas that show decay again; the change of the orthophoto in Lab and, with a second survey, the change of the
relief; the check that the two surveys are in the same system (markers, else phase correlation of the orthophotos)
and nothing written when they are not."""
from __future__ import annotations

import numpy as np
import pytest

import conservation_samples as S
from ppg_conservation import compare, records
from ppg_conservation.records import NotReady

Z0 = S.Z0


def _two_campaigns(ch, entry):
    """Campaign A and campaign B on the same face: in B the black crust has grown by 0.2 m2 and a new crack."""
    a = records.new_campaign(ch, entry["id"], date="2026-05-14", name="A")
    b = records.new_campaign(ch, entry["id"], date="2027-05-14", name="B")
    u0, v0, u1, v1 = S.BLACK_CRUST
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    records.add_feature(ch, b["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box((u0, v0, u1 + 0.25, v1)))
    records.add_feature(ch, a["id"], "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    records.add_feature(ch, b["id"], "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    new = records.add_feature(ch, b["id"], "crack", "ISCS.CRACK", f"LINESTRING (3.5 {Z0 + 0.3}, 3.9 {Z0 + 1.1})",
                              width_mm=1.0)
    clean = records.add_feature(ch, a["id"], "treatment", "INT.CLEANING", S.box(S.EFFLORESCENCE),
                                treatment={"technique": "INT.CLEANING", "done": True})
    eu0, ev0, eu1, ev1 = S.EFFLORESCENCE
    records.add_feature(ch, b["id"], "decay", "ISCS.EFFLORESCENCE", S.box((eu0, ev0, eu0 + 0.5, ev1)))
    return a, b, new, clean


def test_two_campaigns_of_one_face(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    a, b, new, clean = _two_campaigns(ch, entry)
    r = compare.compare(ch, a["id"], b["id"])
    assert r["alignment"]["ok"] and r["alignment"]["method"] == "same"
    crust = next(row for row in r["terms"] if row["code"] == "ISCS.CRUST.BLACK_CRUST")
    assert crust["gained"] == pytest.approx(0.2, rel=0.05)
    assert crust["lost"] == pytest.approx(0.0, abs=1e-6)
    assert crust["area_a"] == pytest.approx(0.8, rel=0.01) and crust["area_b"] == pytest.approx(1.0, rel=0.01)
    states = {row["id_b"]: row["state"] for row in r["cracks"]}
    assert states[new["id"]] == "new"
    assert list(states.values()).count("same") == 1
    treated = next(row for row in r["treatments"] if row["id"] == clean["id"])
    assert treated["decay_area"] == pytest.approx(0.2, rel=0.05)
    folder = ch.data_dir / "conservation" / entry["id"] / f"compare_{a['id']}_{b['id']}"
    assert r["folder"] == folder
    for key in ("change", "xlsx", "json"):
        assert r[key].exists() and r[key].parent == folder
    assert r.get("pdf") or r.get("png")
    assert r["relief_change"] is None  # one survey: no change of the relief
    import rasterio
    from openpyxl import load_workbook

    with rasterio.open(r["change"]) as src, rasterio.open(ch.data_dir / entry["orthophoto"]) as ortho:
        assert src.transform == ortho.transform and src.shape == ortho.shape and src.dtypes[0] == "float32"
        dE = src.read(1)
    assert np.nanmax(np.abs(dE)) < 1e-3  # the same orthophoto: no change of colour
    assert np.isnan(dE[0, -1])  # far from every feature (more than 0.5 m): masked
    sheets = load_workbook(r["xlsx"]).sheetnames
    assert sheets == ["Confronto per termine", "Lesioni", "Interventi"]
    # the comparison is files, never records
    assert set(ch.meta["conservation"]) == {"campaigns", "features", "assessments", "legend", "local_terms"}


def test_second_survey_on_the_same_grid_and_its_relief(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc, offset=0.02)
    built = compare.ensure_elevation(other, entry)
    import rasterio

    with rasterio.open(ch.data_dir / entry["orthophoto"]) as a, \
            rasterio.open(other.data_dir / built["orthophoto"]) as b:
        assert a.transform == b.transform and a.shape == b.shape
    assert compare.ensure_elevation(other, entry)["id"] == built["id"]  # found, not built again
    assert len(other.meta["elevations"]) == 1
    S.copy_orthophoto(ch, entry, other, built)
    a = records.new_campaign(ch, entry["id"], date="2026-05-14")
    b = records.new_campaign(other, built["id"], date="2027-05-14")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    records.add_feature(other, b["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    r = compare.compare(ch, a["id"], b["id"], other=other)
    assert r["alignment"]["ok"] and r["alignment"]["method"] == "phase"
    assert r["alignment"]["shift_px"] < 0.5
    with rasterio.open(r["relief_change"]) as src:
        d = src.read(1)
    assert np.nanmedian(d) == pytest.approx(2.0, abs=0.5)  # cm, B minus A, towards the viewer
    assert r["relief"]["median_cm"] == pytest.approx(2.0, abs=0.5)


def test_a_survey_not_aligned_is_refused_and_nothing_is_written(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc, shift_u=0.05)
    built = compare.ensure_elevation(other, entry)
    S.copy_orthophoto(ch, entry, other, built, shift_u=0.05)
    check = compare.check_alignment(ch, entry, other)
    assert not check["ok"] and check["method"] == "phase"
    assert check["shift_px"] == pytest.approx(5.0, abs=0.5)
    assert check["shift_m"] == pytest.approx(0.05, abs=0.005)
    assert check["shift_uv"][0] == pytest.approx(0.05, abs=0.005) and abs(check["shift_uv"][1]) < 0.005
    assert "Allinea chunk" in check["message"]
    a = records.new_campaign(ch, entry["id"], date="2026-05-14")
    b = records.new_campaign(other, built["id"], date="2027-05-14")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    with pytest.raises(NotReady, match="Allinea chunk"):
        compare.compare(ch, a["id"], b["id"], other=other)
    assert not (ch.data_dir / "conservation" / entry["id"] / f"compare_{a['id']}_{b['id']}").exists()
    assert not (ch.data_dir / "conservation").exists()


def test_markers_on_both_faces_decide(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc)
    built = compare.ensure_elevation(other, entry)  # its orthophoto is empty: the markers decide
    for label, u, v in (("T1", 1.0, Z0 + 0.5), ("T2", 5.0, Z0 + 2.5)):
        S.put_marker(ch, label, u, v)
        S.put_marker(other, label, u, v, du=0.004)
    check = compare.check_alignment(ch, entry, other)
    assert check["ok"] and check["method"] == "markers" and check["markers"] == 2
    assert check["shift_m"] == pytest.approx(0.004, abs=1e-4)
    far = S.second_chunk(doc, label="Muro 2028")
    compare.ensure_elevation(far, entry)
    for label, u, v in (("T1", 1.0, Z0 + 0.5), ("T2", 5.0, Z0 + 2.5)):
        S.put_marker(far, label, u, v, du=0.05)
    check = compare.check_alignment(ch, entry, far)
    assert not check["ok"] and check["method"] == "markers"
    assert check["shift_m"] == pytest.approx(0.05, abs=1e-3)
    assert "Allinea chunk" in check["message"]
    assert built["id"] != entry["id"] or other is not ch


def _surveys(tmp_path):
    """A face with campaign A, a later survey 2 cm in front (aligned) with campaign B, and a survey 5 cm along the
    wall (not aligned) with campaign C: the project saved."""
    doc, ch, entry = S.face(tmp_path)
    later = S.second_chunk(doc, label="Rilievo 2027", offset=0.02)
    moved = S.second_chunk(doc, label="Rilievo spostato", shift_u=0.05)
    doc.chunk = ch
    a = records.new_campaign(ch, entry["id"], date="2026-05-14", name="Prima")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    built = {}
    for other, shift, name in ((later, 0.0, "Seconda"), (moved, 0.05, "Spostata")):
        e = compare.ensure_elevation(other, entry)
        S.copy_orthophoto(ch, entry, other, e, shift_u=shift)
        c = records.new_campaign(other, e["id"], date="2027-05-14", name=name)
        u0, v0, u1, v1 = S.BLACK_CRUST
        records.add_feature(other, c["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box((u0, v0, u1 + 0.25, v1)))
        built[name] = (other, e, c)
    doc.save()
    return doc, ch, entry, a, built


def test_the_comparison_in_the_pane(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from test_plugin import _window

    from ppg_conservation import ConservationPlugin
    from pyarchinit_photogrammetry.plugins import Context

    import locale

    numbers = locale.setlocale(locale.LC_NUMERIC)  # Qt sets the system's: the tests after this one want theirs
    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    later = S.second_chunk(doc, label="Rilievo 2027", offset=0.02)
    moved = S.second_chunk(doc, label="Rilievo spostato", shift_u=0.05)
    doc.chunk = ch
    a = records.new_campaign(ch, entry["id"], date="2026-05-14", name="Prima")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said, asked, answers = [], [], {}
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append(({f.name: f for f in fields}, intro))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin.refresh_pane()
        plugin.pane.select_campaign(a["id"])
        assert not plugin.pane.overlay.isEnabled()
        # no B yet: the later survey's elevation is built on the same plane
        answers["other"] = f"b:{later.key}:"
        plugin._requested("compare")
        w.runner.wait()
        fields, intro = asked[-1]
        assert set(dict(fields["other"].choices)) == {f"b:{later.key}:", f"b:{moved.key}:"}
        assert "Allinea chunk" in intro
        assert "costruito" in said[-1]
        built = compare.find_elevation(later, entry)
        assert built is not None
        S.copy_orthophoto(ch, entry, later, built)
        b = records.new_campaign(later, built["id"], date="2027-05-14", name="Seconda")
        u0, v0, u1, v1 = S.BLACK_CRUST
        records.add_feature(later, b["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box((u0, v0, u1 + 0.25, v1)))
        answers["other"] = f"c:{later.key}:{b['id']}"
        plugin._requested("compare")
        w.runner.wait()
        assert said[-1].startswith("Confronto:") and "confronto.xlsx" in said[-1], said[-1]
        assert plugin.overlay is not None and plugin.overlay.shown() and plugin.overlay.key.endswith(":colour")
        assert plugin._shown_eid() == entry["id"]
        assert plugin.pane.overlay_kind() == "colour" and plugin.pane.overlay.isEnabled()
        plugin.pane.opacity_slider.setValue(30)
        assert plugin.overlay.opacity() == pytest.approx(0.3)
        keys = [plugin.pane.overlay.itemData(i) for i in range(plugin.pane.overlay.count())]
        plugin.pane.overlay.setCurrentIndex(keys.index("relief"))
        assert plugin.overlay.shown() and plugin.overlay.key.endswith(":relief")
        assert plugin.overlay.limit >= 2.0  # cm: the later survey 2 cm in front
        w.show_elevation(entry)  # the raster again (the scene cleared): the overlay shown again on its face
        assert plugin.overlay.shown() and plugin.overlay.key.endswith(":relief")
        plugin.pane.overlay.setCurrentIndex(0)
        assert not plugin.overlay.shown()
        # a survey not aligned: refused with the shift, nothing written
        e = compare.ensure_elevation(moved, entry)
        S.copy_orthophoto(ch, entry, moved, e, shift_u=0.05)
        m = records.new_campaign(moved, e["id"], date="2027-05-14", name="Spostata")
        answers["other"] = f"c:{moved.key}:{m['id']}"
        plugin._requested("compare")
        w.runner.wait()
        assert "Allinea chunk" in said[-1] and "5,0 px" in said[-1], said[-1]
        assert not (ch.data_dir / "conservation" / entry["id"] / f"compare_{a['id']}_{m['id']}").exists()
        assert not errors, errors
        plugin.deactivate()
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()
        locale.setlocale(locale.LC_NUMERIC, numbers)


def test_the_comparison_from_the_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app
    from test_outputs import _plugin_folder

    doc, ch, entry, a, built = _surveys(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()

    def run(*args):
        return cli.invoke(app, ["plugin", "conservation", "compare", str(doc.path), *args])

    r = run("--campaign", "Prima", "--with", "Seconda", "--chunk-b", "Rilievo 2027")
    assert r.exit_code == 0, r.output
    assert "guadagnata 0,200" in r.output and "confronto.xlsx" in r.output and "relief_change.tif" in r.output
    r = run("--campaign", "Prima", "--with", "Spostata", "--chunk-b", "Rilievo spostato")
    assert r.exit_code == 1 and "Allinea chunk" in r.output, r.output
    r = run("--campaign", "Prima", "--with", "Nessuna")
    assert r.exit_code == 1 and "Nessuna campagna" in r.output, r.output
    r = run("--campaign", "Prima", "--with", "Seconda", "--chunk-b", "Nessuno")
    assert r.exit_code == 1 and "Nessun chunk" in r.output, r.output
