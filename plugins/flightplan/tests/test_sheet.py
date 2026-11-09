"""The flight sheet of the plan, in Excel and HTML: one row per leg, the blocks, the plan, the waypoints, with the
same numbers as the computation; the predicted coverage and the comparison after the flight when given.

Ported from the planner's tests on the neutral place of the tests (45.0 N 10.0 E), maps on the grey background."""
import io

from openpyxl import load_workbook

from ppg_flightplan import legmap, sheet
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import camera


def test_the_sheet_has_one_row_per_leg_with_the_numbers_of_the_computation(plan):
    piano = dict(plan, decollo=[44.99, 9.99])
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(piano, images=False)))
    assert wb.sheetnames == ["Tratte", "Blocchi", "Piano", "Waypoint"]
    ws = wb["Tratte"]
    titles = [c.value for c in ws[3]]
    assert titles == sheet.HEADER_LEGS and len(titles) == 32
    assert titles[:6] == ["N.", "Blocco", "Tipo", "Tratta", "Quota (m)", "Camera (°)"]
    assert "Batterie" in titles and "GSD (cm/px)" in titles and "Decollo (lat)" in titles
    legs = pl.plan_legs(piano)
    rows = [r for r in ws.iter_rows(min_row=4, max_row=3 + len(legs), values_only=True)]
    assert len(rows) == len(legs) >= 2
    col = {t: i for i, t in enumerate(titles)}
    first = rows[0]
    assert first[col["Blocco"]] == "A nadirale" and first[col["Tipo"]] == "griglia"
    assert first[col["Tratta"]].startswith("1 di ")
    assert first[col["Quota (m)"]] == 110 and first[col["Camera (°)"]] == -90
    assert first[col["Sovrapp. avanti (%)"]] == 80
    assert first[col["Sovrapp. laterale (%)"]] == 70 and first[col["Direzione strisciate (°)"]] == 0
    assert first[col["Risoluzione (MP)"]] == 49 and first[col["Scatto"]] == "a intervallo"
    assert first[col["GSD (cm/px)"]] == round(camera("mini4pro").gsd_cm(110, True), 2)
    assert first[col["Decollo (lat)"]] == 44.99 and first[col["File"]].endswith(".kmz")
    # the ring: no side overlap nor direction, but the point of aim
    ring_ = next(r for r in rows if r[col["Tipo"]] == "anello")
    assert ring_[col["Sovrapp. laterale (%)"]] is None and ring_[col["Punto di mira"]] is not None
    assert ring_[col["Risoluzione (MP)"]] == 12
    # the totals agree with the computation
    totals = pl.compute(piano)["totale"]
    total = [c.value for c in ws[4 + len(legs)]]
    assert total[1] == "Totale" and total[col["Scatti"]] == totals["scatti"]
    assert total[col["Batterie"]] == totals["batterie"]
    assert sum(r[col["Scatti"]] for r in rows) >= totals["scatti"] - len(legs)   # the legs split the strips
    # blocks and plan
    b = list(wb["Blocchi"].iter_rows(min_row=2, values_only=True))
    assert len(b) == 2 and b[0][1] == "A nadirale" and b[0][3] == "sì"
    items = {r[0]: r[1] for r in wb["Piano"].iter_rows(min_row=2, values_only=True)}
    assert items["Scatti"] == totals["scatti"] and items["Tratte"] == len(legs)
    assert items["Drone / camera"] == "DJI Mini 4 Pro"
    # the waypoints: all those of the legs
    wp = list(wb["Waypoint"].iter_rows(min_row=2, values_only=True))
    assert len(wp) == sum(len(leg.route.points) for leg in legs) and wp[0][0] == 1 and wp[0][2] == 0


def test_one_sheet_per_leg_with_the_values_and_the_map(plan):
    legs = pl.plan_legs(plan)
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(plan)))
    assert wb.sheetnames[:4] == ["Tratte", "Blocchi", "Piano", "Waypoint"]
    sheets = wb.sheetnames[4:]
    assert len(sheets) == len(legs) and sheets[0] == "T1 A nadirale" and all(len(n) <= 31 for n in sheets)
    ws = wb[sheets[0]]
    assert ws["A1"].value.startswith("Campo di prova · A nadirale · tratta 1 di")
    items = {r[0]: r[1] for r in ws.iter_rows(min_row=5, max_col=2, values_only=True) if r[0]}
    assert items["Quota (m)"] == 110 and items["Tratta"].startswith("1 di") and items["File"].endswith(".kmz")
    assert len(ws._images) == 1 and ws._images[0].anchor._from.col == 3   # the map in D4
    last = wb[sheets[-1]]
    assert last["A1"].value.startswith("Campo di prova · D anello alto")


def test_the_html_sheet_has_the_tables_and_one_section_per_leg_with_the_map(plan):
    legs = pl.plan_legs(plan)
    page = sheet.sheet_html(plan)
    assert page.startswith("<!doctype html>") and "<title>Campo di prova · scheda delle tratte</title>" in page
    assert page.count('<section class="tratta"') == len(legs)
    assert page.count('<img src="data:image/jpeg;base64,') == len(legs)
    assert "A nadirale · tratta 1 di" in page and "D anello alto" in page and "GSD (cm/px)" in page
    totals = pl.compute(plan)["totale"]
    assert f"<b>{totals['scatti']}</b><span>scatti</span>" in page
    without = sheet.sheet_html(plan, images=False)
    assert "<img" not in without and without.count('<section class="tratta"') == len(legs)
    # no HTML from the name of the plan
    odd = dict(plan, nome="<b>x</b>")
    assert "<b>x</b>" not in sheet.sheet_html(odd, images=False)
    assert "Photo3D" not in page and "pyarchinit-photogrammetry" in page


def test_the_sheet_carries_the_dji_fly_mission_of_every_leg(plan):
    legs = pl.plan_legs(plan)
    piano = dict(plan, djiFly={legs[0].key: {"cartella": "550E8400-E29B-41D4-A716-446655440000",
                                             "nomeDji": "08/09/2026 08:01:35"}})
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(piano, images=False)))
    ws = wb["Tratte"]
    titles = [c.value for c in ws[3]]
    col = {t: i for i, t in enumerate(titles)}
    rows = list(ws.iter_rows(min_row=4, max_row=3 + len(legs), values_only=True))
    assert rows[0][col["Missione DJI Fly"]] == "08/09/2026 08:01:35"
    assert rows[0][col["Cartella DJI Fly"]].startswith("550E8400")
    assert rows[1][col["Missione DJI Fly"]] is None
    assert "08/09/2026 08:01:35" in sheet.sheet_html(piano, images=False)
    assert "DJI Fly 08/09/2026 08:01:35" in legmap.caption(piano, legs[0])[0]


# ------------------------------------------------------------------ the plugin's additions

COVERAGE = {"stamp": "abc", "files": {}, "stats": {"cells": 12345, "median": 17.0, "p10": 9.0, "low_fraction": 0.031,
                                                   "gsd_median_cm": 0.81, "gsd_max_cm": 1.4, "occlusion": False}}


def test_the_coverage_sheet(plan):
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(plan, images=False, coverage=COVERAGE)))
    assert wb.sheetnames == ["Tratte", "Blocchi", "Piano", "Waypoint", "Copertura"]
    values = [c for row in wb["Copertura"].iter_rows(values_only=True) for c in row if c is not None]
    assert 17.0 in values and 0.031 in values and 12345 in values
    labels = {r[0]: r[1] for r in wb["Copertura"].iter_rows(min_row=2, values_only=True) if r[0]}
    assert labels["Foto per cella (mediana)"] == 17.0
    # the stats alone work too, and the HTML has the section
    wb2 = load_workbook(io.BytesIO(sheet.sheet_xlsx(plan, images=False, coverage=COVERAGE["stats"])))
    assert "Copertura" in wb2.sheetnames
    page = sheet.sheet_html(plan, images=False, coverage=COVERAGE)
    assert "<h2>Copertura prevista</h2>" in page and "0.031" in page


def test_the_comparison_sheet(plan):
    comparison = {"scatti_previsti": 640, "foto_trovate": 612,
                  "mancanti": [{"tratta": 1, "n": 4}, {"tratta": 2, "n": 9}]}
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(plan, images=False, comparison=comparison)))
    assert wb.sheetnames[-1] == "Confronto"
    values = [c for row in wb["Confronto"].iter_rows(values_only=True) for c in row if c is not None]
    assert 640 in values and 612 in values and 9 in values
    assert "Confronto" in sheet.sheet_html(plan, images=False, comparison=comparison)


def test_the_waypoints_carry_their_own_height(plan):
    piano = dict(plan)
    b = dict(piano["blocchi"][0])
    p, _ = pl.route_of_block(b, piano)
    b["quote"] = [110.0 + k for k in range(len(p.points))]
    piano["blocchi"] = [b]
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(piano, images=False)))
    heights = [r[5] for r in wb["Waypoint"].iter_rows(min_row=2, values_only=True)]
    assert heights[:3] == [110.0, 111.0, 112.0]


def test_the_english_sheet(plan, monkeypatch):
    from ppg_flightplan import texts

    monkeypatch.setattr(texts, "lang", lambda: "en")
    wb = load_workbook(io.BytesIO(sheet.sheet_xlsx(plan, images=False, coverage=COVERAGE)))
    assert wb.sheetnames == ["Legs", "Blocks", "Plan", "Waypoints", "Coverage"]
    assert [c.value for c in wb["Legs"][3]][:5] == ["No.", "Block", "Kind", "Leg", "Height (m)"]
