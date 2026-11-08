"""The sheet of the flight plan, in Excel and in HTML.

One row per leg with every quantity (height, camera, speed, overlaps, direction, shooting, waypoints, shots,
minutes, batteries, GSD, takeoff), the totals per block and per plan, all the waypoints (each with its own height
above the takeoff), and for each leg its map with only the route it covers. The same numbers as the planner, to
read, print and send to whoever flies. With the predicted coverage (``coverage``) and the comparison after the flight
(``comparison``) the sheet gets their pages too.

Ported from the author's Photo3D planner; the texts in two languages, the waypoints' own heights and the coverage
and comparison pages are the plugin's.
"""
from __future__ import annotations

import base64
import datetime
import html as html_
import io
import re
from typing import Optional

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import legmap
from . import plan as pl
from .camera import Camera
from .texts import t, warning_text

PROGRAM = "pyarchinit-photogrammetry"

KINDS = {"griglia": ("griglia", "grid"), "anello": ("anello", "ring"), "percorso": ("percorso", "route")}

HEADER_LEGS_PAIRS = [
    ("N.", "No."), ("Blocco", "Block"), ("Tipo", "Kind"), ("Tratta", "Leg"), ("Quota (m)", "Height (m)"),
    ("Camera (°)", "Camera (°)"), ("Velocità richiesta (m/s)", "Speed asked (m/s)"),
    ("Velocità effettiva (m/s)", "Actual speed (m/s)"), ("Sovrapp. avanti (%)", "Front overlap (%)"),
    ("Sovrapp. laterale (%)", "Side overlap (%)"), ("Direzione strisciate (°)", "Strip direction (°)"),
    ("Risoluzione (MP)", "Resolution (MP)"), ("Scatto", "Shooting"), ("Intervallo (s)", "Interval (s)"),
    ("Passo fra gli scatti (m)", "Step between shots (m)"), ("Waypoint", "Waypoints"), ("Scatti", "Shots"),
    ("Lunghezza (km)", "Length (km)"), ("Minuti di volo", "Flight minutes"), ("Batterie", "Batteries"),
    ("GSD (cm/px)", "GSD (cm/px)"), ("Impronta a terra (m)", "Ground footprint (m)"),
    ("Punto di mira", "Point of aim"), ("Decollo (lat)", "Takeoff (lat)"), ("Decollo (lon)", "Takeoff (lon)"),
    ("Primo punto (lat)", "First point (lat)"), ("Primo punto (lon)", "First point (lon)"),
    ("Ultimo punto (lat)", "Last point (lat)"), ("Ultimo punto (lon)", "Last point (lon)"), ("File", "File"),
    ("Missione DJI Fly", "DJI Fly mission"), ("Cartella DJI Fly", "DJI Fly folder"),
]
# The 32 columns of the legs in Italian (the stored and tested names); :func:`header_legs` in the interface language.
HEADER_LEGS = [it for it, _ in HEADER_LEGS_PAIRS]

HEADER_BLOCKS_PAIRS = [
    ("N.", "No."), ("Blocco", "Block"), ("Tipo", "Kind"), ("Acceso", "Active"), ("Quota (m)", "Height (m)"),
    ("Camera (°)", "Camera (°)"), ("Velocità effettiva (m/s)", "Actual speed (m/s)"),
    ("Sovrapp. avanti (%)", "Front overlap (%)"), ("Sovrapp. laterale (%)", "Side overlap (%)"),
    ("Direzione strisciate (°)", "Strip direction (°)"), ("Risoluzione (MP)", "Resolution (MP)"),
    ("Scatto", "Shooting"), ("Intervallo (s)", "Interval (s)"), ("Passo (m)", "Step (m)"), ("Svolte", "Turns"),
    ("Waypoint", "Waypoints"), ("Scatti", "Shots"), ("Lunghezza (km)", "Length (km)"), ("Minuti", "Minutes"),
    ("Batterie", "Batteries"), ("Tratte", "Legs"), ("GSD (cm/px)", "GSD (cm/px)"), ("Avvisi", "Warnings"),
]

COVERAGE_LABELS = {
    "cells": ("Celle nell'area", "Cells in the area"),
    "median": ("Foto per cella (mediana)", "Photos per cell (median)"),
    "p10": ("Foto per cella (10° percentile)", "Photos per cell (10th percentile)"),
    "low_fraction": ("Celle con poche foto (frazione)", "Cells with few photos (fraction)"),
    "gsd_median_cm": ("GSD mediano (cm/px)", "Median GSD (cm/px)"),
    "gsd_max_cm": ("GSD massimo (cm/px)", "Largest GSD (cm/px)"),
    "occlusion": ("Ostacoli considerati", "Occlusion considered"),
}


def header_legs() -> list[str]:
    return [t(it, en) for it, en in HEADER_LEGS_PAIRS]


def _header_blocks() -> list[str]:
    return [t(it, en) for it, en in HEADER_BLOCKS_PAIRS]


def _kind(b: dict) -> str:
    k = b.get("tipo", "percorso")
    return t(*KINDS[k]) if k in KINDS else k


def _yes(v: bool) -> str:
    return t("sì", "yes") if v else t("no", "no")


def _shooting(mode: str) -> str:
    return t("a intervallo", "by interval") if mode == "intervallo" else t("uno per punto", "one per point")


def dji_mission(piano: dict, leg: pl.PlanLeg) -> tuple[Optional[str], Optional[str]]:
    """The name (the creation time, as DJI Fly shows it) and the folder of the placeholder mission given to the
    leg when copying into DJI Fly, if the plan remembers it."""
    entry = (piano.get("djiFly") or {}).get(leg.key) or {}
    return entry.get("nomeDji") or None, entry.get("cartella") or None


def _megapixels(cam: Camera, high: bool) -> int:
    wide = cam.pixels_for(high)
    tall = wide * cam.height_mm / cam.width_mm
    return round(wide * tall / 1e6)


def _header(ws, row: int, titles: list[str]) -> None:
    fill = PatternFill("solid", fgColor="DDEBF7")
    for k, title in enumerate(titles, 1):
        c = ws.cell(row=row, column=k, value=title)
        c.font = Font(bold=True)
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="center")


def _widths(ws, widths: list[int]) -> None:
    for k, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(k)].width = w


def leg_row(piano: dict, n: int, leg: pl.PlanLeg) -> list:
    """The values of the leg, in the order of HEADER_LEGS."""
    cam = pl.plan_camera(piano)
    battery_minutes = float(piano.get("minutiBatteria", 18))
    b = piano["blocchi"][leg.block_index]
    p = leg.route
    high = bool(b.get("alta", piano.get("alta", False)))
    grid = b.get("tipo", "griglia") == "griglia"
    first, last = p.points[0], p.points[-1]
    return [
        n, b.get("nome", ""), _kind(b), t("{0} di {1}", "{0} of {1}").format(leg.number, leg.total),
        round(p.height, 1), round(p.pitch, 1), round(p.speed, 2), round(p.effective_speed, 2),
        b.get("avanti"), b.get("lato") if grid else None, b.get("orientamento") if grid else None,
        _megapixels(cam, high), _shooting(p.mode),
        p.interval, round(p.effective_step, 1), len(p.points), p.shots(), round(p.length / 1000, 2),
        round(p.minutes(), 1), round(p.minutes() / max(battery_minutes, 1), 2),
        round(cam.gsd_cm(p.height, high), 2), f"{cam.footprint(p.height):.0f} × {cam.footprint_short(p.height):.0f}",
        f"{p.poi[0]:.6f}, {p.poi[1]:.6f}" if p.poi else None,
        round(leg.takeoff[0], 6) if leg.takeoff else None, round(leg.takeoff[1], 6) if leg.takeoff else None,
        round(first[0], 6), round(first[1], 6), round(last[0], 6), round(last[1], 6), f"{leg.file_name}.kmz",
        *dji_mission(piano, leg),
    ]


def _block_row(i: int, b: dict, r: dict, piano: dict, cam: Camera) -> list:
    grid = b.get("tipo", "griglia") == "griglia"
    high = bool(b.get("alta", piano.get("alta", False)))
    p, _ = pl.route_of_block(b, piano)
    return [i, b.get("nome", ""), _kind(b), _yes(r["attivo"]), r["quota"], r["inclinazione"], r["velocitaEffettiva"],
            b.get("avanti"), b.get("lato") if grid else None, b.get("orientamento") if grid else None,
            _megapixels(cam, high), _shooting(r["modo"]), r["intervallo"], r["passo"], p.turns,
            r["waypoint"], r["scatti"], round(r["lunghezzaM"] / 1000, 2), r["minuti"], r["batterie"], len(r["tratte"]),
            r["gsdCm"], "; ".join(warning_text(w) for w in (r.get("avvisi") or []))]


def _sheet_name(n: int, block: dict) -> str:
    name = re.sub(r"[\[\]:*?/\\]", " ", f"T{n} {block.get('nome', '')}").strip()
    return name[:31]


def _plan_items(piano: dict, cam: Camera, totals: dict, when: str) -> list[tuple[str, object]]:
    high = bool(piano.get("alta", False))
    takeoff = pl.plan_takeoff(piano)
    return [
        (t("Nome", "Name"), piano.get("nome", "piano")), (t("Esportato il", "Exported on"), when),
        (t("Drone / camera", "Drone / camera"), cam.name),
        (t("Sensore (mm)", "Sensor (mm)"),
         t("{0} × {1}, focale {2}", "{0} × {1}, focal length {2}").format(cam.width_mm, cam.height_mm, cam.focal_mm)),
        (t("Risoluzione di default", "Default resolution"), f"{_megapixels(cam, high)} MP"),
        (t("Minuti per batteria", "Minutes per battery"), float(piano.get("minutiBatteria", 18))),
        (t("Blocchi accesi", "Active blocks"), totals["blocchi"]), (t("Tratte", "Legs"), totals["tratte"]),
        (t("Scatti", "Shots"), totals["scatti"]), (t("Minuti di volo", "Flight minutes"), totals["minuti"]),
        (t("Batterie", "Batteries"), totals["batterie"]), (t("Chilometri", "Kilometres"), totals["km"]),
        (t("Decollo generale", "General takeoff"), ", ".join(f"{x:.6f}" for x in takeoff) if takeoff else "—"),
        (t("Decolli per tratta", "Takeoffs per leg"), len(piano.get("decolli") or {})),
        (t("Quote", "Heights"), t("sopra il punto di decollo di ogni tratta", "above the takeoff point of each leg")),
    ]


def _coverage_items(coverage: dict) -> list[tuple[str, object]]:
    stats = coverage.get("stats", coverage) if isinstance(coverage, dict) else {}
    out = []
    for key, value in stats.items():
        label = t(*COVERAGE_LABELS[key]) if key in COVERAGE_LABELS else key
        if isinstance(value, bool):
            value = _yes(value)
        out.append((label, value))
    return out


def _comparison_parts(comparison: dict) -> tuple[list[tuple[str, object]], list[tuple[str, list[dict]]]]:
    """The comparison after the flight as single values and as tables (lists of rows)."""
    single, tables = [], []
    for key, value in comparison.items():
        if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
            tables.append((key, value))
        elif isinstance(value, (list, tuple, dict)):
            single.append((key, str(value)))
        else:
            single.append((key, value))
    return single, tables


def _footer_maps(background) -> str:
    return getattr(background, "attribution", "") or t("sfondo grigio", "grey background")


def sheet_xlsx(piano: dict, now: Optional[datetime.datetime] = None, images: bool = True, background=None,
               coverage: Optional[dict] = None, comparison: Optional[dict] = None) -> bytes:
    """The Excel file: "Tratte", "Blocchi", "Piano", "Waypoint" and, with ``images``, one sheet per leg with its
    values and the map of only the route it covers (on ``background``, else grey); "Copertura" with the predicted
    coverage (its ``stats``) and "Confronto" with the comparison after the flight, when given."""
    cam = pl.plan_camera(piano)
    name = piano.get("nome", "piano")
    totals_all = pl.compute(piano)
    legs = pl.plan_legs(piano)
    blocks = piano.get("blocchi", [])
    when = (now or datetime.datetime.now()).strftime("%d/%m/%Y %H:%M")

    wb = Workbook()

    # ---- one row per leg
    ws = wb.active
    ws.title = t("Tratte", "Legs")
    ws.cell(row=1, column=1, value=t("{0} · {1} · tratte del piano ({2})", "{0} · {1} · legs of the plan ({2})").format(
        PROGRAM, name, when)).font = Font(bold=True, size=13)
    _header(ws, 3, header_legs())
    for n, leg in enumerate(legs, 1):
        ws.append(leg_row(piano, n, leg))
    last = 3 + len(legs)
    tot = totals_all["totale"]
    ws.cell(row=last + 1, column=2, value=t("Totale", "Total")).font = Font(bold=True)
    for col, value in ((16, sum(len(leg.route.points) for leg in legs)), (17, tot["scatti"]), (18, tot["km"]),
                       (19, tot["minuti"]), (20, tot["batterie"])):
        ws.cell(row=last + 1, column=col, value=value).font = Font(bold=True)
    ws.freeze_panes = "C4"
    _widths(ws, [5, 22, 10, 9, 9, 9, 11, 11, 10, 10, 11, 10, 13, 10, 11, 9, 8, 10, 10, 9, 9, 14, 22, 12, 12, 12, 12,
                 12, 12, 36, 20, 38])
    for row in ws.iter_rows(min_row=4, max_row=last):
        for c in row:
            c.alignment = Alignment(vertical="top")

    # ---- the blocks
    wb2 = wb.create_sheet(t("Blocchi", "Blocks"))
    _header(wb2, 1, _header_blocks())
    for i, (b, r) in enumerate(zip(blocks, totals_all["blocchi"]), 1):
        wb2.append(_block_row(i, b, r, piano, cam))
    wb2.freeze_panes = "C2"
    _widths(wb2, [5, 22, 10, 8, 9, 9, 11, 10, 10, 11, 10, 13, 10, 9, 8, 9, 8, 10, 8, 9, 8, 9, 40])

    # ---- the plan
    wb3 = wb.create_sheet(t("Piano", "Plan"))
    _header(wb3, 1, [t("Voce", "Item"), t("Valore", "Value")])
    for item, value in _plan_items(piano, cam, tot, when):
        wb3.append([item, value])
    _widths(wb3, [26, 60])

    # ---- the waypoints, each with its own height above the takeoff of its leg
    wb4 = wb.create_sheet(t("Waypoint", "Waypoints"))
    _header(wb4, 1, [t("Tratta", "Leg"), t("Blocco", "Block"), t("Indice", "Index"), t("Latitudine", "Latitude"),
                     t("Longitudine", "Longitude"), t("Quota (m)", "Height (m)")])
    for n, leg in enumerate(legs, 1):
        b = blocks[leg.block_index]
        for k, (lat, lon) in enumerate(leg.route.points):
            wb4.append([n, b.get("nome", ""), k, round(lat, 7), round(lon, 7), round(leg.route.height_at(k), 1)])
    wb4.freeze_panes = "A2"
    _widths(wb4, [8, 22, 8, 13, 13, 10])

    # ---- one sheet per leg: its values and the map with only the route it covers
    if images:
        for n, leg in enumerate(legs, 1):
            b = blocks[leg.block_index]
            wsx = wb.create_sheet(_sheet_name(n, b))
            line1, line2 = legmap.caption(piano, leg)
            wsx.cell(row=1, column=1, value=line1).font = Font(bold=True, size=13)
            wsx.cell(row=2, column=1, value=line2).font = Font(italic=True, color="555555")
            _header(wsx, 4, [t("Voce", "Item"), t("Valore", "Value")])
            for item, value in zip(header_legs(), leg_row(piano, n, leg)):
                wsx.append([item, value])
            _widths(wsx, [26, 22, 2])
            image = ExcelImage(io.BytesIO(legmap.leg_image(piano, n - 1, background=background)))
            image.width, image.height = 840, 600
            wsx.add_image(image, "D4")

    # ---- the predicted coverage and the comparison after the flight
    if coverage:
        wc = wb.create_sheet(t("Copertura", "Coverage"))
        _header(wc, 1, [t("Voce", "Item"), t("Valore", "Value")])
        for item, value in _coverage_items(coverage):
            wc.append([item, value])
        _widths(wc, [36, 20])
    if comparison:
        wk = wb.create_sheet(t("Confronto", "Comparison"))
        single, tables = _comparison_parts(comparison)
        _header(wk, 1, [t("Voce", "Item"), t("Valore", "Value")])
        for item, value in single:
            wk.append([item, value])
        for title, rows in tables:
            wk.append([])
            wk.append([title])
            wk.cell(row=wk.max_row, column=1).font = Font(bold=True)
            columns = list(dict.fromkeys(k for r in rows for k in r))
            _header(wk, wk.max_row + 1, columns)
            for r in rows:
                wk.append([r.get(c) if not isinstance(r.get(c), (list, dict)) else str(r.get(c)) for c in columns])
        _widths(wk, [30, 20, 14, 14, 14])

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------- the sheet in HTML

HTML_STYLE = """
body { font: 15px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c1c1c; margin: 0;
  background: #f4f5f7; }
main { max-width: 1160px; margin: 0 auto; padding: 24px 20px 60px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 20px; margin: 36px 0 10px; }
h3 { font-size: 17px; margin: 0 0 6px; }
.sotto { color: #555; margin: 0 0 18px; }
.riassunto { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 6px; }
.riassunto div { background: #fff; border: 1px solid #dde1e6; border-radius: 8px; padding: 10px 16px;
  min-width: 110px; }
.riassunto b { display: block; font-size: 22px; } .riassunto span { color: #666; font-size: 13px; }
.scorri { overflow-x: auto; background: #fff; border: 1px solid #dde1e6; border-radius: 8px; }
table { border-collapse: collapse; font-size: 13px; white-space: nowrap; }
th, td { padding: 6px 9px; border-bottom: 1px solid #e6e8eb; text-align: left; vertical-align: top; }
th { background: #ddebf7; position: sticky; top: 0; } tr.totale td { font-weight: 600; background: #f7f9fb; }
.tratta { background: #fff; border: 1px solid #dde1e6; border-radius: 10px; padding: 16px 18px; margin: 18px 0; }
.tratta .numeri { color: #444; font-size: 14px; margin: 0 0 12px; }
.tratta img { width: 100%; height: auto; border-radius: 6px; display: block; }
.tratta table { margin-top: 14px; white-space: normal; width: 100%; }
.tratta td:first-child { color: #555; width: 34%; }
.piede { color: #777; font-size: 12px; margin-top: 30px; }
@media print { body { background: #fff; } .tratta { page-break-before: always; border: 0; padding: 0; }
  .scorri { border: 0; } th { position: static; } }
"""


def sheet_html(piano: dict, now: Optional[datetime.datetime] = None, images: bool = True, background=None,
               coverage: Optional[dict] = None, comparison: Optional[dict] = None) -> str:
    """The sheet in one HTML page, with the maps of the legs inside (base64): it opens anywhere, it prints, it is
    sent to whoever flies. The coverage and the comparison have their sections when given."""
    e = html_.escape
    cam = pl.plan_camera(piano)
    high = bool(piano.get("alta", False))
    name = piano.get("nome", "piano")
    computed = pl.compute(piano)
    tot = computed["totale"]
    legs = pl.plan_legs(piano)
    blocks = piano.get("blocchi", [])
    when = (now or datetime.datetime.now()).strftime("%d/%m/%Y %H:%M")

    def cell(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:g}"
        return e(str(v))

    def table(head: list[str], rows: list[list]) -> str:
        return ("<div class=\"scorri\"><table><thead><tr>" + "".join(f"<th>{e(h)}</th>" for h in head)
                + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>"
                                                    for r in rows) + "</tbody></table></div>")

    lang = t("it", "en")
    parts = [f"<!doctype html><html lang=\"{lang}\"><head><meta charset=\"utf-8\">"
             f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
             f"<title>{e(name)} · {t('scheda delle tratte', 'sheet of the legs')}</title>"
             f"<style>{HTML_STYLE}</style></head><body><main>",
             f"<h1>{e(name)}</h1><p class=\"sotto\">{t('Scheda delle tratte', 'Sheet of the legs')} · {PROGRAM} · "
             f"{when} · {e(cam.name)} · {_megapixels(cam, high)} MP · "
             f"{float(piano.get('minutiBatteria', 18)):g} {t('minuti per batteria', 'minutes per battery')}</p>",
             "<div class=\"riassunto\">"
             + "".join(f"<div><b>{cell(v)}</b><span>{k}</span></div>" for k, v in (
                 (t("blocchi accesi", "active blocks"), tot["blocchi"]), (t("tratte", "legs"), tot["tratte"]),
                 (t("scatti", "shots"), tot["scatti"]), (t("minuti di volo", "flight minutes"), tot["minuti"]),
                 (t("batterie", "batteries"), tot["batterie"]), (t("chilometri", "kilometres"), tot["km"])))
             + "</div>"]

    # ---- the table of the legs
    head = header_legs()
    parts.append(f"<h2>{t('Le tratte', 'The legs')}</h2><div class=\"scorri\"><table><thead><tr>"
                 + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr></thead><tbody>")
    for n, leg in enumerate(legs, 1):
        parts.append("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in leg_row(piano, n, leg)) + "</tr>")
    totals = {15: sum(len(leg.route.points) for leg in legs), 16: tot["scatti"], 17: tot["km"], 18: tot["minuti"],
              19: tot["batterie"]}
    parts.append("<tr class=\"totale\">" + "".join(
        f"<td>{t('Totale', 'Total') if k == 1 else cell(totals.get(k, ''))}</td>" for k in range(len(head)))
        + "</tr>")
    parts.append("</tbody></table></div>")

    # ---- the blocks
    parts.append(f"<h2>{t('I blocchi', 'The blocks')}</h2>"
                 + table(_header_blocks(), [_block_row(i, b, r, piano, cam)
                                            for i, (b, r) in enumerate(zip(blocks, computed["blocchi"]), 1)]))

    # ---- the coverage and the comparison
    if coverage:
        parts.append(f"<h2>{t('Copertura prevista', 'Predicted coverage')}</h2>"
                     + table([t("Voce", "Item"), t("Valore", "Value")],
                             [list(x) for x in _coverage_items(coverage)]))
    if comparison:
        single, tables = _comparison_parts(comparison)
        parts.append(f"<h2>{t('Confronto col volo', 'Comparison with the flight')}</h2>"
                     + table([t("Voce", "Item"), t("Valore", "Value")], [list(x) for x in single]))
        for title, rows in tables:
            columns = list(dict.fromkeys(k for r in rows for k in r))
            parts.append(f"<h3>{e(title)}</h3>" + table(columns, [[r.get(c) for c in columns] for r in rows]))

    # ---- one section per leg, with its map
    parts.append(f"<h2>{t('Tratta per tratta', 'Leg by leg')}</h2>")
    for n, leg in enumerate(legs, 1):
        line1, line2 = legmap.caption(piano, leg)
        parts.append(f"<section class=\"tratta\" id=\"tratta-{n}\"><h3>{n}. {e(line1)}</h3>"
                     f"<p class=\"numeri\">{e(line2)}</p>")
        if images:
            data = base64.b64encode(legmap.leg_image(piano, n - 1, background=background)).decode("ascii")
            parts.append(f"<img src=\"data:image/jpeg;base64,{data}\" alt=\"{e(line1)}\">")
        parts.append("<table><tbody>" + "".join(
            f"<tr><td>{e(item)}</td><td>{cell(value)}</td></tr>"
            for item, value in zip(head, leg_row(piano, n, leg))) + "</tbody></table></section>")

    parts.append(f"<p class=\"piede\">{t('Mappe', 'Maps')}: {e(_footer_maps(background))}. "
                 + t("Quote relative al punto di decollo di ogni tratta. Generato da {0}.",
                     "Heights relative to the takeoff point of each leg. Made by {0}.").format(PROGRAM)
                 + "</p></main></body></html>")
    return "".join(parts)
