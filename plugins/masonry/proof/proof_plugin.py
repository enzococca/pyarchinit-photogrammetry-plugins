"""The plugin's part of the proof (see ``make_proof.py``): the face, its shape, the stones, the USM, the outputs, the
numbers against the truth, ``results.json``, ``LEGGIMI.md`` and the figures."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

import make_proof as M

COURSE_TOP = M.TOP_OF_COURSES


def _truth_courses():
    _polys, truths = M.stones_truth()
    return sorted({round(c, 4) for _w, _h, c in truths})


def _stones_with_sam(chunk, entry, notes):
    """The stones of the face's elevation by the program's stone tool (SAM 2.1 at tiles), None without SAM."""
    from pyarchinit_photogrammetry.ai import sam_env
    from pyarchinit_photogrammetry.archaeo.stones import find_stones
    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    mode = sam_env.available()
    if mode is None:
        notes.append("SAM 2.1 non è installato su questo computer: pietre generate messe sulla faccia")
        return None
    ortho, _relief = elevation_paths(chunk, entry)
    u0, v0, u1, _v1 = entry["bounds"]
    t = time.monotonic()
    seg = sam_env.make_segmenter("small")
    try:
        found = find_stones(ortho, seg, bounds=(u0, v0, u1, COURSE_TOP + 0.02), min_size=0.2, max_size=0.9,
                            resolution=0.008, progress=M._progress("pietre (SAM)"))
    finally:
        close = getattr(seg, "close", None)
        if close is not None:
            close()
    notes.append(f"pietre trovate con SAM 2.1 small ({mode}) in {time.monotonic() - t:.0f} s")
    return found


def _match(found, truth_polys, truths):
    """Each true stone with the found stone that holds its centre (one at most): sizes compared."""
    from shapely.geometry import Point

    used, rows = set(), []
    for k, (p, (w, h, _c)) in enumerate(zip(truth_polys, truths)):
        c = Point(p.centroid.x, p.centroid.y)
        hit = next((i for i, f in enumerate(found) if i not in used and f.contains(c)), None)
        if hit is None:
            continue
        used.add(hit)
        b = found[hit].bounds
        rows.append({"true": k, "found": hit, "dw": (b[2] - b[0]) - w, "dh": (b[3] - b[1]) - h,
                     "iou": float(found[hit].intersection(p).area / found[hit].union(p).area)})
    return rows, len(found) - len(used)


def run_plugin(folder: Path, use_sam: bool = True) -> dict:
    import ifcopenshell
    import ifcopenshell.validate
    from shapely.geometry import box

    import pyarchinit_photogrammetry as ppg
    from ppg_masonry import faces, ifc, outputs, shape, stones, usm, walls

    doc = ppg.Document.open(folder / "muro.ppg")
    doc.meta["sito"] = "Prova murature"
    ch = doc.chunk
    for w in walls.walls(ch):
        walls.remove_wall(ch, w["id"], elevations_too=True)
    notes = []
    r = {"foto": len(ch.cameras), "allineate": len(ch.aligned_cameras), "tempi_s": ch.meta.get("proof_times")}
    t = time.monotonic()
    w = walls.new_wall(ch, "Muro della prova")
    entry = faces.plane_face(ch, w["id"], (M.X0, 0.0), (M.X0 + M.LENGTH, 0.0), side=(0.0, -5.0), resolution=0.004,
                             name="Faccia sud")
    r["faccia"] = {**walls.face_info(ch, entry["id"]), "tempo_s": time.monotonic() - t}
    entry = walls.need_face(ch, w["id"], entry["id"])[1]
    # the shape
    sh = shape.face_shape(ch, w["id"], entry["id"])
    import rasterio

    with rasterio.open(ch.data_dir / sh["bulge"]) as src:
        b = src.read(1)
        tr = src.transform
    rows, cols = np.indices(b.shape)
    u = tr.c + (cols + 0.5) * tr.a
    v = tr.f + (rows + 0.5) * tr.e
    truth = 100 * M.bulge(u, v) * np.cos(np.radians(M.LEAN_DEG))
    from rasterio import features

    polys, truths = M.stones_truth()
    on_stone = features.rasterize([(p.buffer(-0.02), 1) for p in polys], out_shape=b.shape, transform=tr, fill=0,
                                  dtype="uint8").astype(bool)
    upper = v > COURSE_TOP + 0.05
    ok = np.isfinite(b)
    err = (b - truth)[ok & (on_stone | upper)]
    offset = float(np.median(err))
    r["forma"] = {"strapiombo_gradi": [sh["lean_deg"], M.LEAN_DEG], "strapiombo_verso": sh["lean_toward"],
                  "strapiombo_azimut": sh["lean_azimuth"], "spanciamento_max_cm": [sh["bulge_max_cm"], 100 * M.BULGE],
                  "spanciamento_scarto_cm": {"mediana": offset, "p90_assoluto": float(np.percentile(np.abs(err), 90)),
                                             "p90_assoluto_tolto_lo_scarto_medio":
                                                 float(np.percentile(np.abs(err - offset), 90))},
                  "avviso_fascia": sh.get("band_warning")}
    heights, st_u = [0.5, 1.3, 2.1], [1.0, 3.0, 5.0]
    sec = shape.sections(ch, w["id"], entry["id"], heights=heights, stations=st_u)
    k = np.tan(np.radians(M.LEAN_DEG))
    r["sezioni"] = [{"u": s["u"],
                     "scarti_cm": [None if o["offset"] is None else 100 * o["offset"] for o in s["offsets"]],
                     "veri_cm": [100 * ((h - heights[0]) * k + M.bulge(s["u"], h) - M.bulge(s["u"], heights[0]))
                                 for h in heights]} for s in sec["stations"]]
    # the stones
    found = _stones_with_sam(ch, entry, notes) if use_sam else None
    used_sam = found is not None and len(found) > 0
    if not used_sam:
        if use_sam and found is not None:
            notes.append("SAM non ha trovato pietre: pietre generate messe sulla faccia")
        found = polys
    frames = ch.meta.setdefault("drawings", {}).setdefault("frames", {})
    frames.setdefault(entry["id"], {})["stones"] = {entry["id"]: [p.wkt for p in found]}
    units = {"1": box(0.0, 0.0, M.LENGTH, COURSE_TOP), "2": box(0.0, COURSE_TOP, M.LENGTH, M.HEIGHT)}
    frames[entry["id"]]["units"] = [{"wkt": g.wkt, "us": us, "tipo": "USM", "area": "1", "sito": "Prova murature",
                                     "attributes": {"d_stratigrafica": d}} for (us, g), d in
                                    zip(units.items(), ("Muratura a filari", "Muratura in pietrame"))]
    meas = stones.measure_face(ch, w["id"], entry["id"], region=units["1"])
    matched, extra = _match(found, polys, truths)
    true_mortar = 1 - sum(p.area for p in polys) / units["1"].area
    rc = meas["course_rows"]
    tc = _truth_courses()
    r["pietre"] = {"sam": used_sam, "vere": len(polys), "trovate": len(found), "abbinate": len(matched),
                   "in_piu": extra,
                   "larghezza_scarto_cm": {"mediana": 100 * float(np.median([m["dw"] for m in matched])),
                                           "p90_assoluto": 100 * float(np.percentile([abs(m["dw"]) for m in matched],
                                                                                     90))},
                   "altezza_scarto_cm": {"mediana": 100 * float(np.median([m["dh"] for m in matched])),
                                         "p90_assoluto": 100 * float(np.percentile([abs(m["dh"]) for m in matched],
                                                                                   90))},
                   "iou_mediana": float(np.median([m["iou"] for m in matched])),
                   "filari": [meas["courses"], len(tc)],
                   "quote_filari": [[c["height"] for c in rc if c["counts"]], tc],
                   "malta": [meas["mortar"], true_mortar], "tipo": meas["typology"]["line"]}
    # the USM
    found_rel = usm.propose(ch, w["id"], entry["id"])
    r["usm"] = {"proposti": [usm.proposal_text(p) for p in found_rel],
                "atteso": "USM 2 copre USM 1"}  # it lies on the whole top of the coursed wall
    walls.set_result(ch, w["id"], entry["id"], "usm", {"accepted_full": found_rel, "proposed": len(found_rel)})
    # the outputs
    out = folder / "esportazioni"
    if out.exists():
        shutil.rmtree(out)
    files = outputs.export_face(ch, w["id"], entry["id"], out)
    outputs.plate(ch, w["id"], entry["id"], folder, stem="figura_tavola", formats=("png",))
    res = ifc.export_wall(ch, w["id"], out / "muro.ifc", thickness=M.DEPTH)
    model = ifcopenshell.open(res["path"])
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    r["uscite"] = {"file": sorted(p.name for p in out.iterdir()), "scala_tavola": files.get("plate_scale"),
                   "ifc": {"spessore": res["thickness"], "parti": res["parts"], "rapporti": res["relations"],
                           "errori_validazione": len(logger.statements)}}
    r["note"] = notes
    doc.save()
    (folder / "results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    leggimi(folder, r)
    return r


def figures(folder: Path) -> None:
    """The bulge measured beside the truth, the stones found over the face's orthophoto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    import pyarchinit_photogrammetry as ppg
    from ppg_masonry import stones, walls

    doc = ppg.Document.open(folder / "muro.ppg")
    ch = doc.chunk
    w = walls.walls(ch)[0]
    eid = w["faces"][0]
    res = walls.results(ch, w["id"], eid)
    with rasterio.open(ch.data_dir / res["shape"]["bulge"]) as src:
        b = src.read(1)
        bb = src.bounds
        tr = src.transform
    rows, cols = np.indices(b.shape)
    u = tr.c + (cols + 0.5) * tr.a
    v = tr.f + (rows + 0.5) * tr.e
    truth = np.where(np.isfinite(b), 100 * M.bulge(u, v), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, data, title in ((axes[0], b, "spanciamento misurato (cm)"), (axes[1], truth, "spanciamento vero (cm)")):
        im = ax.imshow(data, extent=(bb.left, bb.right, bb.bottom, bb.top), cmap="RdBu_r", vmin=-3.5, vmax=3.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u (m)")
        ax.set_ylabel("quota (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(folder / "figura_spanciamento.png", dpi=130)
    plt.close(fig)
    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    entry = walls.elevation(ch, eid)
    ortho, _r = elevation_paths(ch, entry)
    with rasterio.open(ortho) as src:
        img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        ob = src.bounds
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(img, extent=(ob.left, ob.right, ob.bottom, ob.top))
    polys, _t = M.stones_truth()
    for p in polys:
        xy = np.asarray(p.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="cyan", lw=0.6)
    for p in stones.face_stones(ch, eid):
        xy = np.asarray(p.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="yellow", lw=0.8)
    ax.set_title("pietre: vere (azzurro), trovate (giallo)", fontsize=9)
    ax.set_xlim(ob.left, ob.right)
    ax.set_ylim(ob.bottom, M.TOP_OF_COURSES + 0.3)
    fig.tight_layout()
    fig.savefig(folder / "figura_pietre.png", dpi=130)
    plt.close(fig)


def _it(x, digits=2) -> str:
    """A number as an Italian text writes it (decimal comma)."""
    return f"{x:.{digits}f}".replace(".", ",")


def _pair(x, digits=2, k=1.0, unit=""):
    return f"{_it(x[0] * k, digits)}{unit} (vero {_it(x[1] * k, digits)}{unit})"


def leggimi(folder: Path, r: dict) -> None:
    f, p, s = r["forma"], r["pietre"], r["uscite"]
    toward = "verso chi guarda" if f["strapiombo_verso"] == "out" else "all'indietro"
    lines = ["# Prova del plugin Murature sulla catena vera del programma", "",
             "Un muro sintetico (6 x 2,6 m: cinque filari di pietre di contorno noto e sopra un'unità di pietrame, "
             "strapiombo di 2° verso chi guarda, spanciamento gaussiano di 3 cm) fotografato dal programma "
             f"({r['foto']} foto, {r['allineate']} allineate), con nuvola densa e modello del programma; poi il "
             "plugin: faccia su un piano, forma, sezioni, pietre, USM, tavola, DXF, GeoPackage, IFC.", "",
             "## Forma", "",
             f"- strapiombo: {_pair(f['strapiombo_gradi'], 3, unit='°')}, {toward}, azimut "
             f"{_it(f['strapiombo_azimut'], 1)}° (vero 180°)",
             f"- spanciamento massimo (99,5° percentile): {_pair(f['spanciamento_max_cm'], 2, unit=' cm')}",
             f"- carta dello spanciamento contro la verità (sulle pietre e sull'unità superiore, non sui giunti): "
             f"scarto mediano {_it(f['spanciamento_scarto_cm']['mediana'], 2)} cm, p90 assoluto "
             f"{_it(f['spanciamento_scarto_cm']['p90_assoluto'], 2)} cm "
             f"({_it(f['spanciamento_scarto_cm']['p90_assoluto_tolto_lo_scarto_medio'], 2)} cm tolto lo scarto "
             "mediano)",
             "", "Sezioni alle quote 0,5, 1,3, 2,1 m (scarti dalla più bassa, cm, misurati / veri):", ""]
    for st in r["sezioni"]:
        pairs = ", ".join(f"{_it(a, 1)} / {_it(b, 1)}" if a is not None else f"- / {_it(b, 1)}"
                          for a, b in zip(st["scarti_cm"], st["veri_cm"]))
        lines.append(f"- u = {_it(st['u'], 1)} m: {pairs}")
    lines += ["", "## Pietre e filari", "",
              ("Pietre trovate dallo strumento Pietre del programma (SAM 2.1) sul prospetto della faccia."
               if p["sam"] else "SAM non usato: le pietre generate sono messe sulla faccia come disegnate; il "
               "riconoscimento NON è stato provato."),
              "",
              f"- pietre vere {p['vere']}, trovate {p['trovate']}, abbinate {p['abbinate']}, in più {p['in_piu']}",
              f"- larghezza: scarto mediano {_it(p['larghezza_scarto_cm']['mediana'], 2)} cm, p90 "
              f"{_it(p['larghezza_scarto_cm']['p90_assoluto'], 2)} cm; altezza: scarto mediano "
              f"{_it(p['altezza_scarto_cm']['mediana'], 2)} cm, p90 "
              f"{_it(p['altezza_scarto_cm']['p90_assoluto'], 2)} cm; "
              f"IoU mediana {_it(p['iou_mediana'], 3)}",
              f"- filari: {p['filari'][0]} (veri {p['filari'][1]}); quote "
              + ", ".join(f"{_it(x, 3)}" for x in p["quote_filari"][0]) + " (vere "
              + ", ".join(f"{_it(x, 3)}" for x in p["quote_filari"][1]) + ")",
              f"- malta: {_pair(p['malta'], 1, 100, ' %')}",
              f"  (la malta misurata è la parte dell'unità che non è pietra trovata: {p['vere'] - p['abbinate']} "
              "pietre non trovate - quasi tutte ai bordi, dove l'ortofoto del prospetto ha dei vuoti - e i contorni "
              "trovati appena dentro le pietre la fanno crescere; con le pietre disegnate a mano la misura segue il "
              "disegno)",
              f"- tipo: {p['tipo']}", "", "## USM", "",
              f"- proposti: {'; '.join(r['usm']['proposti']) or 'nessuno'} (atteso: {r['usm']['atteso']})", "",
              "## Uscite", "",
              f"- tavola in scala 1:{s['scala_tavola']} (SVG, PDF), DXF della muratura, GeoPackage, tabelle, "
              f"spanciamento: {len(s['file'])} file in `esportazioni/`",
              f"- IFC: spessore {_it(s['ifc']['spessore'], 2)} m (dato), {s['ifc']['parti']} USM come parti, "
              f"{s['ifc']['rapporti']} rapporti, errori di validazione {s['ifc']['errori_validazione']}", ""]
    if r.get("note"):
        lines += ["## Note", ""] + [f"- {n}" for n in r["note"]] + [""]
    lines += ["Figure: `figura_spanciamento.png`, `figura_pietre.png`, `figura_tavola.png`. Numeri in `results.json`.",
              ""]
    (folder / "LEGGIMI.md").write_text("\n".join(lines), encoding="utf-8")
