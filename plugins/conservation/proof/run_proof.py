"""A proof run of the plugin on a real elevation: the detectors on a window of the face, a campaign with the proposals
that carry a term accepted, the plate, the tables, the drawings, the IFC 4.3 file, and a ``LEGGIMI.md`` with the
numbers and the times. Nothing is copied anywhere else.

The project given is never written: it is copied first - the ``.ppg`` file and its WHOLE ``.ppg.files`` folder
(every chunk's photos links, models, rasters: cloned on APFS when possible, a full copy elsewhere, so mind the disk)
- into ``FOLDER/progetto/``, and the run adds its campaign to the copy and writes under the
copy's data. ``--out`` inside the project's data folder is refused, and so is a ``FOLDER/progetto/`` that is
already there (choose another folder or remove it).

    python run_proof.py PROJECT.ppg --elevation "NAME" --window U0 V0 U1 V1 --out FOLDER
        [--chunk LABEL] [--sam] [--min-depth-cm 2] [--min-area-m2 0.0025] [--min-length-cm 10]
        [--max-width-mm 12] [--colours 5] [--thickness 0.5]

``--window`` in (u, v) metres of the elevation (the coordinates the program shows on it); ``--sam`` adds SAM 2.1's
segments (they stay proposals still to be named: the restorer names them). Colour classes are counted, not
accepted: naming a class is the restorer's choice."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _elevation(chunk, name: str) -> dict:
    found = [e for e in chunk.meta.get("elevations") or [] if e.get("name") == name or e.get("id") == name]
    if not found:
        names = ", ".join(f"«{e.get('name')}» ({e.get('id')})" for e in chunk.meta.get("elevations") or [])
        raise SystemExit(f"no elevation {name!r} in the chunk «{chunk.label}»: {names or 'none'}")
    return found[0]


def copy_project(project: Path, out: Path) -> Path:
    """``project`` and its data folder copied into ``out/progetto/``; the path of the copy's ``.ppg``."""
    import shutil
    import subprocess

    project = project.resolve()
    files = project.with_name(project.name + ".files")
    out = out.resolve()
    if files == out or files in out.parents:
        raise SystemExit(f"--out {out} is inside the project's data folder {files}: choose a folder outside it")
    target = out / "progetto"
    if target.exists():
        raise SystemExit(f"{target} is already there: choose another --out or remove it")
    target.mkdir(parents=True)
    shutil.copy2(project, target / project.name)
    if files.is_dir():
        dest = target / files.name
        cloned = sys.platform == "darwin" and subprocess.run(
            ["cp", "-cRp", str(files), str(dest)], capture_output=True).returncode == 0
        if not cloned:
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(files, dest)
    return target / project.name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("project")
    p.add_argument("--elevation", required=True)
    p.add_argument("--window", nargs=4, type=float, required=True, metavar=("U0", "V0", "U1", "V1"))
    p.add_argument("--out", required=True)
    p.add_argument("--chunk")
    p.add_argument("--sam", action="store_true")
    p.add_argument("--min-depth-cm", type=float, default=2.0)
    p.add_argument("--min-area-m2", type=float, default=0.0025)
    p.add_argument("--min-length-cm", type=float, default=10.0)
    p.add_argument("--max-width-mm", type=float, default=12.0, help="wider lines are joints (0: no limit)")
    p.add_argument("--colours", type=int, default=5)
    p.add_argument("--thickness", type=float, default=0.5)
    a = p.parse_args(argv)

    import pyarchinit_photogrammetry as ppg

    from ppg_conservation import assess, detect, ifc43, outputs, records
    from ppg_conservation.vocabulary import UNNAMED

    out = Path(a.out).resolve()
    times, lines = {}, []
    t0 = time.perf_counter()
    project = copy_project(Path(a.project), out)  # the project given is never written
    times["copia del progetto"] = time.perf_counter() - t0
    doc = ppg.Document.open(project)
    chunk = doc.chunk if a.chunk is None else next((c for c in doc.chunks if c.label == a.chunk), None)
    if chunk is None:
        raise SystemExit(f"no chunk {a.chunk!r}")
    entry = _elevation(chunk, a.elevation)
    times["apertura"] = time.perf_counter() - t0 - times["copia del progetto"]
    bounds = tuple(a.window)
    c = records.new_campaign(chunk, entry["id"], name="Prova del plugin", method="proposte automatiche (prova)")

    def run(name, fn):
        s = time.perf_counter()
        result = fn()
        times[name] = time.perf_counter() - s
        print(f"{name}: {times[name]:.1f} s", flush=True)
        return result

    found = {}
    crack_stats: dict = {}
    found["cracks"] = run("lesioni", lambda: detect.tiled(
        detect.cracks, chunk, entry, bounds, min_length=a.min_length_cm / 100.0, max_width_mm=a.max_width_mm or None,
        stats=crack_stats))
    found["material_loss"] = run("mancanze", lambda: detect.tiled(
        detect.material_loss, chunk, entry, bounds, min_depth=a.min_depth_cm / 100.0, min_area=a.min_area_m2))
    classes = run("classi di colore", lambda: detect.colour_classes(chunk, entry, bounds, k=a.colours))
    if a.sam:
        from pyarchinit_photogrammetry.ai.sam_env import make_segmenter

        found["sam"] = run("SAM", lambda: detect.sam_areas(chunk, entry, bounds, segmenter=make_segmenter()))
    made = []
    for name, items in found.items():
        for f in items:
            cover = float(f.get("coverage") or 0.0)
            made.append(records.add_feature(
                chunk, c["id"], f["kind"], f["code"] or UNNAMED, f["geometry"], status="proposed",
                source=f"proposed:{name}", width_mm=None if f.get("width_mm") is None else round(float(f["width_mm"]),
                                                                                                 1),
                depth_mm=f.get("depth_mm"), notes=f["notes"] + f"; rilievo sul {cover * 100:.0f} % della finestra"))
    accepted = records.accept(chunk, [f["id"] for f in made if f["code"] != UNNAMED])
    totals = records.totals(chunk, c["id"])
    cc, cc_rule = assess.suggest_cc(chunk, c["id"])
    exported = run("tavola, tabelle, disegni", lambda: outputs.export_campaign(chunk, c["id"], out / "campagna"))
    try:
        ifc = run("IFC 4.3", lambda: ifc43.export(chunk, [c["id"]], out / "conservazione.ifc", thickness=a.thickness))
    except Exception as e:  # a section, or no ifcopenshell: said in the notes, the rest stays
        ifc = {"error": str(e)}
    doc.save()
    times["totale"] = time.perf_counter() - t0

    def n(items, **match):
        return sum(1 for f in items if all(f.get(k) == v for k, v in match.items()))

    widths = sorted(f["width_mm"] for f in found["cracks"] if f.get("width_mm") is not None)
    depths = sorted(f["depth_mm"] for f in found["material_loss"] if f.get("depth_mm") is not None)
    median_width = f"{widths[len(widths) // 2]:.1f}" if widths else "-"
    deepest = f"{depths[-1]:.0f}" if depths else "-"
    lines += ["# Prova del plugin Conservazione", "",
              f"Progetto: una copia di `{Path(a.project).name}` in `{project.parent.relative_to(out)}/` "
              f"(l'originale non è stato toccato), chunk «{chunk.label}», prospetto «{entry['name']}» "
              f"({entry['id']}, {entry['resolution'] * 1000:.1f} mm/px).",
              f"Finestra (u, v): {bounds[0]:.2f} {bounds[1]:.2f} - {bounds[2]:.2f} {bounds[3]:.2f} m "
              f"({(bounds[2] - bounds[0]) * (bounds[3] - bounds[1]):.2f} m²).", "",
              "## Proposte", "",
              f"- Lesioni: {len(found['cracks'])} (lunghezza totale "
              f"{sum(f['geometry'].length for f in found['cracks']):.2f} m; apertura mediana "
              f"{median_width} mm; {crack_stats.get('wide', 0)} linee più larghe di {a.max_width_mm:g} mm tolte "
              f"come giunti; pietre {'disegnate' if crack_stats.get('stones') else 'non disegnate'})",
              f"- Mancanze di materiale: {len(found['material_loss'])} "
              f"(mancanza {n(found['material_loss'], code=detect.MISSING_PART)}, "
              f"alveolizzazione {n(found['material_loss'], code=detect.ALVEOLIZATION)}, da nominare "
              f"{n(found['material_loss'], code=None)}, di cui da verificare "
              f"{n(found['material_loss'], check=True)}; profondità massima {deepest} mm)",
              f"- Classi di colore: {len(classes)} (" + ", ".join(
                  f"{cl['class'] + 1}: {cl['share'] * 100:.0f} %{' ' + cl['looks'] if cl['looks'] else ''}"
                  for cl in classes) + "), non accettate: le nomina il restauratore"]
    if "sam" in found:
        lines.append(f"- Segmenti di SAM: {len(found['sam'])} (da nominare)")
    lines += ["", "## Campagna", "",
              f"- Proposte aggiunte {len(made)}, accettate {accepted} (quelle con un termine), da nominare "
              f"{len(made) - accepted}",
              "- Aree di degrado: " + (", ".join(f"{k} {v:.3f} m²" for k, v in totals["area_by_code"].items())
                                       or "nessuna"),
              "- Lesioni: " + (", ".join(f"{k} {v:.2f} m" for k, v in totals["length_by_code"].items()) or "nessuna"),
              "- Quota della faccia degradata: "
              + ("-" if totals["share_of_face"] is None else f"{totals['share_of_face'] * 100:.2f} %"),
              f"- Suggerimento CC (soglie del plugin): {cc} - {cc_rule}", "",
              "## File", ""]
    for key, value in exported.items():
        if isinstance(value, Path):
            lines.append(f"- {key}: `{value.relative_to(out)}`")
    lines.append("- IFC: " + (f"`{Path(ifc['path']).name}` ({ifc['hosts']} pareti, {ifc['features']} elementi)"
                               if "path" in ifc else f"non scritto ({ifc['error']})"))
    lines += ["", "## Tempi", ""] + [f"- {k}: {v:.1f} s" for k, v in times.items()]
    lines += ["", "Le figure di questa prova restano private: non si pubblicano."]
    (out / "LEGGIMI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
