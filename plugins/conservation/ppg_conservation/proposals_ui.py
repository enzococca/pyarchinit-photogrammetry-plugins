"""The proposals in the window (the *Proposte...*, *Come questo*, *Accetta*, *Accetta tutte* and *Scarta* buttons): the
plugin's handlers live here, as its forms in :mod:`.forms`; the detectors are :mod:`.detect`. Each function takes the
plugin."""
from __future__ import annotations

from .texts import num, t
from .vocabulary import UNNAMED


def parts(geom, kind: str):
    """The parts of ``geom`` (a proposal clipped to its window) a feature of ``kind`` can have: polygons for an area,
    lines for a crack; None when nothing of them is left."""
    import shapely
    from shapely.geometry import MultiLineString, MultiPolygon

    want = "Polygon" if kind == "decay" else "LineString"
    flat = [p for g in shapely.get_parts(geom) for p in shapely.get_parts(g)]  # a collection may hold multi-parts
    found = [p for p in flat if p.geom_type == want and not p.is_empty]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return MultiPolygon(found) if kind == "decay" else MultiLineString(found)


def segmenter(plugin):
    """The program's SAM 2.1 worker; NotReady with the way to install it when it is missing."""
    from .records import NotReady

    window = plugin.context.window
    if window is not None and hasattr(window, "_segmenter"):
        try:
            return window._segmenter()
        except ImportError:
            pass
    try:
        from pyarchinit_photogrammetry.ai.sam_env import make_segmenter

        return make_segmenter()
    except ImportError:
        raise NotReady(t("SAM 2.1 non è installato: installalo dagli strumenti SAM del programma (per esempio "
                         "Pietre (SAM)), poi riprova.", "SAM 2.1 is not installed: install it from the program's "
                         "SAM tools (for instance Stones (SAM)), then try again.")) from None


def chosen_area(plugin, chunk):
    """The one polygon feature chosen in the table, or None."""
    from . import records

    ids = [i for i in (plugin.pane.selected_ids() if plugin.pane is not None else []) if i]
    if len(ids) != 1:
        return None
    rec = records.feature(chunk, ids[0])
    return rec if rec["geometry"].upper().startswith(("POLYGON", "MULTIPOLYGON")) else None


def restore(chunk, before) -> None:
    """The records as they were before a change that failed half-way."""
    import copy

    if before is None:
        chunk.meta.pop("conservation", None)
    else:
        chunk.meta["conservation"] = copy.deepcopy(before)


def do_propose(plugin) -> None:
    """The *Proposte* dialog: a detector of :mod:`.detect` on the area chosen in the table, on a window drawn on the
    view or on the whole face, in the background; its results join the campaign as proposals (``source``
    ``proposed:<detector>``)."""
    from . import forms, texts
    from .vocabulary import load

    chunk, entry, c = plugin._need_campaign()
    selected = plugin._chosen_area(chunk)
    code = plugin.pane.term_code() if plugin.pane is not None else None
    vocab = load(chunk)
    term = vocab.term(code) if code else None
    v = forms.proposals_form(plugin, selected, None if term is None else vocab.name(term.code, texts.lang()))
    if v is None:
        return
    if v["window"] == "draw" and v["detector"] != "like":
        plugin._request("polygon", entry, lambda xy: run(plugin, chunk, entry, c, v, selected, term,
                                                         plugin._geometry("polygon", xy)),
                        t("Finestra delle proposte: clic per i vertici (o trascina un riquadro), doppio clic o Invio "
                          "per chiudere; Esc annulla", "Window of the proposals: click the vertices (or drag a box), "
                          "double click or Enter to close; Esc cancels"))
        return
    from shapely import wkt

    area = None
    if v["window"] == "selected" and selected is not None and v["detector"] != "like":  # like: the example is inside
        area = wkt.loads(selected["geometry"])
    run(plugin, chunk, entry, c, v, selected, term, area)


def _stones_first(plugin, v: dict) -> bool:
    """Cracks on a face with no stones drawn: said before running, with the choice to go on with the largest opening
    (or another one) or to draw the stones first. False when the user stops."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import forms
    from .records import NotReady

    most = forms.number(v.get("max_width_mm"), positive=True) or 0.0
    limit = t(f"le linee più larghe di {num(most, 0)} mm saranno tolte come giunti",
              f"lines wider than {num(most, 0)} mm will be dropped as joints") if most else \
        t("nessuna apertura massima: anche i giunti saranno proposti", "no largest opening: the joints will be "
          "proposed too")
    answer = plugin._choose(
        t(f"Le pietre di questo prospetto non sono disegnate: su un muro di pietra i giunti di malta possono sembrare "
          f"lesioni. Senza pietre {limit}.", f"The stones of this elevation are not drawn: on a stone wall the mortar "
          f"joints may look like cracks. Without stones {limit}."),
        [t("Continua", "Go on"), t("Cambia l'apertura massima...", "Change the largest opening..."),
         t("Disegno prima le pietre", "I draw the stones first")])
    if answer is None:
        return False
    if answer == 2:
        raise NotReady(t("Disegna le pietre sul prospetto (Strumenti > Pietre (SAM), o il plugin Murature), poi "
                         "torna a Proposte...: le lesioni sui giunti fra due pietre non saranno proposte.",
                         "Draw the stones on the elevation (Tools > Stones (SAM), or the Masonry plugin), then come "
                         "back to Proposals...: the cracks on joints between two stones will not be proposed."))
    if answer == 1:
        got = plugin._ask(t("Apertura massima", "Largest opening"), [
            Field("max_width_mm", t("Apertura massima di una lesione (mm, 0: nessun limite)",
                                    "Largest opening of a crack (mm, 0: no limit)"), float(most), minimum=0.0,
                  maximum=1000.0, decimals=1)],
            intro=t("I giunti fra le pietre di questo muro sono in genere più larghi delle lesioni: misurane uno.",
                    "The joints between the stones of this wall are usually wider than the cracks: measure one."))
        if got is None:
            return False
        v["max_width_mm"] = got["max_width_mm"]
    return True


def run(plugin, chunk, entry: dict, c: dict, v: dict, selected, term, area) -> None:
    """The detector ``v["detector"]`` on the window ``area`` (None: the whole face) in the background. Cracks and
    losses on a window over ``plugin.pixel_budget`` pixels run by tiles; the colour classes are refused there (one
    k-means for a whole big face: draw a window)."""
    from shapely import wkt

    from . import detect, forms
    from .records import NotReady

    name = v["detector"]
    # what a proposal with no term of its own gets: the working term "da nominare", or - for SAM's segments only,
    # when the user asked for it - the term chosen in the list
    default = UNNAMED
    if name == "sam" and v.get("use_term") and term is not None and term.kind == "decay":
        default = term.code
    reference = None
    if name == "like":
        if selected is None:
            raise NotReady(t("Come questo: scegli nella tabella l'area che fa da esempio.",
                             "Like this: choose in the table the area that is the example."))
        if selected["code"] == UNNAMED:
            raise NotReady(t("Come questo: l'area d'esempio è ancora da nominare; dalle prima un termine.",
                             "Like this: the example area is still to be named; give it a term first."))
        reference, default = wkt.loads(selected["geometry"]), selected["code"]
    bounds = None if area is None else area.bounds
    budget = int(getattr(plugin, "pixel_budget", None) or detect.PIXEL_BUDGET)
    extent = detect.window_extent(chunk, entry, bounds)
    if name == "colours" and extent is not None and extent[2] > budget:
        raise NotReady(t(f"Classi di colore: la finestra ha {num(extent[2] / 1e6, 1)} milioni di pixel, oltre i "
                         f"{num(budget / 1e6, 0)} che si leggono in una volta. Disegna una finestra (Finestra: "
                         "disegna una finestra) o scegli un'area nella tabella.",
                         f"Colour classes: the window has {num(extent[2] / 1e6, 1)} million pixels, beyond the "
                         f"{num(budget / 1e6, 0)} read at once. Draw a window (Window: draw a window) or choose an "
                         "area in the table."))
    if name == "cracks" and not detect.stones_drawn(chunk, entry) and not _stones_first(plugin, v):
        return
    seg = plugin._segmenter() if name in ("sam", "like") else None
    res = forms.number(v.get("resolution_mm"), positive=True)
    res = None if not res else res / 1000.0
    most = forms.number(v.get("max_width_mm"), positive=True)
    stats: dict = {}

    def work(progress):
        ext = detect.window_extent(chunk, entry, bounds)
        if ext is None:
            return None
        if ext[2] <= budget:  # a big window is not read whole just to see it has a surface
            win = detect.read_window(chunk, entry, bounds)
            if win is None or win.empty:
                return None
        if name == "cracks":
            return detect.tiled(detect.cracks, chunk, entry, bounds, budget=budget, progress=progress,
                                min_length=float(v["min_length_cm"]) / 100.0, max_width_mm=most or None,
                                stats=stats)
        if name == "loss":
            return detect.tiled(detect.material_loss, chunk, entry, bounds, budget=budget, progress=progress,
                                min_depth=float(v["min_depth_cm"]) / 100.0, min_area=float(v["min_area_m2"]),
                                stats=stats)
        if name == "colours":
            return detect.colour_classes(chunk, entry, bounds, k=int(v["k"]), progress=progress)
        found = detect.sam_areas(chunk, entry, bounds, thoroughness=v.get("thoroughness") or "normale",
                                 resolution=res, progress=progress, segmenter=seg)
        if name == "like":
            radius = float(v["radius"])
            keep = detect.like_this(chunk, entry, [f["geometry"] for f in found], reference, radius)
            found = [found[i] for i in keep]
            for f in found:
                f["notes"] += t(f"; come l'area {selected['id']} (distanza massima {num(radius, 2)})",
                                f"; like the area {selected['id']} (largest distance {num(radius, 2)})")
        return found

    plugin.context.run(t("Conservazione: proposte", "Conservation: proposals"), work,
                       lambda result: plugin._guarded(proposed, plugin, chunk, c, name, area, default, result,
                                                      stats, most))


def proposed(plugin, chunk, c: dict, name: str, area, default: str, result, stats: dict | None = None,
             most: float | None = None) -> None:
    """The detector's results as proposals of the campaign ``c``: clipped to the window ``area``, those with no
    term given ``default`` (the working term "da nominare" unless the user chose otherwise), the colour classes
    named by the user first. A failure half-way leaves the records as they were. The message says how many lines
    were dropped as joints (``stats``) and how many tiles the window took."""
    from . import forms, records

    if result is None:
        plugin._message("information", t("La finestra non ha superficie: niente da proporre.",
                                         "The window has no surface: nothing to propose."))
        return
    stats = stats or {}
    source = "proposed:" + {"like": "sam", "loss": "material_loss"}.get(name, name)
    found = result
    if name == "colours":
        named = forms.colour_classes_form(plugin, chunk, result) if result else {}
        if named is None:
            return
        found = [{"kind": "decay", "code": named[cl["class"]], "geometry": g, "width_mm": None, "depth_mm": None,
                  "notes": cl["notes"], "coverage": cl["coverage"]}
                 for cl in result if cl["class"] in named for g in cl["polygons"]]
    before = plugin._snapshot(chunk)
    added, unnamed, check = 0, 0, 0
    try:
        for p in found:
            geom = p["geometry"] if area is None else parts(p["geometry"].intersection(area), p["kind"])
            if geom is None:
                continue
            code = p["code"] or default
            unnamed += code == UNNAMED
            check += bool(p.get("check"))
            cover = float(p.get("coverage") or 0.0)
            notes = p["notes"] + t(f"; rilievo sul {num(cover * 100, 0)} % della finestra",
                                   f"; relief on {num(cover * 100, 0)} % of the window")
            records.add_feature(chunk, c["id"], p["kind"], code, geom, status="proposed", source=source,
                                width_mm=None if p.get("width_mm") is None else round(float(p["width_mm"]), 1),
                                depth_mm=p.get("depth_mm"), notes=notes)
            added += 1
    except Exception:
        restore(chunk, before)
        raise
    if added:
        plugin._changed(chunk, before, t("proposte aggiunte", "proposals added"))
    text = t(f"{added} proposte aggiunte alla campagna (layer Proposte): Accetta o Scarta.",
             f"{added} proposals added to the campaign (layer Proposte): Accept or Discard.") if added else \
        t("Nessuna proposta nella finestra.", "No proposal in the window.")
    if name == "cracks":
        if most and stats.get("wide") is not None:
            text += " " + t(f"{stats['wide']} linee più larghe di {num(most, 0)} mm tolte come giunti.",
                            f"{stats['wide']} lines wider than {num(most, 0)} mm dropped as joints.")
        if stats.get("on_joints"):
            text += " " + t(f"{stats['on_joints']} linee sui giunti fra le pietre disegnate tolte.",
                            f"{stats['on_joints']} lines on joints between the drawn stones dropped.")
    if stats.get("tiles"):
        text += " " + t(f"La finestra è stata letta in {stats['tiles']} tasselli.",
                        f"The window was read in {stats['tiles']} tiles.")
    if check:
        text += " " + t(f"{check} depressioni molto estese sono da verificare: possono essere la forma del muro.",
                        f"{check} very wide depressions are to check: they may be the shape of the wall.")
    if unnamed:
        text += " " + t(f"{unnamed} sono da nominare: scegline una, dalle un termine con Modifica... e usa Come "
                        "questo per le simili.", f"{unnamed} are to be named: choose one, give it a term with "
                        "Edit... and use Like this for the similar ones.")
    plugin._message("information", text)


def do_like(plugin) -> None:
    """*Come questo*: the campaign's proposed areas from SAM (and those still to be named) that look like the
    area chosen in the table take its term (:func:`.detect.like_this`, in the background; one entry of the
    undo)."""
    from shapely import wkt

    from . import detect, records

    chunk, entry, c = plugin._need_campaign()
    ref = plugin._chosen_area(chunk)
    if ref is None or ref["code"] == UNNAMED:
        raise records.NotReady(t("Come questo: scegli nella tabella un'area con il termine giusto.",
                                 "Like this: choose in the table an area with the right term."))
    others = [f for f in records.features(chunk, c["id"], status="proposed")
              if f["id"] != ref["id"] and f["kind"] == "decay"
              and (str(f.get("source", "")).startswith("proposed:sam") or f["code"] == UNNAMED)]
    if not others:
        raise records.NotReady(t("La campagna non ha aree proposte da SAM o da nominare: usa prima Proposte...",
                                 "The campaign has no area proposed by SAM or to be named: use Proposals... "
                                 "first."))
    radius = float((plugin.context.settings.get("proposals") or {}).get("radius", 1.0))
    shapes = [wkt.loads(f["geometry"]) for f in others]
    reference = wkt.loads(ref["geometry"])

    def work(_progress):
        return detect.like_this(chunk, entry, shapes, reference, radius)

    plugin.context.run(t("Conservazione: come questo", "Conservation: like this"), work,
                       lambda keep: plugin._guarded(liked, plugin, chunk, ref, others, keep, radius))


def liked(plugin, chunk, ref: dict, others: list, keep: list, radius: float) -> None:
    from . import records

    known = {f["id"] for f in records.features(chunk)}
    changed = [others[i] for i in keep if others[i]["code"] != ref["code"] and others[i]["id"] in known]
    if not changed:
        plugin._message("information", t("Nessun'altra area proposta somiglia a quella scelta.",
                                         "No other proposed area looks like the chosen one."))
        return
    before = plugin._snapshot(chunk)
    try:
        for f in changed:
            records.update_feature(chunk, f["id"], code=ref["code"], notes=(f.get("notes") or "") + t(
                f"; termine come l'area {ref['id']} (distanza massima {num(radius, 2)})",
                f"; term as the area {ref['id']} (largest distance {num(radius, 2)})"))
    except Exception:
        restore(chunk, before)
        raise
    plugin._changed(chunk, before, t("termine come questo", "term like this"))
    plugin._message("information", t(f"{len(changed)} aree proposte prendono il termine dell'area scelta.",
                                     f"{len(changed)} proposed areas take the term of the chosen area."))


def do_accept(plugin) -> None:
    from . import records

    chunk, _entry, _c = plugin._need_campaign()
    found = [f for f in plugin._selected(chunk, one=False) if f["status"] == "proposed"]
    if not found:
        raise records.NotReady(t("Gli elementi scelti non sono proposte.", "The features chosen are no proposals."))
    before = plugin._snapshot(chunk)
    records.accept(chunk, [f["id"] for f in found])
    plugin._changed(chunk, before, t("proposte accettate", "proposals accepted"))


def do_accept_all(plugin) -> None:
    """*Accetta tutte*: the campaign's proposals that have a term are accepted; those still to be named stay
    proposals, and the message says how many."""
    from . import records

    chunk, _entry, c = plugin._need_campaign()
    found = records.features(chunk, c["id"], status="proposed")
    if not found:
        raise records.NotReady(t("La campagna non ha proposte.", "The campaign has no proposals."))
    named = [f for f in found if f["code"] != UNNAMED]
    left = len(found) - len(named)
    if not named:
        raise records.NotReady(t(f"Le {left} proposte sono tutte da nominare: dai un termine prima di accettare "
                                 "(Modifica..., o Come questo).", f"The {left} proposals are all to be named: give "
                                 "them a term before accepting (Edit..., or Like this)."))
    before = plugin._snapshot(chunk)
    records.accept(chunk, [f["id"] for f in named])
    plugin._changed(chunk, before, t("proposte accettate", "proposals accepted"))
    if left:
        plugin._message("information", t(f"{len(named)} proposte accettate; {left} sono ancora da nominare (restano "
                                         "proposte: Modifica..., o Come questo).",
                                         f"{len(named)} proposals accepted; {left} are still to be named (they stay "
                                         "proposals: Edit..., or Like this)."))


def do_discard(plugin) -> None:
    from . import records

    chunk, _entry, _c = plugin._need_campaign()
    found = plugin._selected(chunk, one=False)
    accepted = [f for f in found if f["status"] != "proposed"]
    if accepted and not plugin._confirm(
            t(f"Fra gli elementi scelti {len(accepted)} non sono proposte ma elementi del rilievo: rimuoverli "
              "comunque? Modifica > Annulla li riporta.",
              f"Among the features chosen {len(accepted)} are no proposals but features of the survey: remove "
              "them anyway? Edit > Undo brings them back.")):
        return
    before = plugin._snapshot(chunk)
    records.discard(chunk, [f["id"] for f in found])
    plugin._changed(chunk, before, t("elementi scartati", "features discarded"))


__all__ = ["parts", "segmenter", "chosen_area", "restore", "do_propose", "run", "proposed", "do_like", "liked",
           "do_accept", "do_accept_all", "do_discard"]
