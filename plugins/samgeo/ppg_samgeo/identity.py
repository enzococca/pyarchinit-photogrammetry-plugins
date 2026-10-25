"""Optional host bulk-identification UI, before creating a run's units."""
from __future__ import annotations


def convert_dialog(plugin, run_id):
    """Return False only for an older host, which uses Samgeo's original conversion dialog."""
    try:
        from pyarchinit_photogrammetry.archaeo.stratigraphy import units_from_polygons
        from pyarchinit_photogrammetry.archaeo.unit_identity import apply_changes
        from pyarchinit_photogrammetry.gui.unit_identity_dialog import UnitIdentityDialog
        from pyarchinit_photogrammetry.gui.unit_identity_actions import reserved_units
    except ImportError:
        return False
    from .runs import features, unit_rows
    from .view import selected
    from .texts import t

    chunk, run = plugin._run_of(run_id)
    if run is None:
        return True
    made = set(run.get("units_made") or [])
    rows = [(g, a) for g, a in features(chunk, run) if int(a.get("n") or 0) not in made]
    if not rows:
        plugin._message("info", t("Le proposte sono già unità. Selezionale e usa «Identifica / raggruppa...» "
                                   "nel pannello Layer per assegnare numeri, nomi e tipo UT.",
                                   "The proposals are already units. Select them and use 'Identify / group...' "
                                   "in the Layers pane to assign numbers, names and UT kind."))
        return True
    window = plugin.context.window
    site = window.project_site() or ""
    geoms, attributes = unit_rows(run, rows)
    kind = "USM" if any(w in run.get("prompt", "") for w in ("wall", "brick")) else "US"
    units = units_from_polygons(geoms, tipo=kind, attributes=attributes, sito=site or None)
    chosen = selected(plugin._view(), run)
    indices = {i for i, (_, a) in enumerate(rows) if int(a.get("n") or 0) in chosen}
    # Existing identities, including those in other chunks and a connected archive, are reserved.
    reserved = [*window.units, *reserved_units(window, units)]
    changes = UnitIdentityDialog.ask(units, indices, parent=window, reserved=reserved,
                                     defaults={"scope": "selected" if indices else "all", "tipo": kind,
                                               "progetto": site, "number_mode": "missing"})
    if changes is None:
        return True
    apply_changes(units, changes)
    # The dialog's plan contains only the chosen objects: an unselected proposal is not converted.
    selected_indices = [change.index for change in changes]
    wanted = [rows[i] for i in selected_indices]
    values = [{**units[i].attributes, "us": units[i].us, "tipo": units[i].tipo,
               "area": units[i].area, "sito": units[i].sito} for i in selected_indices]
    if wanted:
        plugin.make_units(run_id, wanted, attributes=values)
    return True
