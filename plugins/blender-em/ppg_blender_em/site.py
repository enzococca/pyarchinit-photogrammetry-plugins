"""The epochs of the em.json as groups of the unit volumes on the web site of a survey.

Since program 2.1.0 (site hook 3) a plugin can group the stratigraphic units of a chunk; the site colours their
volumes by group and gives a switch per group. The groups here are the epochs of the em.json paired with the chunk
(each unit in the group of its ``has_first_epoch``). Only names, colours and unit numbers are given: nothing of the
Blender scene (its manifest, paths, reference system or shift) reaches the site.
"""
from __future__ import annotations

import re

_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
TITLE = {"it": "Epoche", "en": "Epochs"}


def _unit(unit) -> dict | None:
    """A unit as the site names it: type and number, with the area when known; None without a number."""
    us = unit.us
    if not us:
        return None
    out = {"tipo": unit.tipo or "US", "us": us}
    if unit.area:
        out["area"] = unit.area
    return out


def unit_groups(matrix) -> list[dict]:
    """The ``unit_groups`` contribution of a matrix (:class:`..emjson.Matrix`): one group per epoch that holds
    units, ordered by start time (epochs without one last) then name; ``[]`` without a matrix or epochs. Units
    without an epoch are not sent: the site keeps them in its own group with the program's colours."""
    if matrix is None or not matrix.epochs:
        return []
    members: dict[str, list] = {}
    graphs: dict[str, set] = {}
    epochs = {}
    for unit in matrix.units:
        if unit.epoch is None:
            continue
        named = _unit(unit)
        if named is None:
            continue
        epochs[unit.epoch.id] = unit.epoch
        if named not in members.setdefault(unit.epoch.id, []):
            members[unit.epoch.id].append(named)
        graphs.setdefault(unit.epoch.id, set()).add(unit.graph_id)
    if not members:
        return []

    def label(epoch) -> str:
        return (epoch.name or "").strip() or epoch.id

    order = sorted(epochs.values(), key=lambda e: (e.start is None, e.start or 0.0, label(e), e.id))
    counts: dict[str, int] = {}
    for e in order:
        counts[label(e)] = counts.get(label(e), 0) + 1
    groups, used = [], set()
    for e in order:
        name = label(e)
        if counts[name] > 1:  # one name in several graphs (a site split per saggio): kept apart
            where = sorted(g for g in graphs.get(e.id, ()) if g)
            name = f"{name} ({', '.join(where) or e.id})"
            if name in used:
                name = f"{label(e)} ({e.id})"
        used.add(name)
        group = {"name": name, "units": members[e.id]}
        if isinstance(e.color, str) and _COLOUR.match(e.color.strip()):
            group["colour"] = e.color.strip()
        groups.append(group)
    return [{"kind": "unit_groups", "id": "epochs", "title": dict(TITLE), "groups": groups}]
