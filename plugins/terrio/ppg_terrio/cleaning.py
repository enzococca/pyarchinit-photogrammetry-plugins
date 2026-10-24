"""Cleaning the dense cloud with terrio's neighbourhood filters, which give PDAL's results:

* **isolated points** - the local outlier factor (LOF, PDAL's ``filters.lof``) of each point,
  the density of its neighbours over its own: about 1 on a surface, several times more for a
  point alone in the air or under the ground (a wrong depth, a reflection). Above the threshold
  the point goes to class 7 (low point, noise), which the program leaves out of DEMs and meshes;
* **detached groups** - clusters of DBSCAN (``filters.dbscan``) much smaller than the cloud and
  apart from every other point by a gap (:data:`GAP_FACTOR` DBSCAN radii, measured with the
  Euclidean clusters of ``filters.cluster``): birds, a drone, people or a pole standing apart,
  floating fragments of wrong depths. Removed, or given a class (18, high noise, by default).
  The gap keeps the sparse margins of a survey, which DBSCAN breaks into small groups at the
  radius of the dense part, attached to the rest.

:func:`analyse` finds them and counts them without changing anything; :func:`apply` changes the
cloud with an analysis (the counts can be read, and the points seen, before). The points labelled
by hand are never changed nor removed; the points already in the noise classes (7, 18) are left
out of the analysis. Before the change the cloud is kept as a version (*Versioni*: make it
active again to undo). The plugin's dimensions of the points go with them.

The distances follow the point spacing: the median distance from each point to its nearest one
(``nndistance``); the DBSCAN radius is :data:`EPS_FACTOR` times that, unless given. LOF is a
ratio of densities: it needs no distance.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from pyarchinit_photogrammetry.core.progress import as_progress

from . import cloudfile, environment
from .texts import num, t

NOISE_CLASSES = (7, 18)
ISOLATED_CLASS = 7  # low point (noise): the isolated points
DETACHED_CLASS = 18  # high noise: the detached groups, when they are given a class
LOF_NEIGHBOURS, LOF_THRESHOLD = 20, 3.0  # 2 marks the edges between dense and sparse parts of a real cloud
EPS_FACTOR, MIN_POINTS = 4.0, 6
GAP_FACTOR = 4.0  # a detached group is this many DBSCAN radii from any other point
MAX_SHARE = 1.0  # per cent of the cloud: a smaller group is detached
PREVIEW_POINTS = 3_000_000  # the other points of a preview at most (the 3D view shows 3 million)


def _feature() -> str:
    return t("La pulizia della nuvola (LOF, DBSCAN)", "Cleaning the cloud (LOF, DBSCAN)")


def spacing(cloud) -> float:
    """The median distance from each point of a terrio cloud to its nearest neighbour."""
    d = np.asarray(cloud.nndistance(k=1, mode="kth").to_dict()["nndistance"], np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else 0.0


def analyse(chunk, lof: bool = True, lof_neighbours: int = LOF_NEIGHBOURS, lof_threshold: float = LOF_THRESHOLD,
            dbscan: bool = True, eps: float | None = None, min_points: int = MIN_POINTS,
            max_share: float = MAX_SHARE, gap: float | None = None, progress=None) -> dict:
    """The isolated points (LOF above ``lof_threshold`` with ``lof_neighbours``) and the detached
    groups (DBSCAN clusters of radius ``eps`` - by default :data:`EPS_FACTOR` times the point
    spacing - and ``min_points``, holding less than ``max_share`` per cent of the points and
    ``gap`` metres - by default :data:`GAP_FACTOR` radii - from any other point; the largest group
    never) of the dense cloud. Nothing changes: the masks and counts for
    :func:`apply` and for the summary."""
    from pyarchinit_photogrammetry.core.pointcloud import NOISE_CLASSES as PROGRAM_NOISE

    terrio = environment.load_features(_feature())
    progress = as_progress(progress)
    if not (lof or dbscan):
        raise ValueError("choose LOF, DBSCAN or both")
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    n = len(pc)
    manual = pc.manual if pc.manual is not None else np.zeros(n, bool)
    used = np.nonzero(~np.isin(pc.classification, PROGRAM_NOISE))[0]
    isolated, detached = np.zeros(n, bool), np.zeros(n, bool)
    report = {"points": n, "used": int(len(used)), "spacing": 0.0, "eps": 0.0, "groups": [], "largest": 0,
              "params": {"lof": bool(lof), "lof_neighbours": int(lof_neighbours), "lof_threshold": float(lof_threshold),
                         "dbscan": bool(dbscan), "eps": eps, "min_points": int(min_points),
                         "max_share": float(max_share), "gap": gap}}
    if len(used) > max(int(lof_neighbours), int(min_points)):
        progress(0.05, t("{0} punti", "{0} points").format(num(len(used))))
        world = chunk.to_world(pc.xyz[used])
        cloud = terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2]})
        del world
        report["spacing"] = spacing(cloud)
        if lof:
            progress(0.15, t("Fattore di anomalia locale (LOF, {0} vicini)", "Local outlier factor (LOF, {0} neighbours)")
                     .format(int(lof_neighbours)))
            factor = np.asarray(cloud.lof(minpts=int(lof_neighbours)).to_dict()["local_outlier_factor"], np.float64)
            # a point among identical ones has no factor (0/0): never isolated
            isolated[used[np.isfinite(factor) & (factor > float(lof_threshold))]] = True
            report["lof_median"] = float(np.nanmedian(factor[np.isfinite(factor)])) if np.isfinite(factor).any() else 0.0
        if dbscan:
            e = float(eps) if eps else EPS_FACTOR * report["spacing"]
            if e <= 0:
                raise RuntimeError("the points have no spacing: give the DBSCAN radius")
            report["eps"] = e
            progress(0.55, t("Gruppi di punti (DBSCAN, raggio {0:.3g} m)", "Groups of points (DBSCAN, radius {0:.3g} m)")
                     .format(e))
            ids = np.asarray(cloud.dbscan(eps=e, min_points=int(min_points)).to_dict()["cluster_id"], np.int64)
            sizes = np.bincount(ids[ids >= 0]) if (ids >= 0).any() else np.zeros(0, np.int64)
            if len(sizes):
                largest = int(np.argmax(sizes))
                limit = float(max_share) / 100.0 * len(used)
                small = np.nonzero(sizes < limit)[0]
                small = small[small != largest]
                candidate = np.isin(ids, small)
                g = float(gap) if gap else GAP_FACTOR * e
                report["gap"] = g
                if candidate.any():
                    # the groups apart: their Euclidean cluster at the gap (DBSCAN's noise included) is small too
                    progress(0.8, t("Distacco dei gruppi (oltre {0:.3g} m)", "Gaps around the groups (over {0:.3g} m)")
                             .format(g))
                    coarse = np.asarray(cloud.cluster(tolerance=g, min_points=1).to_dict()["cluster_id"], np.int64)
                    csize = np.bincount(coarse)
                    apart = csize[coarse] < limit
                    apart &= coarse != int(np.argmax(csize))
                    candidate &= apart
                    small = np.unique(ids[candidate])
                detached[used[candidate]] = True
                report["groups"] = sorted((int(s) for s in np.bincount(ids[candidate])[small] if s), reverse=True) \
                    if candidate.any() else []
                report["largest"] = int(sizes[largest])
                report["clusters"] = int(len(sizes))
            report["dbscan_noise"] = int((ids < 0).sum())
        del cloud
    report["manual_kept"] = int(((isolated | detached) & manual).sum())
    isolated &= ~manual
    detached &= ~manual
    report.update({"isolated": isolated, "detached": detached, "n_isolated": int((isolated & ~detached).sum()),
                   "n_detached": int(detached.sum())})
    progress(1.0, summary(report))
    return report


def apply(chunk, analysis: dict, detached_action: str = "class", detached_class: int = DETACHED_CLASS,
          isolated_class: int = ISOLATED_CLASS, keep_version: bool = True, progress=None) -> dict:
    """Change the dense cloud with an :func:`analyse` of it: the isolated points to
    ``isolated_class`` (7), the detached groups removed (``detached_action="remove"``) or to
    ``detached_class`` (18). The cloud of before is kept as a version first (``keep_version``).
    Returns the counts."""
    from pyarchinit_photogrammetry.ai.classify import keep_manual_labels
    from pyarchinit_photogrammetry.core import versions

    progress = as_progress(progress)
    if detached_action not in ("class", "remove"):
        raise ValueError("detached_action: class or remove")
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    if len(pc) != analysis["points"]:
        raise RuntimeError("the dense cloud changed after the analysis: analyse it again")
    path = chunk.product_path("point_cloud")
    extras = cloudfile.read_extras(path)
    isolated, detached = analysis["isolated"], analysis["detached"]
    remove = detached if detached_action == "remove" else np.zeros(len(pc), bool)
    codes = pc.classification.copy()
    if detached_action == "class":
        codes[detached] = int(detached_class)
    codes[isolated & ~remove] = int(isolated_class)
    keep_manual_labels(pc, codes)
    changed = int((codes != pc.classification).sum())
    if remove.sum() > len(pc) - 10:
        raise RuntimeError("too few points would remain")
    kept = None
    if keep_version and (changed or remove.any()):
        progress(0.2, t("La nuvola di prima tra le versioni", "The cloud of before among the versions"))
        kept = versions.keep_active(chunk, "point_cloud", move=True)
    pc.classification = codes
    if remove.any():
        pc = pc.subset(~remove)
        extras = cloudfile.subset(extras, ~remove)
    progress(0.6, t("Scrittura della nuvola", "Writing the cloud"))
    cloudfile.write(chunk, pc, extras)
    if kept is not None:
        versions.set_active_name(chunk, "point_cloud", t("Nuvola pulita con terrio del {0}", "Cloud cleaned with terrio, {0}")
                                 .format(datetime.now().strftime("%d/%m %H:%M")))
    out = {"isolated": int((isolated & ~remove).sum()), "detached": int(detached.sum()),
           "groups": len(analysis.get("groups") or []), "removed": int(remove.sum()), "classified": changed,
           "points_before": int(analysis["points"]), "points_after": len(pc), "action": detached_action,
           "detached_class": int(detached_class), "isolated_class": int(isolated_class),
           "version": None if kept is None else kept["id"], "version_name": None if kept is None else kept["name"]}
    chunk.meta["terrio_clean"] = dict(analysis["params"]) | {k: v for k, v in out.items() if k != "version_name"} | {
        "spacing": analysis.get("spacing", 0.0), "eps_used": analysis.get("eps", 0.0), "gap_used": analysis.get("gap", 0.0)}
    progress(1.0, applied_summary(out))
    return out


def clean(chunk, detached_action: str = "class", detached_class: int = DETACHED_CLASS, dry_run: bool = False,
          progress=None, **options) -> tuple[dict, dict | None]:
    """:func:`analyse` and :func:`apply` in one: ``(analysis, applied)``, ``applied`` None with
    ``dry_run``."""
    progress = as_progress(progress)
    found = analyse(chunk, progress=progress.sub(0.0, 0.7), **options)
    if dry_run:
        return found, None
    return found, apply(chunk, found, detached_action, detached_class, progress=progress.sub(0.7, 0.3))


def preview(chunk, analysis: dict, path) -> str:
    """A cloud to look at before applying, written to ``path`` (internal frame, as the 3D view
    wants it): the isolated points red, the detached groups orange, the others grey - all the
    points found, the others thinned to :data:`PREVIEW_POINTS`."""
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    pc = chunk.point_cloud
    found = analysis["isolated"] | analysis["detached"]
    others = np.nonzero(~found)[0]
    if len(others) > PREVIEW_POINTS:
        others = np.sort(np.random.default_rng(0).choice(others, PREVIEW_POINTS, replace=False))
    keep = np.sort(np.concatenate([np.nonzero(found)[0], others]))
    rgb = np.full((len(pc), 3), 170, np.uint8)
    if pc.rgb is not None:  # the colours of the points, dimmed
        rgb = (0.35 * pc.rgb.astype(np.float32) + 0.65 * 170).astype(np.uint8)
    rgb[analysis["detached"]] = (255, 140, 0)
    rgb[analysis["isolated"] & ~analysis["detached"]] = (230, 20, 20)
    out = PointCloud(pc.xyz[keep], rgb[keep], classification=pc.classification[keep])
    return str(cloudfile.save(out, path))


def summary(analysis: dict) -> str:
    """The analysis, in words, in the language of the program."""
    p = analysis["params"]
    lines = [t("{0} punti, distanza media fra i punti {1:.3g} m", "{0} points, spacing between the points {1:.3g} m")
             .format(num(analysis["points"]), analysis.get("spacing", 0.0))]
    if p["lof"]:
        lines.append(t("  Punti isolati (LOF > {0:g}, {1} vicini): {2}", "  Isolated points (LOF > {0:g}, {1} neighbours): {2}")
                     .format(p["lof_threshold"], p["lof_neighbours"], num(analysis.get("n_isolated", 0))))
    if p["dbscan"]:
        groups = analysis.get("groups") or []
        lines.append(t("  Gruppi staccati (DBSCAN, raggio {0:.3g} m, sotto il {1:g} % della nuvola, a oltre {4:.3g} m "
                       "dal resto): {2} con {3} punti",
                       "  Detached groups (DBSCAN, radius {0:.3g} m, under {1:g} % of the cloud, over {4:.3g} m from "
                       "the rest): {2} with {3} points")
                     .format(analysis.get("eps", 0.0), p["max_share"], num(len(groups)), num(analysis.get("n_detached", 0)),
                             analysis.get("gap", 0.0)))
        if groups:
            lines.append(t("    i piu' grandi: {0}", "    the largest: {0}").format(", ".join(num(g) for g in groups[:5])))
    if analysis.get("manual_kept"):
        lines.append(t("  {0} punti classificati a mano lasciati come sono", "  {0} points labelled by hand left as they are")
                     .format(num(analysis["manual_kept"])))
    return "\n".join(lines)


def applied_summary(out: dict) -> str:
    lines = [t("{0} punti isolati in classe {1}", "{0} isolated points in class {1}").format(num(out["isolated"]),
                                                                                         out["isolated_class"])]
    if out["action"] == "remove":
        lines.append(t("{0} punti di {1} gruppi staccati rimossi", "{0} points of {1} detached groups removed")
                     .format(num(out["removed"]), num(out["groups"])))
    else:
        lines.append(t("{0} punti di {1} gruppi staccati in classe {2}", "{0} points of {1} detached groups in class {2}")
                     .format(num(out["detached"]), num(out["groups"]), out["detached_class"]))
    lines.append(t("Punti: {0} prima, {1} ora", "Points: {0} before, {1} now").format(num(out["points_before"]),
                                                                                 num(out["points_after"])))
    if out.get("version_name"):
        lines.append(t("La nuvola di prima e' fra le versioni: \"{0}\"", "The cloud of before is among the versions: \"{0}\"")
                     .format(out["version_name"]))
    return "\n".join(lines)
