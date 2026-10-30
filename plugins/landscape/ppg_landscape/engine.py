"""The simulation, run INSIDE the worker's process: a Landlab grid from the arrays of the DEM,
hillslope diffusion and fluvial erosion/deposition stepped through time, snapshots, the mass
balance.

Landlab (and its compiled helpers: richdem, netCDF) is imported only by :func:`run` and
:func:`versions`: the plugin imports this module for the pure helpers (:func:`default_cell`,
:func:`plan_targets`, the stability limits, :data:`DEFAULTS`, :data:`PRESETS`) without Landlab.
Nothing of the program or of the plugin is imported here (the worker loads this file by its path
in an environment that has only Landlab and numpy).

Arrays come and go in raster order: row 0 is the NORTH edge of the DEM (as rasterio reads it),
NaN where the DEM has no value. Landlab numbers its nodes from the south-west corner, so the grid
is the raster flipped upside down (:func:`to_nodes`, :func:`to_raster`).

The processes, both optional (at least one):

* hillslope diffusion, :class:`landlab.components.LinearDiffuser` (``D`` in m2/yr): soil creep, the
  flux proportional to the slope; split into sub-steps no longer than ``DIFFUSION_ALPHA dx^2 / D``
  (the explicit scheme's stability, Landlab's own factor is 0.15);
* running water, :class:`landlab.components.SpaceLargeScaleEroder` (SPACE: Stream Power with
  Alluvium Conservation and Entrainment, Shobe et al. 2017): erosion of a soil layer (``K_sed``)
  and of the material below (``K_br``), transport and deposition of the sediment (settling
  velocity ``v_s``, porosity ``phi``, fines lost as wash load ``F_f``, soil thickness for full
  entrainment ``H_star``), discharge = drainage area x a uniform runoff rate (m/yr). The flow is
  routed D8 by :class:`~landlab.components.PriorityFloodFlowRouter` (richdem, depressions filled)
  or, without richdem, by :class:`~landlab.components.FlowAccumulator` with
  :class:`~landlab.components.DepressionFinderAndRouter`. Without SpaceLargeScaleEroder:
  :class:`~landlab.components.Space`, then :class:`~landlab.components.ErosionDeposition` (both
  with their adaptive solver). Each step is no longer than a CFL-like limit from the fastest
  incision wave, ``FLUVIAL_CFL dx / (K Q^m n S^(n-1))``.

Before a run :func:`estimate` gives the sub-steps, an upper bound of the steps and the seconds, with a
warning above 10 minutes and a refusal above 2 hours.

Boundaries: cells without a value are closed. ``open``: the valid cells along the outer edge of
the data (the grid's edge or nodata reaching it) are fixed-value nodes - water and sediment leave
there; holes of nodata inside the DEM stay closed. ``closed``: those cells are closed too; with
running water the lowest of them beside a cell inside stays open as ONE artificial outlet (else every
drop would stay in the model as a lake); with diffusion only nothing leaves and the volume is conserved.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

DIFFUSION_ALPHA = 0.14  # diffusion sub-step <= ALPHA dx^2 / D (Landlab's LinearDiffuser uses 0.15)
FLUVIAL_CFL = 0.3  # fluvial step <= CFL dx / incision wave celerity
MAX_SIDE = 500  # default simulation grid: at most this many cells on the longer side
WARN_SIDE = 1000  # above this the plugin warns (memory, time)
STEPS = 100  # default largest step = years / STEPS (the bar moves, snapshots land on time)
EPS = 1e-9

# parameters of the processes and their defaults (SPACE's names; lengths in metres, time in years)
DEFAULTS = {
    "diffusion": {"D": 0.01},
    "fluvial": {"K_sed": 1e-3, "K_br": 2e-4, "F_f": 0.0, "phi": 0.3, "H_star": 0.5, "v_s": 1.0,
                "m_sp": 0.5, "n_sp": 1.0, "runoff": 0.3, "soil_depth": 0.5},
}
# scenarios with defensible values: D from the literature on soil-mantled slopes and earthworks
# (1e-3 - 2e-2 m2/yr), K low for a cohesive mudbrick mass; runoff of a semi-arid to Mediterranean climate
PRESETS = {
    "mudbrick_tell": {"years": 1000, "diffusion": {"D": 0.01},
                      "fluvial": {"K_sed": 5e-4, "K_br": 1e-4, "F_f": 0.5, "phi": 0.3, "H_star": 0.5, "v_s": 1.0,
                                  "m_sp": 0.5, "n_sp": 1.0, "runoff": 0.2, "soil_depth": 0.5}},
    "earth_slope": {"years": 1000, "diffusion": {"D": 0.005},
                    "fluvial": {"K_sed": 1e-3, "K_br": 2e-4, "F_f": 0.2, "phi": 0.3, "H_star": 0.5, "v_s": 1.0,
                                "m_sp": 0.5, "n_sp": 1.0, "runoff": 0.3, "soil_depth": 0.5}},
    "exposed_excavation": {"years": 10, "diffusion": {"D": 0.02},
                           "fluvial": {"K_sed": 2e-3, "K_br": 5e-4, "F_f": 0.3, "phi": 0.3, "H_star": 0.2,
                                       "v_s": 1.0, "m_sp": 0.5, "n_sp": 1.0, "runoff": 0.3, "soil_depth": 0.2}},
}
ROUTERS = ("auto", "priority_flood", "depression_finder")
ERODERS = ("auto", "space", "space_classic", "erosion_deposition")
BOUNDARIES = ("open", "closed")


class Stop(Exception):
    """Raised by a progress callback to stop the simulation (tests; the plugin kills the process)."""


# ---------------------------------------------------------------------- pure helpers
def default_cell(shape, resolution: float, max_side: int = MAX_SIDE) -> float:
    """The simulation cell: the DEM's ``resolution`` coarsened by a whole factor so the longer side
    has at most ``max_side`` cells."""
    longest = max(int(shape[0]), int(shape[1]))
    return float(resolution) * max(1, math.ceil(longest / max_side))


def diffusion_dt_limit(dx: float, D: float) -> float:
    """The longest stable step of explicit linear diffusion (years)."""
    return math.inf if D <= 0 else DIFFUSION_ALPHA * dx * dx / D


def fluvial_dt_limit(dx: float, K: float, discharge, slope, m: float, n: float, cfl: float = FLUVIAL_CFL,
                     phi: float = 0.0) -> float:
    """The longest step for stream-power incision: the celerity of a knickpoint is K Q^m n S^(n-1)
    (K Q^m for n = 1), divided by 1 - phi because SPACE thins the porous soil that much faster; a
    step moves it at most ``cfl`` cells."""
    q = np.asarray(discharge, float)
    if K <= 0 or q.size == 0:
        return math.inf
    c = K * np.power(np.maximum(q, 0.0), m) / max(1.0 - phi, 1e-6)
    if abs(n - 1.0) > 1e-12:
        s = np.maximum(np.asarray(slope, float), 1e-6)
        c = c * n * np.power(s, n - 1.0)
    top = float(np.nanmax(c)) if c.size else 0.0
    return math.inf if not top > 0 else cfl * dx / top


# seconds per cell measured on an Apple M4 Pro with Landlab 2.11 (500 x 500 cells): a diffusion sub-step
# ~0.15 s, a step of running water (routing + SPACE) ~1.2-1.8 s; the larger figure, scaled by the cells
DIFFUSION_SECONDS_PER_CELL = 0.15 / 250_000
FLUVIAL_SECONDS_PER_CELL = 1.8 / 250_000
WARN_SECONDS = 600  # the pane warns above 10 minutes
REFUSE_SECONDS = 7200  # and refuses above 2 hours


def estimate(shape, cell: float, years: float, params: dict, lang: str = "it") -> dict:
    """What a run will cost, before any routing: the diffusion sub-steps exactly, an upper bound of
    the steps of running water (the fastest knickpoint possible: all the grid's area draining to one
    cell, ``K (A runoff)^m``, with the slope bounded by ``params["relief"]`` (m) when n != 1), the
    seconds from the costs measured on an M4 Pro scaled by the cells; ``warning`` (a text in
    ``lang``, "it" or "en", or "") above :data:`WARN_SECONDS`, ``refuse`` above :data:`REFUSE_SECONDS`.
    ``params``: the job's ``diffusion``, ``fluvial``, ``dt`` and optionally ``relief``."""
    rows, cols = int(shape[0]), int(shape[1])
    cells = rows * cols
    years = float(years)
    diffusion, fluvial = merged(params)
    dt_max = _largest_step(params.get("dt"), years)
    outer = math.ceil(years / dt_max - 1e-9)
    fluvial_steps = 0
    if fluvial is not None:
        q = cells * cell * cell * fluvial["runoff"]
        relief = float(params.get("relief") or 0.0)
        n = fluvial["n_sp"]
        if n > 1:
            slope = min(max(relief / cell, 1e-3), 10.0) if relief > 0 else 10.0
        else:
            slope = max(relief / (max(rows, cols) * cell), 1e-3) if relief > 0 else 1e-3
        limit = fluvial_dt_limit(cell, max(fluvial["K_sed"], fluvial["K_br"]), [q], [slope], fluvial["m_sp"], n,
                                 phi=fluvial["phi"])
        fluvial_steps = max(outer, math.ceil(years / min(limit, dt_max) - 1e-9))
    steps = fluvial_steps or outer
    substeps = 0
    if diffusion is not None:
        substeps = max(steps, math.ceil(years / diffusion_dt_limit(cell, diffusion["D"]) - 1e-9))
    seconds = cells * (fluvial_steps * FLUVIAL_SECONDS_PER_CELL + substeps * DIFFUSION_SECONDS_PER_CELL)
    warning = ""
    if seconds > WARN_SECONDS:
        hours = seconds / 3600
        when = (f"{hours:.1f} ore" if lang == "it" else f"{hours:.1f} hours") if hours >= 1 else \
            (f"{seconds / 60:.0f} minuti" if lang == "it" else f"{seconds / 60:.0f} minutes")
        when = when.replace(".", ",") if lang == "it" else when
        warning = (f"La simulazione può durare circa {when}: una cella più grande o meno anni la accorciano."
                   if lang == "it" else
                   f"The simulation may take about {when}: a larger cell or fewer years shorten it.")
    return {"cells": cells, "steps": steps, "fluvial_steps": fluvial_steps, "diffusion_substeps": substeps,
            "seconds": seconds, "warning": warning, "refuse": seconds > REFUSE_SECONDS}


def _largest_step(dt, years: float) -> float:
    """The job's ``dt`` (years) checked, or the default ``years / STEPS``."""
    if dt is None or dt == 0:
        return years / STEPS
    dt = float(dt)
    if not (dt > 0 and math.isfinite(dt)):
        raise ValueError(f"dt must be a positive number of years, not {dt}")
    return dt


def plan_targets(years: float, snapshots=()) -> list[float]:
    """The times the stepping must land on exactly: the snapshots inside (0, years] and the end."""
    years = float(years)
    if not years > 0:
        raise ValueError(f"years must be positive, not {years}")
    times = {float(s) for s in snapshots or () if 0 < float(s) < years}
    return sorted(times) + [years]


def snapshot_name(year: float) -> str:
    """``snap_10.npy``, ``snap_2.5.npy``."""
    return f"snap_{float(year):g}.npy"


def perimeter(valid: np.ndarray) -> np.ndarray:
    """The valid cells on the outer edge of the data: on the grid's edge or beside a nodata cell
    connected to the grid's edge (holes of nodata inside the DEM are not an edge)."""
    valid = np.asarray(valid, bool)
    outside = ~valid
    if outside.any():
        from scipy import ndimage

        labels, _n = ndimage.label(outside)
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        outside = np.isin(labels, edge[edge > 0])
    pad = np.pad(outside, 1, constant_values=True)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    return valid & near


def to_nodes(raster: np.ndarray) -> np.ndarray:
    """Raster order (row 0 north) -> Landlab's node order (row 0 south)."""
    return np.ascontiguousarray(np.flipud(raster)).ravel()


def to_raster(nodes: np.ndarray, shape) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(np.asarray(nodes).reshape(shape)))


def merged(job: dict) -> tuple[dict | None, dict | None]:
    """The parameters of the two processes with the defaults filled in; None for a process off."""
    out = []
    for name in ("diffusion", "fluvial"):
        given = job.get(name)
        if given is None or given is False:
            out.append(None)
            continue
        params = dict(DEFAULTS[name])
        params.update({k: float(v) for k, v in dict(given if isinstance(given, dict) else {}).items()})
        out.append(params)
    diffusion, fluvial = out
    if diffusion is not None and not diffusion["D"] > 0:
        diffusion = None
    if diffusion is None and fluvial is None:
        raise ValueError("no process: diffusion and running water are both off")
    return diffusion, fluvial


# ---------------------------------------------------------------------- the simulation
@dataclass
class Result:
    initial: np.ndarray  # raster order, float64, NaN outside the data
    final: np.ndarray
    area: np.ndarray | None  # drainage area (m2) of the final surface
    snapshots: dict = field(default_factory=dict)  # year -> elevation
    balance: dict = field(default_factory=dict)

    @property
    def dz(self) -> np.ndarray:
        return self.final - self.initial


def versions() -> dict:
    """The versions of Landlab and of what it runs with (for the hello and the check)."""
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for name, dist in (("landlab", "landlab"), ("numpy", "numpy"), ("richdem", "py-richdem")):
        try:
            out[name] = md.version(dist)
        except Exception:  # noqa: BLE001
            out[name] = ""
    return out


def outlets_on_the_frame(grid) -> bool:
    """Whether every open (fixed-value) node lies on the outer frame of the grid. PriorityFloodFlowRouter
    fills the depressions with richdem, which floods from the cells of the array's frame only and gives the
    closed nodes an infinite height: an open node inside the array - the edge of a DEM whose data does not
    reach the frame, surrounded by nodata, or the one outlet of a closed edge - is never an outlet there, the
    whole surface is "filled" to infinity and every cell drains to itself."""
    from landlab.grid.nodestatus import NodeStatus

    fixed = np.flatnonzero(np.asarray(grid.status_at_node) == NodeStatus.FIXED_VALUE)
    return bool(np.isin(fixed, np.asarray(grid.perimeter_nodes)).all())


def _router(grid, kind: str, runoff: float):
    """The flow router and its name: PriorityFlood (richdem) when every outlet lies on the frame of the
    grid (:func:`outlets_on_the_frame`), else FlowAccumulator + DepressionFinder, which routes to any
    open node."""
    from landlab.components import FlowAccumulator

    if kind == "priority_flood" and not outlets_on_the_frame(grid):
        raise ValueError("router priority_flood: the edge of the data does not lie on the frame of the grid "
                         "(nodata around the DEM, or a closed edge); PriorityFlood would see no outlet - use "
                         "depression_finder or auto")
    if kind == "priority_flood" or (kind == "auto" and outlets_on_the_frame(grid)):
        try:
            from landlab.components import PriorityFloodFlowRouter

            return PriorityFloodFlowRouter(grid, flow_metric="D8", runoff_rate=runoff, depression_handler="fill",
                                           suppress_out=True), "priority_flood"
        except ImportError:  # richdem missing: PriorityFlood cannot run
            if kind == "priority_flood":
                raise
    return FlowAccumulator(grid, flow_director="D8", depression_finder="DepressionFinderAndRouter",
                           runoff_rate=runoff), "depression_finder"


def _eroder(grid, kind: str, p: dict):
    """The fluvial eroder, its name and whether it keeps soil and bedrock itself (SPACE does)."""
    import landlab.components as lc

    order = ("space", "space_classic", "erosion_deposition") if kind == "auto" else (kind,)
    for name in order:
        if name == "space" and hasattr(lc, "SpaceLargeScaleEroder"):
            return lc.SpaceLargeScaleEroder(grid, K_sed=p["K_sed"], K_br=p["K_br"], F_f=p["F_f"], phi=p["phi"],
                                            H_star=p["H_star"], v_s=p["v_s"], m_sp=p["m_sp"],
                                            n_sp=p["n_sp"]), name, True
        if name == "space_classic" and hasattr(lc, "Space"):
            return lc.Space(grid, K_sed=p["K_sed"], K_br=p["K_br"], F_f=p["F_f"], phi=p["phi"], H_star=p["H_star"],
                            v_s=p["v_s"], m_sp=p["m_sp"], n_sp=p["n_sp"], solver="adaptive"), name, True
        if name == "erosion_deposition" and hasattr(lc, "ErosionDeposition"):
            return lc.ErosionDeposition(grid, K=p["K_sed"], v_s=p["v_s"], m_sp=p["m_sp"], n_sp=p["n_sp"], F_f=p["F_f"],
                                        solver="adaptive"), name, False
    raise ValueError(f"no fluvial eroder {kind!r} in this Landlab")


def _to_soil(H: np.ndarray, br: np.ndarray, dz: np.ndarray, nodes: np.ndarray) -> None:
    """A change of the surface made by another process given to the soil layer: deposits thicken
    it, erosion thins it and then lowers the material below (surface = bedrock + soil stays true).
    Bulk volume for bulk volume: when creep digs into the bedrock, that solid material becomes porous
    soil and the solid volume shrinks by phi times it - the balance measures it (the diffusion's
    solid out), it does not hide it."""
    H[nodes] += dz[nodes]
    thin = nodes[H[nodes] < 0]
    br[thin] += H[thin]
    H[thin] = 0.0


def run(job: dict, elevation: np.ndarray, valid: np.ndarray | None = None, progress=None) -> Result:
    """Simulate ``job`` on ``elevation`` (raster order, NaN = no value; ``valid``: another mask of
    the cells to use). ``progress(fraction, phase, **info)`` is called on the way (phases ``grid``,
    ``run``, ``area``); it may raise :class:`Stop`. The job's keys::

        cell (m), years, dt (largest step, years; default years / 100), snapshots ([years]),
        boundary ("open" | "closed"), diffusion ({"D"} | null), fluvial ({K_sed, K_br, F_f, phi,
        H_star, v_s, m_sp, n_sp, runoff, soil_depth} | null), router, eroder

    The balance (``Result.balance``, volumes in m3):

    * ``eroded_m3``, ``deposited_m3``, ``net_m3``: BULK volumes, from dz - what the map of dz shows,
      pores included;
    * ``eroded_solid_m3``, ``deposited_solid_m3``: SOLID volumes, per core cell (1 - phi) dH + d bedrock
      (the soil is porous, the material below is not); equal to the bulk ones with diffusion alone;
    * ``bedload_out_m3``, ``wash_load_m3`` (``fluvial_out_m3`` their sum): solid sediment leaving with
      the water (SPACE only; None with the other eroders), the wash load being the fines F_f of the
      eroded bedrock;
    * ``diffusion_out_m3`` (bulk) and ``diffusion_out_solid_m3``: what creep took out of the model
      across open edges, the solid one including the pores gained when creep turns bedrock into soil;
    * ``closure_m3`` = eroded solid - deposited solid - everything out, ``closure_percent`` of the
      eroded solid: ~0 when the books close (None when the eroder does not report its outflow).
    """
    say = progress or (lambda *a, **k: None)
    t0 = time.monotonic()
    z_raster = np.asarray(elevation, dtype=np.float64)
    if z_raster.ndim != 2 or min(z_raster.shape) < 3:
        raise ValueError(f"the elevation must be a grid of at least 3 x 3 cells, not {z_raster.shape}")
    ok = np.isfinite(z_raster)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if ok.sum() < 9:
        raise ValueError("fewer than 9 cells with a value: nothing to simulate")
    dx = float(job["cell"])
    if not dx > 0:
        raise ValueError(f"cell must be positive, not {dx}")
    years = float(job["years"])
    targets = plan_targets(years, job.get("snapshots") or ())
    wanted = {float(s) for s in job.get("snapshots") or () if 0 < float(s) <= years}
    dt_max = _largest_step(job.get("dt"), years)
    boundary = str(job.get("boundary", "open"))
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary {boundary!r}: {' or '.join(BOUNDARIES)}")
    router_kind = str(job.get("router", "auto"))
    if router_kind not in ROUTERS:
        raise ValueError(f"router {router_kind!r}: {', '.join(ROUTERS)}")
    diffusion, fluvial = merged(job)
    shape = z_raster.shape
    from landlab import RasterModelGrid  # after the checks: a bad job fails without Landlab
    from landlab.components import LinearDiffuser
    from landlab.grid.nodestatus import NodeStatus

    say(0.0, "grid")

    # the grid: closed where there is no value, the edge of the data open or closed
    grid = RasterModelGrid(shape, xy_spacing=dx)
    edge = perimeter(ok)
    status = np.full(ok.size, NodeStatus.CORE, dtype=np.uint8)
    status[to_nodes(~ok)] = NodeStatus.CLOSED
    status[to_nodes(edge)] = NodeStatus.FIXED_VALUE if boundary == "open" else NodeStatus.CLOSED
    filled = np.where(ok, z_raster, np.nanmax(np.where(ok, z_raster, np.nan)))
    z = grid.add_field("topographic__elevation", to_nodes(filled).astype(np.float64), at="node")
    outlet = None
    if boundary == "closed" and fluvial is not None:
        # running water needs a way out: one artificial outlet, the lowest edge cell beside a cell inside
        # (a corner touches only other edge cells: water would reach it diagonally, creep never)
        inside = to_nodes(ok & ~edge)
        on_edge = np.flatnonzero(to_nodes(edge))
        beside = [i for i in on_edge if any(inside[j] for j in grid.adjacent_nodes_at_node[i] if j >= 0)]
        candidates = np.asarray(beside or on_edge)
        outlet = int(candidates[np.argmin(z[candidates])])
        status[outlet] = NodeStatus.FIXED_VALUE
    grid.status_at_node = status  # all at once: perimeter nodes must never be core, not even for a moment
    core = grid.core_nodes
    if core.size == 0:
        raise ValueError("no cell inside the edge of the data: the grid is too small")
    cell_area = dx * dx
    z_start = z.copy()

    ld = LinearDiffuser(grid, linear_diffusivity=diffusion["D"], deposit=True) if diffusion else None
    d_limit = diffusion_dt_limit(dx, diffusion["D"]) if diffusion else math.inf
    router = eroder = None
    router_name = eroder_name = ""
    keeps_soil = False
    if fluvial is not None:
        H = grid.add_field("soil__depth", np.full(grid.number_of_nodes, max(fluvial["soil_depth"], 0.0)), at="node")
        br = grid.add_field("bedrock__elevation", z - H, at="node")
        router, router_name = _router(grid, router_kind, fluvial["runoff"])
        router.run_one_step()  # the fields the eroder asks for
        eroder, eroder_name, keeps_soil = _eroder(grid, str(job.get("eroder", "auto")), fluvial)
        K = max(fluvial["K_sed"], fluvial["K_br"])
        phi = fluvial["phi"]
        H_start, br_start = H[core].copy(), br[core].copy()

    t = 0.0
    steps = diffusion_substeps = 0
    outlet_m3 = 0.0 if eroder_name == "space" else None
    wash_m3 = 0.0 if eroder_name == "space" else None
    diffusion_out = diffusion_out_solid = 0.0
    smallest = math.inf
    snapshots: dict[float, np.ndarray] = {}
    last_say = [0.0]

    def tell(now_t: float) -> None:  # the bar moves at most every half second, also inside the sub-steps
        now = time.monotonic()
        if now - last_say[0] >= 0.5 or now_t >= years:
            last_say[0] = now
            say(0.02 + 0.93 * now_t / years, "run", year=round(now_t, 6), years=years, step=steps)
    for target in targets:
        while t < target - EPS * max(1.0, target):
            step = min(target - t, dt_max)
            if fluvial is not None:
                if steps:
                    router.run_one_step()
                f_limit = fluvial_dt_limit(dx, K, grid.at_node["surface_water__discharge"][core],
                                           grid.at_node["topographic__steepest_slope"][core], fluvial["m_sp"],
                                           fluvial["n_sp"], phi=phi)
                step = min(step, f_limit)
                before = z.copy()
                out = eroder.run_one_step(step)
                if eroder_name == "space" and isinstance(out, tuple) and len(out) == 2:
                    wash_m3 += float(out[0]) * step  # vol_SSY_riv is a rate (m3/yr), V_leaving_riv a volume
                    outlet_m3 += float(out[1])
                if not keeps_soil:
                    _to_soil(H, br, z - before, core)
            if ld is not None:
                # LinearDiffuser splits a long step by itself too (at 0.15 dx^2 / D); our own split keeps the
                # soil bookkeeping and the progress bar going sub-step by sub-step
                n = max(1, math.ceil(step / d_limit - 1e-9))
                sub = step / n
                for k in range(n):
                    before = z[core].copy()
                    ld.run_one_step(sub)
                    change = z[core] - before
                    diffusion_out -= float(change.sum()) * cell_area
                    if fluvial is not None:
                        solid_before = (1 - phi) * H[core] + br[core]
                        dzn = np.zeros_like(z)
                        dzn[core] = change
                        _to_soil(H, br, dzn, core)
                        diffusion_out_solid -= float(((1 - phi) * H[core] + br[core] - solid_before).sum()) * cell_area
                    if n > 1:
                        tell(t + sub * (k + 1))
                diffusion_substeps += n
            smallest = min(smallest, step)
            steps += 1
            t = target if target - (t + step) <= EPS * max(1.0, target) else t + step
            tell(t)
            if not np.isfinite(z[core]).all():
                raise FloatingPointError(f"the surface became NaN after {t:g} years: a shorter step (dt) is needed")
        if target in wanted:
            snapshots[target] = np.where(ok, to_raster(z, shape), np.nan)

    # drainage area of the final surface
    say(0.96, "area")
    if router is None:
        router, router_name = _router(grid, router_kind, 1.0)
    router.run_one_step()
    area = np.where(ok, to_raster(grid.at_node["drainage_area"], shape), np.nan)

    initial = np.where(ok, to_raster(z_start, shape), np.nan)
    final = np.where(ok, to_raster(z, shape), np.nan)
    dz = final - initial
    d = dz[ok]
    from_flow = None if outlet_m3 is None else outlet_m3 + wash_m3
    # solid volumes: without pores (soil counts 1 - phi of its thickness); with diffusion alone no soil
    # is modelled and solid = bulk
    if fluvial is not None:
        solid = ((1 - phi) * (H[core] - H_start) + (br[core] - br_start)) * cell_area
        eroded_solid, deposited_solid = float(-solid[solid < 0].sum()), float(solid[solid > 0].sum())
    else:
        eroded_solid, deposited_solid = float(-d[d < 0].sum() * cell_area), float(d[d > 0].sum() * cell_area)
        diffusion_out_solid = diffusion_out
    closure = None
    if fluvial is None or from_flow is not None:  # the eroders other than SPACE do not say what leaves
        closure = eroded_solid - deposited_solid - (from_flow or 0.0) - (diffusion_out_solid if ld is not None else 0.0)
    balance = {
        "years": years, "cell": dx, "cell_area_m2": cell_area, "cells": int(ok.sum()), "core_cells": int(core.size),
        "boundary": boundary, "outlet_node": outlet, "steps": steps, "smallest_step": smallest if steps else None,
        "diffusion_substeps": diffusion_substeps, "diffusion_dt_limit": None if ld is None else d_limit,
        "eroded_m3": float(-d[d < 0].sum() * cell_area), "deposited_m3": float(d[d > 0].sum() * cell_area),
        "net_m3": float(d.sum() * cell_area), "initial_volume_m3": None,
        "max_erosion_m": float(max(0.0, -d.min())), "max_deposition_m": float(max(0.0, d.max())),
        "eroded_solid_m3": eroded_solid, "deposited_solid_m3": deposited_solid,
        "bedload_out_m3": outlet_m3, "wash_load_m3": wash_m3, "fluvial_out_m3": from_flow,
        "diffusion_out_m3": diffusion_out if ld is not None else None,
        "diffusion_out_solid_m3": diffusion_out_solid if ld is not None else None,
        "closure_m3": closure,
        "closure_percent": None if closure is None else (100.0 * closure / eroded_solid if eroded_solid > 0 else 0.0),
        "router": router_name, "eroder": eroder_name or None, "diffusion": diffusion, "fluvial": fluvial,
        "seconds": round(time.monotonic() - t0, 3),
    }
    # volume above the lowest cell: the scale against which the net change reads
    balance["initial_volume_m3"] = float((initial[ok] - np.nanmin(initial)).sum() * cell_area)
    say(1.0, "done")
    return Result(initial, final, area, snapshots, balance)
