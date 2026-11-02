"""The solid of an architectural element with CadQuery, in a process of its own.

CadQuery brings OpenCascade (OCP, hundreds of MB of compiled libraries): the plugin never imports it in the program's
process, it runs this file with a Python that has CadQuery - the program's own when CadQuery is installed there, else
the environment the plugin installs - and only the files and the lines below go between the two.

    python -I -u worker.py JOB.json

This file imports nothing of the program or of the plugin. The job (every length in metres, every array small enough
to travel in the JSON)::

    {"protocol": 1, "kind": "column" | "moulding" | "arch", "output": "/abs/folder", "name": "...",
     "placement": [[4 x 4]],          # the element's frame -> the chunk's export frame (rigid)
     "tolerance": 0.0005,             # of the STL, metres
     "column":   {"profile": [[r, z], ...], "smooth": true},
     "moulding": {"path": [[x, y, z], ...], "section": [[u, v], ...]},
     "arch":     {"outline": {"points": [[u, v], ...], "segments": [["L", i, j], ["A", i, j, k], ...]},
                  "depth": 0.8}}

The frames: a column is the profile ``(r, z)`` revolved 360 degrees about the local z axis; a moulding is the section
swept along the path (given in the export frame: the placement is then the identity), the section in the plane normal
to the path at its first point with ``u`` = up x tangent and ``v`` = tangent x u, mitred corners; an arch is the
outline in the local xy plane extruded along z by
``depth``. ``output`` must not exist: the files (``solid.step``, AP214, and ``solid.stl``) are written in
``<output>.partial`` and the folder is renamed at the end, so a run stopped half-way never leaves a result that looks
complete.

Every message is one line on stdout, the prefix ``@ppg-architecture`` and a JSON object:

* ``{"type": "hello", "protocol": 1, "versions": {...}}``
* ``{"type": "progress", "fraction": 0.4, "phase": "solid" | "step" | "stl" | "check"}``
* ``{"type": "result", "result": {"output": ..., "files": {...}, "volume": ..., "area": ..., "bbox": [...],
  "valid": true, "seconds": ...}}``
* ``{"type": "error", "error": "...", "phase": "...", "exception": "...", "traceback": "..."}``

Other lines are OpenCascade's own: the plugin keeps the last ones for the report of a failure. The exit code is 0
with a result, 1 after an error.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
PREFIX = "@ppg-architecture "
UP = (0.0, 0.0, 1.0)


def emit(message_type: str, **data) -> None:
    """One message of the protocol, on a line of its own even after an unfinished line of OpenCascade."""
    sys.stdout.write("\n" + PREFIX + json.dumps({"type": message_type, **data}) + "\n")
    sys.stdout.flush()


def versions() -> dict:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for name in ("cadquery", "cadquery-ocp", "numpy"):
        try:
            out[name] = md.version(name)
        except Exception:  # noqa: BLE001
            out[name] = ""
    return out


def _norm(v):
    n = sum(c * c for c in v) ** 0.5
    if n < 1e-12:
        raise ValueError("a direction of zero length")
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def column(cq, spec: dict):
    """The profile ``(r, z)`` closed on the axis and revolved about local z."""
    pts = [(float(r), float(z)) for r, z in spec["profile"]]
    if len(pts) < 2:
        raise ValueError("a profile of two points at least")
    if any(r <= 0 for r, _z in pts):
        raise ValueError("a radius of the profile is zero or negative")
    z0, z1 = pts[0][1], pts[-1][1]
    wp = cq.Workplane("XZ").moveTo(0.0, z0).lineTo(*pts[0])
    if spec.get("smooth", True) and len(pts) > 2:
        wp = wp.spline(pts[1:], includeCurrent=True)
    else:
        wp = _polyline(wp, pts[1:])
    wp = wp.lineTo(0.0, z1).close()
    return wp.revolve(360.0, (0, 0, 0), (0, 1, 0)).val()


def _polyline(wp, pts):
    for p in pts:
        wp = wp.lineTo(*p)
    return wp


def moulding(cq, spec: dict):
    """The section swept along the path (an extrusion for a straight path)."""
    path = [tuple(map(float, p)) for p in spec["path"]]
    sec = []
    for u, v in spec["section"]:  # no edge of zero length: OpenCascade refuses them
        if not sec or abs(sec[-1][0] - float(u)) + abs(sec[-1][1] - float(v)) > 1e-9:
            sec.append((float(u), float(v)))
    while len(sec) > 3 and abs(sec[-1][0] - sec[0][0]) + abs(sec[-1][1] - sec[0][1]) <= 1e-9:
        sec.pop()
    if len(path) < 2 or len(sec) < 3:
        raise ValueError("a path of two points and a section of three at least")
    t0 = _norm(_sub(path[1], path[0]))
    up = UP if abs(sum(a * b for a, b in zip(t0, UP))) < 0.999 else (0.0, 1.0, 0.0)
    u0 = _norm(_cross(up, t0))
    plane = cq.Plane(origin=path[0], xDir=u0, normal=t0)
    profile = _polyline(cq.Workplane(plane).moveTo(*sec[0]), sec[1:]).close()
    if len(path) == 2:
        length = sum((a - b) ** 2 for a, b in zip(path[1], path[0])) ** 0.5
        return profile.extrude(length).val()
    wire = cq.Wire.makePolygon([cq.Vector(*p) for p in path], close=False)
    # mitred corners ("right": the faces extended to meet), the section carried by the corrected Frenet frame: with a
    # fixed binormal OpenCascade makes no valid solid at a mitre; the IFC's fixed-reference sweep mitres the same way
    return profile.sweep(cq.Workplane().add(wire), transition="right").val()


def arch(cq, spec: dict):
    """The outline (lines and three-point arcs) in local xy extruded along z by the depth."""
    outline = spec["outline"]
    pts = [(float(u), float(v)) for u, v in outline["points"]]
    segs = outline["segments"]
    depth = float(spec["depth"])
    if depth <= 0:
        raise ValueError("the depth of the arch must be positive")
    first = segs[0][1]
    wp = cq.Workplane("XY").moveTo(*pts[first])
    for seg in segs:
        if seg[0] == "L":
            if seg[2] == first and seg is segs[-1]:
                break
            wp = wp.lineTo(*pts[seg[2]])
        elif seg[0] == "A":
            wp = wp.threePointArc(pts[seg[2]], pts[seg[3]])
        else:
            raise ValueError(f"a segment of kind {seg[0]!r}")
    return wp.close().extrude(depth).val()


BUILDERS = {"column": column, "moulding": moulding, "arch": arch}


def place(cq, shape, matrix):
    """The solid moved by the 4 x 4 rigid ``matrix`` (rows). Through OpenCascade's ``gp_Trsf``: CadQuery's
    ``Matrix`` built from rows is a general transformation that ``transformShape`` refuses as non-orthogonal."""
    m = [[float(c) for c in row] for row in matrix]
    identity = all(abs(m[i][j] - (1.0 if i == j else 0.0)) < 1e-15 for i in range(4) for j in range(4))
    if identity:
        return shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    trsf = gp_Trsf()
    trsf.SetValues(*[m[i][j] for i in range(3) for j in range(4)])
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape())


def stub_casadi() -> bool:
    """CadQuery imports casadi (its assembly solver and a NURBS helper, neither used here): the plugin's
    environment does not install it (183 MB, LGPL-3). When it is missing a stub takes its place, whose every name
    fails if called. Returns whether the stub was put in."""
    import importlib.util
    import types

    if "casadi" in sys.modules or importlib.util.find_spec("casadi") is not None:
        return False

    class Missing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("casadi is not installed: CadQuery's assembly solver is not available here")

    module = types.ModuleType("casadi")
    module.__getattr__ = lambda name: Missing
    sys.modules["casadi"] = module
    return True


def run_job(job: dict) -> dict:
    t0 = time.monotonic()
    where = {"phase": "solid"}
    try:
        emit("progress", fraction=0.02, phase="solid")
        stub_casadi()
        import cadquery as cq

        emit("hello", protocol=PROTOCOL, versions=versions())
        output = Path(job["output"]).absolute()
        if output.exists():
            raise FileExistsError(f"the output folder exists already: {output}")
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        kind = job["kind"]
        if kind not in BUILDERS:
            raise ValueError(f"kind {kind!r}: column, moulding or arch")
        emit("progress", fraction=0.1, phase="solid")
        shape = BUILDERS[kind](cq, job[kind])
        shape = place(cq, shape, job.get("placement") or [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        where["phase"] = "check"
        emit("progress", fraction=0.5, phase="check")
        valid = bool(shape.isValid())
        volume = float(shape.Volume())
        if not valid or volume <= 0:
            raise ValueError(f"OpenCascade made a solid that is not valid (volume {volume:g} m3): "
                             "a self-intersecting section or path")
        bb = shape.BoundingBox()
        partial.mkdir(parents=True)
        where["phase"] = "step"
        emit("progress", fraction=0.6, phase="step")
        cq.exporters.export(shape, str(partial / "solid.step"), exportType="STEP")
        where["phase"] = "stl"
        emit("progress", fraction=0.8, phase="stl")
        tol = float(job.get("tolerance") or 0.0005)
        cq.exporters.export(shape, str(partial / "solid.stl"), exportType="STL", tolerance=tol, angularTolerance=0.05)
        os.replace(partial, output)  # complete, or not there at all
    except Exception as exc:
        try:
            exc.ppg_where = dict(where)
        except AttributeError:
            pass
        raise
    out = {"output": str(output), "files": {"step": "solid.step", "stl": "solid.stl"}, "volume": volume,
           "area": float(shape.Area()), "bbox": [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax], "valid": valid,
           "seconds": round(time.monotonic() - t0, 3)}
    emit("result", result=out)
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ValueError("usage: worker.py JOB.json")
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if int(job.get("protocol", PROTOCOL)) != PROTOCOL:
            raise ValueError(f"protocol {job.get('protocol')}: this worker speaks {PROTOCOL}")
        run_job(job)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the plugin as a message
        where = getattr(exc, "ppg_where", None) or {"phase": "start"}
        emit("error", error=str(exc).strip() or type(exc).__name__, exception=type(exc).__name__,
             phase=where["phase"], traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
