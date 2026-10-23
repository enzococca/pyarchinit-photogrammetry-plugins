"""A 3D Gaussian splat (the PLY of 3D Gaussian Splatting that Spirula writes) moved into another
frame: the survey coordinates of the chunk, from the frame Spirula trained it in.

Spirula recentres and reorients the scene before training; ``scene_transform.json`` of the run
gives ``world_from_train``, the 4 x 4 similarity that brings the splats back into the frame of the
COLMAP model - the internal frame of the chunk - and ``chunk.transform`` brings that into the
world (the CRS of the chunk, or its local frame). A similarity ``x' = s R x + t`` changes, for
every splat:

* the position ``x, y, z``: moved;
* the rotation ``rot_0..rot_3`` (a quaternion, ``w`` first, not necessarily of unit length):
  turned by ``R`` (its length kept);
* the log-scales ``scale_*``: ``+ log(s)``;
* the normals ``nx, ny, nz``, when there: turned;
* the colour that changes with the point of view - the spherical harmonics ``f_rest_*`` (channel
  by channel, degrees 1 to 3 as many as the file holds): turned, so that the colour seen along the
  direction ``R d`` in the new frame is the one seen along ``d`` before. Degree by degree the
  harmonics of a rotated direction are a linear mix of the harmonics: the mix is found by least
  squares on directions spread over the sphere (exact up to rounding). ``f_dc_*`` (the colour of
  degree 0) and the opacity do not change.

Positions in projected coordinates (millions of metres) do not fit a 32-bit float (25 cm steps at
4 000 km): the positions are written as doubles when a coordinate goes beyond 10 km.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

DOUBLE_BEYOND = 10_000.0  # m: beyond this, 32-bit positions lose the millimetre
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_PLY_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint", "f4": "float",
              "f8": "double"}
# the real spherical harmonics as 3D Gaussian Splatting evaluates them (eval_sh of its reference code)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


# ---------------------------------------------------------------------------- PLY
def read_ply(path) -> tuple[list[str], list[tuple[str, np.ndarray]]]:
    """``(comments, [(element, structured array)])`` of a binary PLY whose properties are numbers
    (the PLY of a splat; list properties, as the faces of a mesh, are refused)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{Path(path).name}: not a PLY (no end_header)")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break
        data = f.read()
    if not header or header[0] != "ply":
        raise ValueError(f"{Path(path).name}: not a PLY")
    fmt, comments, elements = None, [], []
    for text in header[1:-1]:
        parts = text.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] in ("comment", "obj_info"):
            comments.append(text)
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if parts[1] == "list" or parts[1] not in _PLY_TYPES or not elements:
                raise ValueError(f"{Path(path).name}: property {' '.join(parts[1:])} is not one of a splat")
            elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{Path(path).name}: a {fmt} PLY; a splat is a binary PLY")
    order = "<" if fmt == "binary_little_endian" else ">"
    out, offset = [], 0
    for name, count, props in elements:
        dtype = np.dtype([(p, order + kind) for p, kind in props])
        size = dtype.itemsize * count
        if offset + size > len(data):
            raise ValueError(f"{Path(path).name}: the file ends before its {name} element")
        arr = np.frombuffer(data[offset:offset + size], dtype, count=count).copy() if size else np.zeros(count, dtype)
        offset += size
        out.append((name, arr))
    return comments, out


def write_ply(path, comments, elements) -> Path:
    """A binary little-endian PLY of ``elements`` (``[(name, structured array)]``)."""
    lines = ["ply", "format binary_little_endian 1.0", *comments]
    for name, arr in elements:
        lines.append(f"element {name} {len(arr)}")
        for field in arr.dtype.names or ():
            lines.append(f"property {_PLY_NAMES[arr.dtype[field].str[1:]]} {field}")
    lines.append("end_header")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        for _name, arr in elements:
            f.write(np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"))).tobytes())
    return path


# ---------------------------------------------------------------------------- the similarity
def similarity(matrix) -> tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` of a 4 x 4 similarity ``x' = s R x + t``; ValueError when the matrix also
    stretches, shears or mirrors (a splat cannot follow that)."""
    m = np.asarray(matrix, np.float64).reshape(4, 4)
    a = m[:3, :3]
    det = float(np.linalg.det(a))
    if not det > 0:
        raise ValueError("the transform mirrors or flattens the scene: not a similarity")
    s = det ** (1.0 / 3.0)
    u, _sv, vt = np.linalg.svd(a / s)
    rot = u @ vt
    if np.abs(a / s - rot).max() > 1e-3:
        raise ValueError("the transform stretches or shears the scene: not a similarity")
    return s, rot, m[:3, 3].copy()


def quaternion(rot) -> np.ndarray:
    """The unit quaternion ``(w, x, y, z)`` of a rotation matrix."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(rot, np.float64)).as_quat()
    return np.array([w, x, y, z])


def quaternion_matrix(q) -> np.ndarray:
    """The rotation matrices of quaternions ``(..., 4)`` (``w`` first, any length)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def multiply(a, b) -> np.ndarray:
    """The quaternion product ``a b`` (``w`` first): the rotation of ``b`` followed by that of ``a``."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


# ---------------------------------------------------------------------------- spherical harmonics
def sh_basis(dirs, degree: int) -> np.ndarray:
    """``(N, (degree + 1)^2)``: the spherical harmonics of unit directions ``(N, 3)`` in the order
    and with the signs 3D Gaussian Splatting gives them (a colour is ``basis @ coefficients + 0.5``)."""
    d = np.asarray(dirs, np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = [np.full_like(x, C0)]
    if degree >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree >= 3:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
                 C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, axis=1)


def _sphere(n: int = 256) -> np.ndarray:
    """``n`` directions spread evenly over the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def sh_rotations(rot, degree: int) -> list[np.ndarray]:
    """For every degree ``l`` from 1 to ``degree``, the matrix ``M_l`` that turns the coefficients
    of that degree: the colour of ``M_l c`` along ``R d`` is the colour of ``c`` along ``d``."""
    dirs = _sphere()
    basis = sh_basis(dirs, degree)
    turned = sh_basis(dirs @ np.asarray(rot, np.float64), degree)  # the harmonics of R^T d
    out = []
    for lv in range(1, degree + 1):
        cols = slice(lv * lv, (lv + 1) * (lv + 1))
        m, *_ = np.linalg.lstsq(basis[:, cols], turned[:, cols], rcond=None)
        out.append(m)
    return out


def sh_degree(rest_count: int) -> int:
    """The degree of the harmonics of a splat with ``rest_count`` properties ``f_rest_*``."""
    if rest_count % 3:
        raise ValueError(f"{rest_count} coefficients f_rest_*: not three colours")
    per_colour = rest_count // 3
    degree = int(round(np.sqrt(per_colour + 1))) - 1
    if (degree + 1) ** 2 - 1 != per_colour:
        raise ValueError(f"{per_colour} coefficients per colour: not a degree of spherical harmonics")
    return degree


def rotate_sh(rest, rot) -> np.ndarray:
    """The coefficients ``f_rest_*`` of the splats (``(N, 3 K)``, colour by colour as the PLY keeps
    them: red ``0..K-1``, green, blue) for the frame turned by ``rot``."""
    rest = np.asarray(rest, np.float64)
    if rest.shape[1] == 0:
        return rest.copy()
    degree = sh_degree(rest.shape[1])
    k = rest.shape[1] // 3
    out = rest.reshape(len(rest), 3, k).copy()
    for lv, m in enumerate(sh_rotations(rot, degree), start=1):
        lo, hi = lv * lv - 1, (lv + 1) * (lv + 1) - 1
        out[:, :, lo:hi] = out[:, :, lo:hi] @ m.T
    return out.reshape(len(rest), 3 * k)


# ---------------------------------------------------------------------------- the splat
def _numbered(names, prefix: str) -> list[str]:
    return sorted((n for n in names if n.startswith(prefix) and n[len(prefix):].isdigit()),
                  key=lambda n: int(n[len(prefix):]))


def transform_vertices(arr: np.ndarray, matrix) -> np.ndarray:
    """The splats of a PLY vertex element (a structured array) in the frame of ``matrix``."""
    s, rot, shift = similarity(matrix)
    names = arr.dtype.names or ()
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("the splat has no positions (x, y, z)")
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
    moved = s * xyz @ rot.T + shift
    wide = bool(len(moved)) and float(np.abs(moved).max()) >= DOUBLE_BEYOND
    dtype = np.dtype([(n, "<f8" if (wide and n in ("x", "y", "z")) else arr.dtype[n].newbyteorder("<"))
                      for n in names])
    out = np.empty(len(arr), dtype)
    for n in names:
        out[n] = arr[n]
    out["x"], out["y"], out["z"] = moved.T
    if {"nx", "ny", "nz"} <= set(names):
        normals = np.column_stack([arr["nx"], arr["ny"], arr["nz"]]).astype(np.float64) @ rot.T
        out["nx"], out["ny"], out["nz"] = normals.T
    rots = [f"rot_{i}" for i in range(4)]
    if set(rots) <= set(names):
        q = multiply(quaternion(rot), np.column_stack([arr[r] for r in rots]))
        for i, r in enumerate(rots):
            out[r] = q[:, i]
    for n in _numbered(names, "scale_"):
        out[n] = arr[n].astype(np.float64) + np.log(s)
    rest = _numbered(names, "f_rest_")
    if rest:
        turned = rotate_sh(np.column_stack([arr[n] for n in rest]), rot)
        for i, n in enumerate(rest):
            out[n] = turned[:, i]
    return out


def transform_splat(src, dst, matrix, comments=()) -> dict:
    """The splat PLY ``src`` written into ``dst`` in the frame of the 4 x 4 similarity ``matrix``;
    other elements of the file are copied. Returns ``{"path", "splats", "double"}``."""
    head, elements = read_ply(src)
    out, splats, double = [], 0, False
    for name, arr in elements:
        if name == "vertex" and len(arr):
            arr = transform_vertices(arr, matrix)
            splats, double = len(arr), arr.dtype["x"] == np.float64
        out.append((name, arr))
    path = write_ply(dst, [*head, *comments], out)
    return {"path": path, "splats": int(splats), "double": bool(double)}


# ---------------------------------------------------------------------------- the splat of a chunk
def train_frame(chunk) -> np.ndarray | None:
    """``world_from_train`` of the splat of the chunk: from the project, else from the
    ``scene_transform.json`` of its run; None when Spirula wrote none."""
    from .runner import scene_transform

    info = chunk.meta.get("splat") or {}
    m = info.get("world_from_train")
    if m is None and info.get("run"):
        m = scene_transform(chunk.data_dir / info["run"])
    return None if m is None else np.asarray(m, np.float64).reshape(4, 4)


def survey_matrix(chunk) -> np.ndarray:
    """The similarity from the frame Spirula trained the splat in to the survey coordinates of the
    chunk: ``chunk.transform @ world_from_train``."""
    from . import t

    m = train_frame(chunk)
    if m is None:
        raise RuntimeError(t("Spirula non ha scritto il sistema dello splat (scene_transform.json del run): lo splat "
                             "si puo' esportare solo nel sistema dell'addestramento.",
                             "Spirula wrote no frame for the splat (scene_transform.json of the run): the splat can "
                             "only be exported in the frame of the training."))
    return np.asarray(chunk.transform, np.float64) @ m


def export_splat(chunk, path, frame: str = "survey", progress=None) -> dict:
    """The splat of the chunk into ``path``: in the survey coordinates of the chunk
    (``frame="survey"``: its CRS, or its local frame) or as Spirula wrote it (``"training"``).
    Returns ``{"path", "frame", "splats", "double"}``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import t

    progress = as_progress(progress)
    splat = chunk.product_path("splat")
    if splat is None or not splat.exists():
        raise RuntimeError(t("Il chunk non ha uno splat: Plugin > Spirula > Addestra Gaussian splatting.",
                             "The chunk has no splat: Plugin > Spirula > Train Gaussian splatting."))
    if frame not in ("survey", "training"):
        raise ValueError(f"frame {frame!r}: survey or training")
    path = Path(path)
    if path.resolve() == splat.resolve():
        raise ValueError(t("Scegli un file diverso dallo splat del chunk.", "Choose a file other than the splat of the chunk."))
    progress(0.1, splat.name)
    if frame == "training":
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(splat, path)
        progress(1.0, str(path))
        return {"path": path, "frame": frame, "splats": None, "double": False}
    matrix = survey_matrix(chunk)
    crs = chunk.crs or "LOCAL"
    progress(0.3, t("Splat nelle coordinate del rilievo", "Splat in the survey coordinates"))
    info = transform_splat(splat, path, matrix,
                           [f"comment pyarchinit-photogrammetry: survey coordinates ({crs}) of {splat.name}"])
    progress(1.0, str(path))
    return {**info, "frame": frame}
