"""The dense cloud file with dimensions of the plugin beside the program's own.

The program keeps the dense cloud in a binary PLY (``point_cloud.ply``) with the coordinates, the
normals, the colours, the classes, the confidence and the hand labels (``manual``), as
:class:`~pyarchinit_photogrammetry.core.pointcloud.PointCloud` writes them. A value the plugin
takes from a raster for every point - a temperature, an index, a height - is one more property of
the same vertices (a ``float``, NaN where the raster has no value), as the program keeps its
confidence: readers that do not know it skip it (CloudCompare shows it as a scalar field). The
program reads the cloud without it and writes it again without it when it changes the cloud
(cleaning, classes, cropping): :func:`extra_names` then finds none, and the plugin says so.
The plugin's own steps carry the values over (:func:`write`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# the properties of the program's PointCloud: never a dimension of the plugin
KNOWN = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "classification", "confidence", "manual")
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_TYPE_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
               "f4": "float", "f8": "double"}


def dimension_name(text: str) -> str:
    """A name for an extra dimension: lower case letters, digits and ``_``, starting with a letter
    (``Temperatura (°C)`` -> ``temperatura_c``); the program's own properties are refused."""
    name = re.sub(r"[^a-z0-9_]+", "_", str(text).strip().lower()).strip("_")
    if name and not name[0].isalpha():
        name = "v_" + name
    if not name:
        raise ValueError("the dimension needs a name")
    if name in KNOWN:
        raise ValueError(f"{name!r} is a property of the program's clouds: choose another name")
    return name[:40]


def _header(path: Path) -> tuple[list[tuple[str, str]], int, int] | None:
    """``(fields, count, offset)`` of the vertex element of a binary little-endian PLY whose
    first element is the vertex one (as the program writes the dense cloud), else None."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            return None
        fmt = fh.readline().decode("ascii", "replace").split()
        if len(fmt) < 2 or fmt[1] != "binary_little_endian":
            return None
        fields, count, element, other = [], 0, None, False
        while True:
            raw = fh.readline()
            if not raw:
                return None
            line = raw.decode("ascii", "replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    count = int(parts[2])
                else:
                    other = True
            elif parts[0] == "property" and element == "vertex":
                if parts[1] == "list" or parts[1] not in _PLY_TYPES:
                    return None
                fields.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
        if other or not fields:  # faces or other elements: not a dense cloud of the program
            return None
        return fields, count, fh.tell()


def extra_names(path) -> list[str]:
    """The properties of the vertices of the cloud file that are not the program's own."""
    if path is None or not Path(path).exists():
        return []
    head = _header(Path(path))
    return [] if head is None else [n for n, _ in head[0] if n not in KNOWN]


def read_extras(path, names=None) -> dict[str, np.ndarray]:
    """The extra properties (``names``, else all) of the cloud file, as arrays."""
    if path is None or not Path(path).exists():
        return {}
    head = _header(Path(path))
    if head is None:
        return {}
    fields, count, offset = head
    wanted = [n for n, _ in fields if n not in KNOWN and (names is None or n in names)]
    if not wanted:
        return {}
    data = np.memmap(path, dtype=fields, mode="r", offset=offset, shape=(count,))
    out = {n: np.array(data[n]) for n in wanted}
    del data
    return out


def _ply_type(values: np.ndarray) -> str:
    """The PLY type of an array: its own when PLY has it, else float."""
    code = np.dtype(values.dtype).str[1:]
    return "<" + code if code in _TYPE_NAMES else "<f4"


def add_properties(path, extras: dict[str, np.ndarray]) -> Path:
    """Write ``extras`` (one value per vertex) as properties of the vertices of the PLY ``path``,
    after the ones it has (a property of the same name is replaced). The file is written beside
    and moved in place at the end."""
    path = Path(path)
    head = _header(path)
    if head is None:
        raise ValueError(f"{path.name}: not a binary PLY point cloud")
    fields, count, offset = head
    extras = {str(k): np.asarray(v) for k, v in (extras or {}).items()}
    for name, values in extras.items():
        if name in KNOWN:
            raise ValueError(f"{name!r} is a property of the program's clouds")
        if len(values) != count:
            raise ValueError(f"{name}: {len(values)} values for {count} points")
    kept = [(n, f) for n, f in fields if n not in extras]
    added = [(n, _ply_type(v)) for n, v in extras.items()]
    new_fields = kept + added
    with open(path, "rb") as fh:
        lines = fh.read(offset).decode("ascii", "replace").splitlines()
    header = []
    for line in lines:
        parts = line.split()
        if parts[:1] == ["property"] and len(parts) == 3 and parts[2] in extras:
            continue  # replaced below
        if line.strip() == "end_header":
            header += [f"property {_TYPE_NAMES[f[1:]]} {n}" for n, f in added]
        header.append(line)
    tmp = path.with_name(path.name + ".terrio.tmp")
    source = np.memmap(path, dtype=fields, mode="r", offset=offset, shape=(count,))
    try:
        with open(tmp, "wb") as out:
            out.write(("\n".join(header) + "\n").encode("ascii"))
            step = 2_000_000
            for start in range(0, count, step):
                stop = min(count, start + step)
                block = np.empty(stop - start, dtype=new_fields)
                for n, _ in kept:
                    block[n] = source[n][start:stop]
                for n, _ in added:
                    block[n] = extras[n][start:stop]
                out.write(block.tobytes())
    finally:
        del source
    os.replace(tmp, path)
    return path


def write(chunk, pc, extras: dict[str, np.ndarray] | None = None) -> Path:
    """``pc`` as the dense cloud of ``chunk`` - written by the program, as its own steps write it -
    with the ``extras`` dimensions of the plugin added to its vertices."""
    chunk.point_cloud = pc
    path = chunk.product_path("point_cloud")
    if extras:
        add_properties(path, extras)
    return path


def save(pc, path, extras: dict[str, np.ndarray] | None = None) -> Path:
    """``pc`` written by the program into ``path`` (a version, a preview), with ``extras``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pc.save_ply(path)
    if extras:
        add_properties(path, extras)
    return path


def subset(extras: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """The extra dimensions of the points kept (``keep``: a mask or indices)."""
    return {k: v[keep] for k, v in (extras or {}).items()}
