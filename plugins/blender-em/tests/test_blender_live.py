"""The live script in a real Blender: the objects of the manifest imported where the program built
them, a proxy built again replaced in place (its object and the material EMtools gave it kept), a
proxy changed in Blender never overwritten, a proxy gone removed.

Blender runs in a process of its own, as when the plugin starts it: ``PPG_TEST_BLENDER`` is a
Blender executable or a Python with the ``bpy`` module (``pip install bpy``); without either the
tests are skipped. The launcher's command line is tested without Blender."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from samples import emjson, project

from ppg_blender_em import blender, scene
from ppg_blender_em import emjson as em


def _bpy_runner():
    exe = os.environ.get("PPG_TEST_BLENDER") or shutil.which("blender")
    if not exe:
        return None
    return exe


DRIVER = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {package!r})
    import bpy
    import blender_live as live

    def state():
        out = {{}}
        for o in bpy.context.scene.objects:
            if o.type != "MESH":
                continue
            co = [o.matrix_world @ v.co for v in o.data.vertices]
            out[o.name] = {{"min": [min(c[i] for c in co) for i in range(3)],
                           "max": [max(c[i] for c in co) for i in range(3)],
                           "collection": o.users_collection[0].name, "kind": o.get("ppg_kind"),
                           "node_id": o.get("ppg_node_id"), "materials": [m.name for m in o.data.materials],
                           "pointer": o.as_pointer(), "detached": bool(o.get("ppg_detached"))}}
        return out

    step, done = {step!r}, {{}}
    if step == "first":
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:
        bpy.ops.wm.open_mainfile(filepath={blend!r})
    if step == "edit":  # a material of EMtools on US3, a vertex of USM2 moved by hand
        mat = bpy.data.materials.new("ep_Eta romana")
        o = bpy.data.objects["US3"]
        o.data.materials.clear()
        o.data.materials.append(mat)
        bpy.data.objects["USM2"].data.vertices[0].co.z += 0.5
    else:
        done = live.sync({manifest!r})
    print("PPG_RESULT " + json.dumps({{"done": done, "state": state(),
                                      "shift": list(bpy.context.scene.get("ppg_shift", []))}}))
    bpy.ops.wm.save_as_mainfile(filepath={blend!r})
''')


def _run(exe, tmp_path, manifest, step):
    script = tmp_path / f"driver_{step}.py"
    script.write_text(DRIVER.format(package=str(Path(blender.LIVE_SCRIPT).parent), manifest=str(manifest),
                                    blend=str(tmp_path / "scena.blend"), step=step), encoding="utf-8")
    if Path(exe).name.startswith("python"):
        cmd = [exe, str(script)]
    else:
        cmd = [exe, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PPG_RESULT ")), None)
    assert line is not None, (r.stdout + r.stderr)[-3000:]
    return json.loads(line.split(" ", 1)[1])


def test_the_command_line_of_blender(tmp_path):
    cmd = blender.command("/opt/blender/blender", tmp_path / "ppg_scene.json", tmp_path / "a.blend", emtools=True)
    assert cmd[:4] == ["/opt/blender/blender", str(tmp_path / "a.blend"), "--python", str(blender.LIVE_SCRIPT)]
    assert cmd[4:] == ["--", str((tmp_path / "ppg_scene.json").resolve()), "--emtools"]
    assert blender.LIVE_SCRIPT.exists()
    assert blender.find_blender(str(blender.LIVE_SCRIPT)) == str(blender.LIVE_SCRIPT)  # a configured file wins


def test_the_environment_of_blender(monkeypatch):
    env = {"PATH": "/usr/bin", "PYTHONHOME": "/app", "QT_PLUGIN_PATH": "/app/qt", "LD_LIBRARY_PATH": "/app",
           "LD_LIBRARY_PATH_ORIG": "/usr/local/lib", "DYLD_LIBRARY_PATH": "/app"}
    out = blender.environment(env)
    assert out == {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/local/lib", "DYLD_LIBRARY_PATH": "/app"}
    monkeypatch.setattr(sys, "frozen", True, raising=False)  # the packaged application, without the originals
    assert blender.environment({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/app"}) == {"PATH": "/usr/bin"}


def test_the_live_script_in_blender(tmp_path):
    exe = _bpy_runner()
    if exe is None:
        pytest.skip("no Blender: PPG_TEST_BLENDER is a Blender or a Python with bpy")
    doc = project(tmp_path)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    report = scene.export_project(doc.chunk, folders)
    manifest = report["manifest"]
    first = _run(exe, tmp_path, manifest, "first")
    assert sorted(first["done"]["imported"]) == ["Rilievo Chunk 1", "US1", "US3", "USM2"]
    state = first["state"]
    assert state["US1"]["collection"] == "PPG Proxy" and state["Rilievo Chunk 1"]["collection"] == "PPG Rilievo"
    assert first["shift"] == report["shift"]
    # in Blender, Z up and shifted: where the program built the proxy, less the shift
    import numpy as np
    import trimesh

    for entry in doc.chunk.meta["proxies"]:
        world = np.asarray(trimesh.load(str(doc.chunk.data_dir / entry["file"]), process=False, force="mesh").vertices)
        got = state[entry["name"]]
        assert got["min"] == pytest.approx(world.min(0) - report["shift"], abs=1e-3)
        assert got["max"] == pytest.approx(world.max(0) - report["shift"], abs=1e-3)
    # the proxies built again in the program, US3 and USM2 deeper: US3 (given a material of EMtools)
    # changes in place, USM2 (changed by hand in Blender) is left as it is
    _run(exe, tmp_path, manifest, "edit")
    for entry in doc.chunk.meta["proxies"][1:]:
        f = doc.chunk.data_dir / entry["file"]
        mesh = trimesh.load(str(f), process=False, force="mesh")
        mesh.vertices[:, 2] -= 0.2
        mesh.export(str(f))
    assert scene.export_project(doc.chunk, folders)["written"] == ["USM2", "US3"]
    second = _run(exe, tmp_path, manifest, "again")
    assert second["done"]["updated"] == ["US3"] and second["done"]["detached"] == ["USM2"]
    us3 = second["state"]["US3"]
    assert us3["materials"] == ["ep_Eta romana"]  # the material EMtools gave it stays
    assert us3["min"][2] == pytest.approx(state["US3"]["min"][2] - 0.2, abs=1e-3)
    assert second["state"]["USM2"]["detached"]
    # US1 gone from the chunk: gone from Blender
    doc.chunk.meta["proxies"] = doc.chunk.meta["proxies"][1:]
    scene.export_project(doc.chunk, folders)
    third = _run(exe, tmp_path, manifest, "again")
    assert third["done"]["removed"] == ["US1"] and "US1" not in third["state"]


def test_the_names_of_the_matrix_in_blender(tmp_path):
    exe = _bpy_runner()
    if exe is None:
        pytest.skip("no Blender: PPG_TEST_BLENDER is a Blender or a Python with bpy")
    doc = project(tmp_path, model=False)
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path)
    report = scene.export_project(doc.chunk, scene.project_folders(emjson=em_path), em.load(em_path))
    first = _run(exe, tmp_path, report["manifest"], "first")
    assert sorted(first["state"]) == ["1.US1", "1.US3", "1.USM2"]  # the names EMtools pairs with the units
    assert all(v["node_id"] for v in first["state"].values())


ROUND = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {package!r})
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    import blender_live as live

    step, out = {step!r}, {{}}
    if step == "first":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        out["done"] = live.sync({manifest!r})
        scene = bpy.context.scene
        cam = bpy.data.objects["IMG_0000"]
        dot = Vector({dot!r})
        view = world_to_camera_view(scene, cam, dot)
        out["view"] = [view.x, view.y, view.z]
        out["resolution"] = [scene.render.resolution_x, scene.render.resolution_y]
        out["background"] = cam.data.background_images[0].image.filepath if cam.data.background_images else ""
        out["cameras"] = sorted(o.name for o in bpy.data.collections["PPG Camere"].objects)
        # USM2 modelled by hand (5 cm higher), then sent to the program
        obj = bpy.data.objects["USM2"]
        for v in obj.data.vertices:
            v.co.z += 0.05
        out["sent"] = live.send([obj], {manifest!r})
    else:
        bpy.ops.wm.open_mainfile(filepath={blend!r})
        out["done"] = live.sync({manifest!r})
        obj = bpy.data.objects["USM2"]
        out["usm2"] = {{"detached": bool(obj.get("ppg_detached")), "file": obj.get("ppg_file"),
                        "count": len([o for o in bpy.data.objects if o.name.startswith("USM2")])}}
    print("PPG_RESULT " + json.dumps(out))
    bpy.ops.wm.save_as_mainfile(filepath={blend!r})
''')


def _round(exe, tmp_path, manifest, step, dot):
    script = tmp_path / f"round_{step}.py"
    script.write_text(ROUND.format(package=str(Path(blender.LIVE_SCRIPT).parent), manifest=str(manifest),
                                   blend=str(tmp_path / "giro.blend"), step=step, dot=[float(v) for v in dot]),
                      encoding="utf-8")
    cmd = [exe, str(script)] if Path(exe).name.startswith("python") else \
        [exe, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PPG_RESULT ")), None)
    assert line is not None, (r.stdout + r.stderr)[-3000:]
    return json.loads(line.split(" ", 1)[1])


def test_cameras_and_the_way_back_in_blender(tmp_path):
    exe = _bpy_runner()
    if exe is None:
        pytest.skip("no Blender: PPG_TEST_BLENDER is a Blender or a Python with bpy")
    import numpy as np
    from samples import DOT, add_cameras

    from ppg_blender_em import inbox

    doc = project(tmp_path)
    add_cameras(doc.chunk, tmp_path / "foto", n=2)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    report = scene.export_project(doc.chunk, folders, cameras=True, max_side=320)
    shift = np.asarray(report["shift"])
    first = _round(exe, tmp_path, report["manifest"], "first", DOT - shift)
    assert first["cameras"] == ["IMG_0000", "IMG_0001"] and first["resolution"] == [320, 240]
    assert first["background"].endswith("IMG_0000.jpg")
    # Blender's camera sees the point where the pinhole of the undistorted photo sees it
    record = scene.read_manifest(folders.base)["cameras"][0]
    pose = np.asarray(record["matrix"])
    p = np.linalg.inv(pose) @ np.append(DOT - shift, 1.0)
    u = record["fx"] * p[0] / p[2] + record["cx"]
    v = record["fy"] * p[1] / p[2] + record["cy"]
    assert first["view"][0] == pytest.approx(u / 320, abs=0.004)  # the 1 % of fx and fy aside, a pixel
    assert first["view"][1] == pytest.approx(1 - v / 240, abs=0.004)
    assert first["view"][2] == pytest.approx(p[2], abs=1e-4)
    # the proxy modelled in Blender comes into the chunk, and back in Blender it is adopted, not doubled
    assert first["sent"] == ["USM2"] and [f.stem for f in inbox.pending(folders)] == ["USM2"]
    taken = inbox.take(doc.chunk, folders, None)
    assert taken["taken"] == ["USM2"] and taken["replaced"] == ["USM2"]
    entry = next(e for e in doc.chunk.meta["proxies"] if e["name"] == "USM2")
    assert entry["base"] == "blender" and entry["closed"] and entry["volume_m3"] > 0
    assert "USM2" in scene.export_project(doc.chunk, folders)["written"]
    again = _round(exe, tmp_path, report["manifest"], "again", DOT - shift)
    assert again["done"]["adopted"] == ["USM2"] and again["done"]["detached"] == []
    assert again["usm2"] == {"detached": False, "file": "EM/proxies/USM2.glb", "count": 1}
