"""The live link in Blender: the objects of a manifest of pyarchinit-photogrammetry, kept up to date.

Blender runs this file (``blender --python blender_live.py -- MANIFEST``: *Apri in Blender* of the
plugin starts it so; *Scripting > Apri* and *Esegui* with the manifest chosen in the panel does the
same). It imports the glTF files the manifest lists - the proxies of the units into the collection
``PPG Proxy``, the survey into ``PPG Rilievo`` - and then reads the manifest again every second, as
QGIS redraws the layers of the live link: a proxy built again in pyarchinit-photogrammetry changes
here by itself.

* A proxy keeps its object: only its mesh is replaced, so what EMtools gave it (the material of its
  epoch, its place in the lists) stays.
* An object changed in Blender (its vertices are not those imported) is never replaced nor removed:
  it is *detached* and said in the panel - the work done here wins.
* The shift of the coordinates goes into the scene (``ppg_shift``, and the ``BL_x_shift``... of
  EMtools / 3D Survey Collection when they are installed and not set yet).
* With EMtools 1.6 the em.json of the manifest can be loaded into EMtools from the panel (*Carica
  l'em.json in EMtools*), which pairs the proxies with the units by name.
* *Invia al programma* sends the selected objects (or those changed here) back: a glTF and a
  description each in ``ppg_inbox`` beside the manifest, which the program takes as the proxies of
  their units. When the program writes them again, the objects sent are *adopted* - they stay as
  they are, followed again by the link - instead of being imported a second time.
* The cameras of the manifest go into ``PPG Camere``, with their pose, focal length, principal point
  and the undistorted photo behind each: looking through one, the proxies lie on the photo.

Only bpy and the Python of Blender: nothing of pyarchinit-photogrammetry is imported here.
"""
from __future__ import annotations

import json
import os
import sys
import time

import bpy

FORMAT = "ppg-blender-em"
COLLECTIONS = {"proxy": "PPG Proxy", "survey": "PPG Rilievo", "camera": "PPG Camere"}
INBOX = "ppg_inbox"  # beside the manifest: what Blender sends back to the program
INTERVAL = 1.0  # seconds between two looks at the manifest
STATE = {"manifest": "", "mtime": None, "revision": None, "live": True, "log": [], "detached": [],
         "counts": {}, "error": ""}


def log(text: str) -> None:
    STATE["log"] = (STATE["log"] + [time.strftime("%H:%M:%S ") + text])[-12:]
    print("[ppg]", text)


def read_manifest(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("format") == FORMAT else None


def mesh_signature(obj) -> str:
    """The vertices of a mesh in a few characters: changed in Blender, or as imported."""
    import numpy as np

    mesh = obj.data
    n = len(mesh.vertices)
    if not n:
        return "0"
    co = np.empty(n * 3, np.float64)
    mesh.vertices.foreach_get("co", co)
    return f"{n}:{len(mesh.polygons)}:{co.sum():.4f}:{np.abs(co).sum():.4f}"


def _collection(scene, kind: str):
    name = COLLECTIONS.get(kind, "PPG")
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def managed(scene=None) -> dict:
    """The objects this link imported: ``{file: [objects]}``."""
    out = {}
    for obj in (scene or bpy.context.scene).objects:
        rel = obj.get("ppg_file")
        if rel:
            out.setdefault(rel, []).append(obj)
    return out


def _edited(obj) -> bool:
    return obj.type == "MESH" and obj.get("ppg_sig") not in (None, mesh_signature(obj))


def _sent(obj) -> bool:
    """The object was sent to the program as it is now (*Invia al programma*)."""
    return obj.type == "MESH" and bool(obj.get("ppg_sent_sig")) and obj.get("ppg_sent_sig") == mesh_signature(obj)


def _sent_for(scene, record: dict) -> list:
    """Objects of this scene the link did not import, sent to the program and now back in the manifest
    as ``record``: by the id of their node, else by name (also with the code of the graph EMtools puts
    before it)."""
    name, node = record.get("name") or "", record.get("node_id")
    found = [o for o in scene.objects if not o.get("ppg_file") and _sent(o)
             and ((node and o.get("ppg_node_id") == node) or o.name == name or o.name.endswith("." + name))]
    return found[:1]


def _import(path: str) -> list:
    """The objects of a glTF file, imported (the importer selects them and puts them in the active
    collection; they are moved afterwards)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def _hex(color: str | None):
    text = str(color or "").lstrip("#")
    try:
        return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
    except ValueError:
        return None


def _tag(obj, record: dict, sha1: str) -> None:
    obj["ppg_file"] = record["file"]
    obj["ppg_sha1"] = sha1
    obj["ppg_kind"] = record.get("kind") or ""
    for key in ("node_id", "us", "area", "tipo", "epoch", "graph_id", "volume_m3", "area_m2"):
        if record.get(key) not in (None, ""):
            obj["ppg_" + key] = record[key]
    if obj.type == "MESH":
        obj["ppg_sig"] = mesh_signature(obj)
    color = _hex(record.get("color"))
    if color is not None:
        obj.color = color


def _drop_mesh(mesh) -> None:
    """A mesh nobody uses any more, with the materials of the link it alone used."""
    if mesh is None or mesh.users:
        return
    mats = [m for m in mesh.materials if m is not None and m.get("ppg")]
    bpy.data.meshes.remove(mesh)
    for m in mats:
        if not m.users:
            bpy.data.materials.remove(m)


def _remove(objs) -> None:
    for obj in objs:
        data = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        _drop_mesh(data)


def _swap_mesh(old, new) -> None:
    """``new``'s mesh into the object ``old``; the materials of ``old`` stay when they are not the
    link's own (EMtools gives the proxies the material of their epoch)."""
    keep = [m for m in old.data.materials] if any(m is not None and not m.get("ppg") for m in old.data.materials) \
        else None
    data_old = old.data
    old.data = new.data
    old.matrix_world = new.matrix_world.copy()
    if keep is not None:
        old.data.materials.clear()
        for m in keep:
            old.data.materials.append(m)
    bpy.data.objects.remove(new, do_unlink=True)
    _drop_mesh(data_old)


def set_shift(scene, manifest: dict) -> None:
    shift = manifest.get("shift") or [0.0, 0.0, 0.0]
    scene["ppg_shift"] = [float(v) for v in shift]
    scene["ppg_crs"] = str(manifest.get("crs") or "")
    # EMtools reads the shift of the scene from 3D Survey Collection: set it when it is not set
    if all(hasattr(scene, a) for a in ("BL_x_shift", "BL_y_shift", "BL_z_shift")):
        if not any((scene.BL_x_shift, scene.BL_y_shift, scene.BL_z_shift)):
            scene.BL_x_shift, scene.BL_y_shift, scene.BL_z_shift = (float(v) for v in shift)
            if hasattr(scene, "BL_epsg") and manifest.get("epsg") and scene.BL_epsg in ("", "NotSet"):
                scene.BL_epsg = str(manifest["epsg"])


def sync(path: str, scene=None) -> dict:
    """Bring the scene up to the manifest at ``path``. Returns what happened: ``imported``,
    ``updated``, ``kept``, ``removed``, ``detached`` (changed in Blender: left as they are)."""
    scene = scene or bpy.context.scene
    manifest = read_manifest(path)
    if manifest is None:
        STATE["error"] = f"manifest not readable: {path}"
        return {"error": STATE["error"]}
    STATE["error"] = ""
    base = os.path.dirname(os.path.abspath(path))
    have = managed(scene)
    done = {"imported": [], "updated": [], "kept": [], "removed": [], "detached": [], "adopted": []}
    wanted = set()
    for record in manifest.get("objects") or []:
        rel, sha1 = record.get("file"), record.get("sha1") or ""
        if not rel:
            continue
        wanted.add(rel)
        objs = have.get(rel, [])
        if objs and all(o.get("ppg_sha1") == sha1 for o in objs):
            done["kept"].append(record.get("name") or rel)
            continue
        adopted = objs if objs and all(_sent(o) for o in objs) else (_sent_for(scene, record) if not objs else [])
        if adopted:  # the geometry came from here: the object stays as it is, the link follows it again
            for o in adopted:
                _tag(o, record, sha1)
                o["ppg_detached"] = False
                del o["ppg_sent_sig"]
            done["adopted"].append(record.get("name") or rel)
            continue
        if any(_edited(o) for o in objs):
            for o in objs:
                o["ppg_detached"] = True
            done["detached"].append(record.get("name") or rel)
            continue
        file = os.path.join(base, rel)
        if not os.path.exists(file):
            continue
        new = _import(file)
        coll = _collection(scene, record.get("kind") or "")
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
            for mat in (o.data.materials if o.type == "MESH" else []):
                if mat is not None:
                    mat["ppg"] = True  # the link's own: replaced with the mesh
        meshes = [o for o in new if o.type == "MESH"]
        if record.get("kind") == "proxy" and len(objs) == 1 and objs[0].type == "MESH" and len(meshes) == 1:
            _swap_mesh(objs[0], meshes[0])
            _remove([o for o in new if o is not meshes[0]])
            _tag(objs[0], record, sha1)
            done["updated"].append(record.get("name") or rel)
            continue
        _remove(objs)
        for o in new:
            _tag(o, record, sha1)
        (done["updated"] if objs else done["imported"]).append(record.get("name") or rel)
    for rel, objs in have.items():
        if rel in wanted:
            continue
        if any(_edited(o) for o in objs):
            for o in objs:
                o["ppg_detached"] = True
            done["detached"].append(objs[0].name)
        else:
            done["removed"].append(objs[0].name)
            _remove(objs)
    set_shift(scene, manifest)
    done["cameras"] = sync_cameras(scene, manifest, base)
    STATE.update(manifest=path, revision=manifest.get("revision"), detached=done["detached"],
                 counts={k: len(v) for k, v in done.items()})
    changed = {k: v for k, v in done.items() if v and k not in ("kept", "cameras")}
    if changed:
        log(f"revision {manifest.get('revision')}: " + "; ".join(f"{k} {', '.join(v)}" for k, v in changed.items()))
    return done


def _safe(name: str) -> str:
    return "".join("_" if c in '\\/:*?"<>|' else c for c in name).strip(" .") or "proxy"


def send(objects, path: str, scene=None) -> list[str]:
    """The ``objects`` to the program, each a glTF in ``ppg_inbox`` beside the manifest at ``path``
    (Y-up and shifted, as the files of the project) with a description: the program takes them as the
    proxies of their units, with their volume. The names sent."""
    import json as _json
    from datetime import datetime, timezone

    scene = scene or bpy.context.scene
    folder = os.path.join(os.path.dirname(os.path.abspath(path)), INBOX)
    os.makedirs(folder, exist_ok=True)
    view = bpy.context.view_layer
    before = [o for o in view.objects if o.select_get()], view.objects.active
    sent = []
    try:
        for obj in objects:
            if obj.type != "MESH":
                continue
            stem = os.path.join(folder, _safe(obj.name))
            for o in view.objects:
                o.select_set(False)
            obj.select_set(True)
            view.objects.active = obj
            part = stem + ".part.glb"
            bpy.ops.export_scene.gltf(filepath=part, export_format="GLB", use_selection=True, export_apply=True,
                                      export_yup=True)
            os.replace(part, stem + ".glb")
            info = {"name": obj.name, "node_id": obj.get("ppg_node_id"), "kind": "proxy",
                    "shift": list(scene.get("ppg_shift", [0.0, 0.0, 0.0])),
                    "sent": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "blender": bpy.app.version_string}
            with open(stem + ".json.part", "w", encoding="utf-8") as f:
                _json.dump(info, f, ensure_ascii=False)
            os.replace(stem + ".json.part", stem + ".json")
            obj["ppg_sent_sig"] = mesh_signature(obj)
            sent.append(obj.name)
    finally:
        for o in view.objects:
            o.select_set(o in before[0])
        view.objects.active = before[1]
    if sent:
        log("sent to the program: " + ", ".join(sent))
    return sent


def _camera_signature(record: dict) -> str:
    import hashlib
    import json as _json

    keys = ("file", "matrix", "width", "height", "fx", "fy", "cx", "cy", "sha1")
    return hashlib.sha1(_json.dumps([record.get(k) for k in keys]).encode()).hexdigest()


def place_camera(obj, record: dict, image_path: str | None) -> None:
    """A camera of the program as a Blender camera: its pose (OpenCV axes: x right, y down, z forward;
    Blender's camera looks along -z with y up), its focal length and principal point on the frame of
    the photo, and the photo - undistorted by the program - behind it."""
    from mathutils import Matrix

    m = Matrix(record["matrix"])
    rot = m.to_3x3() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    obj.matrix_world = Matrix.Translation(m.translation) @ rot.to_4x4()
    cam = obj.data
    w, h = float(record["width"]), float(record["height"])
    side = max(w, h)
    cam.sensor_fit = "AUTO"
    cam.sensor_width = 36.0
    cam.lens = 0.5 * (float(record["fx"]) + float(record["fy"])) * cam.sensor_width / side
    cam.shift_x = (w / 2.0 - float(record["cx"])) / side
    cam.shift_y = (float(record["cy"]) - h / 2.0) / side
    cam.display_size = float(record.get("display_size") or 0.3)
    if image_path and os.path.exists(image_path):
        image = bpy.data.images.load(image_path, check_existing=True)
        image.reload()
        bg = cam.background_images[0] if len(cam.background_images) else cam.background_images.new()
        bg.image = image
        bg.frame_method = "STRETCH"
        bg.alpha = 1.0
        cam.show_background_images = True


def sync_cameras(scene, manifest: dict, base: str) -> dict:
    """The cameras of the manifest in the collection ``PPG Camere``: made, moved, or taken away. A
    manifest without the list leaves them as they are."""
    records = manifest.get("cameras")
    done = {"cameras": 0, "removed": 0}
    if records is None:
        return done
    have = {o.get("ppg_camera"): o for o in scene.objects if o.get("ppg_camera")}
    coll = _collection(scene, "camera") if records else None
    for record in records:
        name = record.get("name")
        if not name:
            continue
        sig = _camera_signature(record)
        obj = have.pop(name, None)
        if obj is not None and obj.get("ppg_sig") == sig:
            continue
        if obj is None:
            obj = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            coll.objects.link(obj)
        place_camera(obj, record, os.path.join(base, record["file"]) if record.get("file") else None)
        obj["ppg_camera"] = name
        obj["ppg_sig"] = sig
        obj["ppg_photo"] = record.get("photo") or ""
        done["cameras"] += 1
    for obj in have.values():
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.cameras.remove(data)
        done["removed"] += 1
    size = manifest.get("resolution")
    if records and size and not scene.get("ppg_resolution"):
        # the frame of the photos, once: the user's own resolution is not touched afterwards
        scene.render.resolution_x, scene.render.resolution_y = int(size[0]), int(size[1])
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        scene["ppg_resolution"] = [int(size[0]), int(size[1])]
    if done["cameras"] or done["removed"]:
        log(f"cameras: {done['cameras']} placed, {done['removed']} removed")
    return done


def load_in_emtools(path: str) -> str:
    """The em.json of the manifest into EMtools 1.6 (``import.em_emjson``); what happened, in words."""
    manifest = read_manifest(path) or {}
    emjson = manifest.get("emjson")
    if not emjson or not os.path.exists(emjson):
        return "the manifest names no em.json"
    ops = getattr(bpy.ops, "import")
    if not hasattr(ops, "em_emjson"):
        return "EMtools 1.6 is not installed (EMtools 1.5 reads GraphML only)"
    try:
        result = ops.em_emjson(filepath=emjson)
    except Exception as exc:  # noqa: BLE001 - EMtools' own refusal, said in the panel
        return f"EMtools did not load {os.path.basename(emjson)}: {exc}"
    return f"em.json loaded in EMtools: {os.path.basename(emjson)}" if "FINISHED" in result else \
        f"EMtools did not load {os.path.basename(emjson)}"


def _tick():
    """The timer: the manifest again when it changed (its modification time)."""
    path = STATE.get("manifest")
    if not STATE.get("live") or not path:
        return INTERVAL
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return INTERVAL
    if mtime != STATE.get("mtime"):
        STATE["mtime"] = mtime
        try:
            sync(path)
        except Exception as exc:  # noqa: BLE001 - the timer must not die
            STATE["error"] = str(exc)
            log(f"error: {exc}")
        _redraw()
    return INTERVAL


def _redraw() -> None:
    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def start(path: str) -> None:
    """Follow the manifest at ``path`` (the timer survives the opening of another .blend)."""
    STATE["manifest"] = os.path.abspath(path)
    STATE["mtime"] = None
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.1, persistent=True)


# ---------------------------------------------------------------------- the panel (3D view, N)
class PPG_OT_reload(bpy.types.Operator):
    bl_idname = "ppg.reload"
    bl_label = "Aggiorna ora"
    bl_description = "Read the manifest of pyarchinit-photogrammetry again"

    def execute(self, context):
        if not STATE.get("manifest"):
            self.report({"WARNING"}, "no manifest")
            return {"CANCELLED"}
        sync(STATE["manifest"], context.scene)
        return {"FINISHED"}


class PPG_OT_live(bpy.types.Operator):
    bl_idname = "ppg.live"
    bl_label = "Aggiornamento dal vivo"
    bl_description = "Follow the changes of pyarchinit-photogrammetry, or stop"

    def execute(self, context):
        STATE["live"] = not STATE.get("live")
        STATE["mtime"] = None
        return {"FINISHED"}


class PPG_OT_choose(bpy.types.Operator):
    bl_idname = "ppg.choose"
    bl_label = "Scegli il manifest..."
    bl_description = "The ppg_scene.json pyarchinit-photogrammetry wrote in the EM project"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if read_manifest(self.filepath) is None:
            self.report({"ERROR"}, "not a manifest of pyarchinit-photogrammetry")
            return {"CANCELLED"}
        start(self.filepath)
        return {"FINISHED"}


class PPG_OT_send(bpy.types.Operator):
    bl_idname = "ppg.send"
    bl_label = "Invia al programma"
    bl_description = ("The selected objects (or those changed here, when none is selected) to "
                      "pyarchinit-photogrammetry, as the proxies of their units, with their volume")

    def execute(self, context):
        if not STATE.get("manifest"):
            self.report({"WARNING"}, "no manifest")
            return {"CANCELLED"}
        objs = [o for o in context.selected_objects if o.type == "MESH"] or \
            [o for o in context.scene.objects if o.type == "MESH" and o.get("ppg_detached")]
        if not objs:
            self.report({"WARNING"}, "select the proxies to send")
            return {"CANCELLED"}
        names = send(objs, STATE["manifest"], context.scene)
        self.report({"INFO"}, f"{len(names)} sent to pyarchinit-photogrammetry")
        return {"FINISHED"}


class PPG_OT_emtools(bpy.types.Operator):
    bl_idname = "ppg.load_emtools"
    bl_label = "Carica l'em.json in EMtools"
    bl_description = "The em.json of the EM project into EMtools 1.6, which pairs the proxies with the units"

    def execute(self, context):
        text = load_in_emtools(STATE.get("manifest") or "")
        log(text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


class PPG_PT_panel(bpy.types.Panel):
    bl_label = "pyarchinit-photogrammetry"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PPG"

    def draw(self, context):
        layout = self.layout
        path = STATE.get("manifest")
        if not path:
            layout.operator("ppg.choose", icon="FILE_FOLDER")
            return
        layout.label(text=os.path.basename(os.path.dirname(path)) or path, icon="FILE")
        layout.label(text=f"revisione {STATE.get('revision')}")
        row = layout.row()
        row.operator("ppg.live", text="Dal vivo: sì" if STATE.get("live") else "Dal vivo: no",
                     icon="PLAY" if STATE.get("live") else "PAUSE")
        row.operator("ppg.reload", text="", icon="FILE_REFRESH")
        layout.operator("ppg.send", icon="EXPORT")
        layout.operator("ppg.load_emtools", icon="NODETREE")
        if STATE.get("error"):
            layout.label(text=STATE["error"], icon="ERROR")
        if STATE.get("detached"):
            box = layout.box()
            box.label(text="Modificati qui, non sostituiti:", icon="INFO")
            for name in STATE["detached"][:8]:
                box.label(text=name)
        for line in STATE.get("log", [])[-4:]:
            layout.label(text=line)
        layout.operator("ppg.choose", text="Altro manifest...", icon="FILE_FOLDER")


CLASSES = (PPG_OT_reload, PPG_OT_live, PPG_OT_choose, PPG_OT_send, PPG_OT_emtools, PPG_PT_panel)


def register() -> None:
    for cls in CLASSES:
        try:
            bpy.utils.register_class(cls)
        except ValueError:  # registered already (the script run twice)
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)


def main(argv=None) -> None:
    argv = list(sys.argv if argv is None else argv)
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    register()
    path = args[0] if args else os.environ.get("PPG_BLENDER_MANIFEST", "")
    if path:
        start(path)
        try:
            sync(STATE["manifest"])
            STATE["mtime"] = os.stat(STATE["manifest"]).st_mtime_ns
        except Exception as exc:  # noqa: BLE001 - the panel says it, the timer tries again
            STATE["error"] = str(exc)
            log(f"error: {exc}")
        if "--emtools" in args:
            log(load_in_emtools(STATE["manifest"]))


if __name__ == "__main__":
    main()
