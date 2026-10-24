# Blender and EMtools

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/blender-em/)**: what the project must hold, the em.json, the EM project, Blender live, the cameras with their photos, the way back from Blender, the sync with EMtools.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 or later): the **3D proxies of the stratigraphic units**, the **photogrammetric survey** and
the **aligned cameras with their photos** in [Blender](https://www.blender.org) and in the Extended
Matrix of [EMtools](https://github.com/zalmoxes-laran/EM-blender-tools), kept up to date while the
work goes on, as the QGIS link keeps the drawings up to date; and what is modelled in Blender back
into the program, as the proxies of the units with their volume.

| | What | How |
|---|---|---|
| **The matrix** | the em.json of the site, written by PyArchInit (*Extended Matrix → Esporta sito in em.json…*, or *Esporta progetto EM per EMStudio…* with the whole `EM/` tree), EMStudio or EMtools 1.6 | read, never written: the units, their epochs and colours, the shift of the project |
| **The pairing** | each proxy of the chunk (*Strumenti > Proxy 3D delle US*) with its unit | by the `node_uuid` of the US form (a PyArchInit database connected), else by area, type and number, else by name; ambiguities said, not guessed |
| **The EM project** | `EM/proxies/<unit>.glb`, `RB/<chunk>/<chunk>.glb`, `ppg_scene.json` | the standard tree of s3dgraphy and EMtools 1.6; each proxy named as its node (`1.US12`), the name EMtools pairs a proxy object with |
| **Blender, live** | Blender opened on the EM project with a script of the plugin | the objects of the manifest imported and followed: a proxy built again is replaced within a second, keeping its object and the material EMtools gave it |
| **EMtools, live** | the selection shared with EMtools 1.6 | the program joins the WebSocket of EMtools' live sync (port 8788, wire 2, as EMStudio): a proxy picked in Blender selects its unit in the pane, a unit picked in the pane selects and frames its proxy in Blender |
| **Cameras and photos** | the aligned cameras of the chunk in Blender, each with its photo behind it | each photo undistorted with the calibration of its sensor (fisheye too, through pycolmap) and made smaller into `RB/<chunk>/foto`; the camera with its pose, focal length and principal point: looking through it, the survey and the proxies lie on the photo |
| **Back from Blender** | the objects modelled in Blender and sent with *Invia al programma* (PPG panel) | into the chunk as the proxies of their units (by node id or by name), a PLY in double precision with its volume (small holes closed first), area in plan and elevations; a proxy of the same name is replaced, and the object stays in Blender as the program's proxy |
| **Proxy in Blender** | the command `create_proxy_for_unit` of EMtools 1.6 | for the units picked that have no proxy, EMtools makes a box bound to the node, as large as the unit drawn in the chunk and with its top on the DEM, to model and send back; EMtools must *Accept commands from EMStudio* |

## The EM project

```
<project>/EM/<site>.em.json        the matrix (PyArchInit's, EMStudio's)
<project>/EM/proxies/<unit>.glb    a proxy per unit, named as its node
<project>/RB/<chunk>/<chunk>.glb   the survey (reality-based), with its colours or its texture
<project>/RB/<chunk>/foto/         the photos of the cameras, undistorted
<project>/ppg_scene.json           the manifest the live script in Blender follows
<project>/ppg_inbox/               what Blender sends back, until the program takes it
```

The folders come from the em.json (its project, or beside it as EMtools does), or from a folder
chosen when no em.json is linked; then the proxies keep the program's names (`US12`, `USM3`).
glTF is Y-up and Blender turns it Z-up. The coordinates are **shifted by whole metres**, since a
float32 keeps a quarter of a metre at four thousand kilometres: by the shift of the em.json (its
`geo_position`) when it has one, else by the lowest corner of what is written, rounded down. Once
chosen the shift stays, so an update never moves what Blender shows; it goes into the scene
(`ppg_shift`, and the `BL_x_shift`... EMtools reads from 3D Survey Collection, when not set).

A file is written again only when what it comes from changed, and **never over a file another
program wrote or changed** in the same folder (EMtools exports the proxies modelled in Blender
there): that is reported and left alone. A proxy gone from the chunk has its file removed only
when it is still the one written here.

## Blender

*Apri in Blender* writes the EM project and starts Blender (the newest `.blend` of the project when
there is one, else a new scene) with `blender_live.py`, which:

* imports the proxies into the collection `PPG Proxy`, the survey into `PPG Rilievo` and the cameras
  into `PPG Camere`, with the unit, area, type, epoch, node id and volume as custom properties of each
  object;
* reads the manifest again every second: a proxy built again changes its mesh only (its object, its
  material from EMtools, its place in EMtools' lists stay), a proxy gone is removed;
* **never overwrites what was changed in Blender**: an object whose vertices are not those imported
  is *detached*, and the panel (*PPG* in the sidebar of the 3D view) lists it;
* loads the em.json into EMtools 1.6 (*Carica l'em.json in EMtools*, and by itself on a new scene),
  which pairs the proxies with the units by name;
* sends the objects picked to the program (*Invia al programma*): a glTF and a description in
  `ppg_inbox`, which the program takes with *Importa da Blender*, or by itself with the live update;
  when the program writes the project again, the object sent is adopted as the proxy, not duplicated.

With *Aggiornamento dal vivo* the program writes the EM project again, in the background, every
time the proxies, the model or the em.json change. Blender is found where it is installed, or where
*Scegli il programma Blender...* or `PPG_BLENDER` say.

## EMtools live

EMtools 1.6 hosts the live sync of the Extended Matrix: start it in Blender (EM panel, *Sync*), then
tick *Sincronizza con EMtools* in the pane. The program connects to `127.0.0.1:8788` (another port
in the settings of the plugin, `port`), says which matrix it has open (`client_info`), asks EMtools'
graph (`snapshot`: with no em.json linked the pane and the names of the proxies follow it), and
exchanges the selection (`select`). With *Proxy in Blender* it sends the command
`create_proxy_for_unit` (`command`, with its deterministic `cmd_id`, computed as EMtools and EMStudio
compute it) and reads the `command_result`. It never sends graph edits: the matrix belongs to EMtools
and EMStudio. It connects again by itself when Blender restarts. EMtools 1.5 has no sync: the files,
the cameras, the live script and the way back from Blender work the same.

## Use

* *Plugin > Blender ed EMtools*: the pane, *Collega l'em.json del sito...*, *Esporta nel progetto
  EM*, *Apri in Blender (aggiornato dal vivo)*, *Importa i proxy da Blender*, *Scegli il programma
  Blender...*; in the pane also *Proxy in Blender* and the switches *Aggiornamento dal vivo*,
  *Sincronizza con EMtools*, *Camere e foto*; the *Progetto EM
  (Blender)* item under the chunk in the workspace.
* Command line:

  ```bash
  ppg plugin blender-em status scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --out Scavo --no-survey
  ppg plugin blender-em export scavo.ppg --cameras --photo-side 3000   # with the cameras and photos
  ppg plugin blender-em open scavo.ppg            # the em.json linked to the chunk
  ppg plugin blender-em take scavo.ppg            # the proxies sent from Blender, into the chunk
  ppg plugin blender-em blender                   # the Blender programs found
  ```

Nothing beyond what the program carries: the em.json is read as JSON, the WebSocket is spoken with
the standard library, the photos are undistorted with pycolmap and OpenCV, the holes are closed with
numpy (no networkx, which the packaged program lacks), the live script runs in Blender's own Python.

## Tests

```bash
pip install -e .
pytest -q tests
PPG_TEST_BLENDER=/path/to/blender pytest -q tests/test_blender_live.py   # or a Python with bpy
```

The em.json of [s3dgraphy](https://pypi.org/project/s3dgraphy/) 1.6 is read in a test when it is
installed; the sync runs against a host that speaks EMtools' wire, with the `cmd_id` checked against
the value EMtools 1.6 gives. The tests with Blender (or the `bpy` module) go the whole way round:
export, import into the scene, a proxy modelled and sent, taken by the program and adopted by Blender
without a duplicate; the cameras registered on the undistorted photos.

Licence: GPL-2.0-or-later, like the program the plugin runs in. Author: Enzo Cocca.
