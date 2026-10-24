# Blender and EMtools

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/blender-em/)**: what the project must hold, the em.json, the EM project, Blender live, the sync with EMtools.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 or later): the **3D proxies of the stratigraphic units** and the **photogrammetric survey**
in [Blender](https://www.blender.org) and in the Extended Matrix of
[EMtools](https://github.com/zalmoxes-laran/EM-blender-tools), kept up to date while the work goes
on, as the QGIS link keeps the drawings up to date.

| | What | How |
|---|---|---|
| **The matrix** | the em.json of the site, written by PyArchInit (*Extended Matrix → Esporta sito in em.json…*, or *Esporta progetto EM per EMStudio…* with the whole `EM/` tree), EMStudio or EMtools 1.6 | read, never written: the units, their epochs and colours, the shift of the project |
| **The pairing** | each proxy of the chunk (*Strumenti > Proxy 3D delle US*) with its unit | by the `node_uuid` of the US form (a PyArchInit database connected), else by area, type and number, else by name; ambiguities said, not guessed |
| **The EM project** | `EM/proxies/<unit>.glb`, `RB/<chunk>/<chunk>.glb`, `ppg_scene.json` | the standard tree of s3dgraphy and EMtools 1.6; each proxy named as its node (`1.US12`), the name EMtools pairs a proxy object with |
| **Blender, live** | Blender opened on the EM project with a script of the plugin | the objects of the manifest imported and followed: a proxy built again is replaced within a second, keeping its object and the material EMtools gave it |
| **EMtools, live** | the selection shared with EMtools 1.6 | the program joins the WebSocket of EMtools' live sync (port 8788, wire 2, as EMStudio): a proxy picked in Blender selects its unit in the pane, a unit picked in the pane selects and frames its proxy in Blender |

## The EM project

```
<project>/EM/<site>.em.json        the matrix (PyArchInit's, EMStudio's)
<project>/EM/proxies/<unit>.glb    a proxy per unit, named as its node
<project>/RB/<chunk>/<chunk>.glb   the survey (reality-based), with its colours or its texture
<project>/ppg_scene.json           the manifest the live script in Blender follows
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

* imports the proxies into the collection `PPG Proxy` and the survey into `PPG Rilievo`, with the
  unit, area, type, epoch, node id and volume as custom properties of each object;
* reads the manifest again every second: a proxy built again changes its mesh only (its object, its
  material from EMtools, its place in EMtools' lists stay), a proxy gone is removed;
* **never overwrites what was changed in Blender**: an object whose vertices are not those imported
  is *detached*, and the panel (*PPG* in the sidebar of the 3D view) lists it;
* loads the em.json into EMtools 1.6 (*Carica l'em.json in EMtools*, and by itself on a new scene),
  which pairs the proxies with the units by name.

With *Aggiornamento dal vivo* the program writes the EM project again, in the background, every
time the proxies, the model or the em.json change. Blender is found where it is installed, or where
*Scegli il programma Blender...* or `PPG_BLENDER` say.

## EMtools live

EMtools 1.6 hosts the live sync of the Extended Matrix: start it in Blender (EM panel, *Sync*), then
tick *Sincronizza con EMtools* in the pane. The program connects to `127.0.0.1:8788` (another port
in the settings of the plugin, `port`), says which matrix it has open (`client_info`), asks EMtools'
graph (`snapshot`: with no em.json linked the pane and the names of the proxies follow it), and
exchanges the selection (`select`). It never sends graph edits or commands: the matrix belongs to
EMtools and EMStudio. It connects again by itself when Blender restarts. EMtools 1.5 has no sync:
the files and the live script work the same.

## Use

* *Plugin > Blender ed EMtools*: the pane, *Collega l'em.json del sito...*, *Esporta nel progetto
  EM*, *Apri in Blender (aggiornato dal vivo)*, *Scegli il programma Blender...*; the *Progetto EM
  (Blender)* item under the chunk in the workspace.
* Command line:

  ```bash
  ppg plugin blender-em status scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --out Scavo --no-survey
  ppg plugin blender-em open scavo.ppg            # the em.json linked to the chunk
  ppg plugin blender-em blender                   # the Blender programs found
  ```

Nothing beyond what the program carries: the em.json is read as JSON, the WebSocket is spoken with
the standard library, the live script runs in Blender's own Python.

## Tests

```bash
pip install -e .
pytest -q tests
PPG_TEST_BLENDER=/path/to/blender pytest -q tests/test_blender_live.py   # or a Python with bpy
```

The em.json of [s3dgraphy](https://pypi.org/project/s3dgraphy/) 1.6 is read in a test when it is
installed; the sync runs against a host that speaks EMtools' wire.

Licence: GPL-2.0-or-later, like the program the plugin runs in. Author: Enzo Cocca.
