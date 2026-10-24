# Mesh (MeshLab)

*[Italiano](README.it.md)*

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 or later): the filters of [MeshLab](https://www.meshlab.net) on the model of the chunk,
through [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) (Visual Computing Lab, ISTI-CNR).
Every result is a **new version of the model**: the one before stays among the versions of the
chunk (*Versioni* in the workspace), to compare them or to go back.

| Operation | What MeshLab does | The texture |
|---|---|---|
| **Decimate** | Quadric edge collapse to a number or a fraction of the faces; borders and topology kept by default; optional planar simplification, quality threshold of the triangles | kept: MeshLab's variant that carries the texture coordinates of the face corners (several texture pages too) |
| **Remesh** | Isotropic explicit remeshing: regular triangles of one edge length (by default the mean edge of the model), sharp edges kept above an angle, adaptive on request, a largest distance from the surface before | to build again |
| **Smooth** | Taubin (λ 0.5, μ -0.53, 10 steps: no shrinking) or HC Laplacian (1 step); the borders stay where they are (on request they move too) | kept |
| **Close holes** | Holes up to a number of border edges closed with new faces (no self-intersection, optional refinement to an edge length); the outline of each separate part - its longest border - stays open unless you ask (the bottom of an object): MeshLab alone would cap a whole excavated surface | to build again, when holes were closed |
| **Repair the topology** | Duplicate vertices and faces, faces without area, non-manifold edges (faces removed or vertices split), non-manifold vertices (split), unused vertices | kept |
| **Measure** | Surface area, volume of a closed (watertight) model, 2-manifold or not, separate parts, border loops, border and non-manifold edges, non-manifold and unused vertices, size, mean edge | - |

Lengths and measures are in metres of the survey (in units of the model when the chunk has no
scale). Before MeshLab sees the model, the vertices it repeats at the same position - a textured
model has them along the seams of its texture - are counted once: otherwise MeshLab takes every
seam for a border, the smoothing opens it and the measures count it as a hole. The properties
MeshLab cannot carry (the confidence of the vertices, any other) come back from the vertex of the
model each new vertex is nearest to; the colours are MeshLab's, interpolated on new vertices.
When MeshLab changes nothing (no hole that small, nothing to repair) the model stays as it is and
no version is added. The new version is named after what MeshLab did (`MeshLab: decimazione a
200.000 facce (texture conservata)`, stored in Italian as the program's own names), or as you name
it; the project is saved with it, as the program does after its own processing steps.

What the program already does stays the program's: the mesh from the dense cloud or the depth maps
(Poisson), the cleaning of the model by confidence, detached pieces and stretched triangles, the
cut with planes, the texture.

## Where

* the **Mesh (MeshLab)** pane of the project dock: the active model, the operations (their
  parameters in a dialog), the measures, *Installa MeshLab...*; the result is shown in the 3D view;
* *Plugin > Mesh (MeshLab)* in the menu, the same operations;
* the command line:

```bash
ppg plugin mesh-tools status
ppg plugin mesh-tools install                       # the environment of the plugin (uv)
ppg plugin mesh-tools decimate scavo.ppg --faces 200000
ppg plugin mesh-tools decimate scavo.ppg --fraction 0.25 --planar --name "Per il web"
ppg plugin mesh-tools remesh scavo.ppg --edge 0.01
ppg plugin mesh-tools smooth scavo.ppg --method hc --steps 2
ppg plugin mesh-tools close-holes scavo.ppg --max-edges 100
ppg plugin mesh-tools close-holes vaso.ppg --max-edges 5000 --close-outline   # the bottom of an object too
ppg plugin mesh-tools repair scavo.ppg --split
ppg plugin mesh-tools measure scavo.ppg --json
```

`--chunk LABEL` works on another chunk than the active one, `--overwrite` keeps no version of the
model before, `--json` prints the whole report.

## How MeshLab runs

pymeshlab carries its own native libraries (Qt among them), and some filters crash on a model they
do not accept (the remeshing of a non-manifold model is a segmentation fault): the program never
imports it. Each operation runs in a process of its own (`ppg_mesh_tools/worker.py`): the plugin
writes the model for it in a work folder beside the model, the worker applies MeshLab's filters
and writes the result, and the two speak through lines of JSON (progress, result, error);
cancelling stops it at once, a crash leaves the model as it was and says so. The worker runs with:

* the Python of the program, when pymeshlab is installed there (`pip install pymeshlab`);
* otherwise an environment the plugin installs on request with
  [uv](https://github.com/astral-sh/uv) - Python 3.12 and pymeshlab from PyPI, about 300 MB, in
  the plugin's folder (`PPG_MESH_TOOLS_ENV` to put it elsewhere): this is how the packaged
  application works, which has no pip. The pane offers it the first time.

pymeshlab has builds for macOS (Apple Silicon and Intel), Linux (x86_64 and aarch64) and Windows
x64; not for Windows on ARM, where the plugin says it is not available.

On Linux pymeshlab needs system libraries its wheels do not carry, `libOpenGL.so.0` above all:
without it it imports but reads no file. Install them once:

```bash
sudo apt install libopengl0 libgl1                 # Debian, Ubuntu
sudo dnf install libglvnd-opengl libglvnd-glx      # Fedora
```

Before the first operation of a session the plugin tries MeshLab on a triangle (`ppg plugin
mesh-tools status` does the same) and, when something is missing, says which library and which
package has it.

## Installing

* In the program: *Strumenti > Plugin...*, `mesh-tools`, *Installa / aggiorna*, *Riavvia ora*.
* With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-mesh-tools @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/mesh-tools"
  pip install pymeshlab        # optional: else the plugin installs its own environment
  ```

Tests: `pip install -e . pytest` in the environment of the program, then `pytest tests`; the tests
of MeshLab itself (`tests/test_meshlab.py`) run when pymeshlab is installed for that Python, the
others use a worker that speaks the same protocol without it.

Licence: **GPL-3.0-or-later** (pymeshlab is GPL-3; the program and the other plugins of this
repository are GPL-2.0-or-later, which allows it). Author: Enzo Cocca.
