# Spirula

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/spirula/)**: the photos to prepare, the training and its presets, the mesh, how to read the result.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) that trains **3D Gaussian splats** of the aligned photos with
[Spirula Studio](https://github.com/harry7557558/spirula-studio) (GPL-3): one executable, Vulkan
compute on NVIDIA, AMD, Intel and Apple GPUs, no Python, CUDA or COLMAP to install.

| Where | What |
|---|---|
| *Plugin > Spirula > Addestra Gaussian splatting...* | The photos of the chunk must be aligned (the window, the command line and the training refuse a chunk that is not). The COLMAP model of the chunk (text, `sparse/0`), its photos (linked, any camera model: Spirula undistorts by itself), its masks and its dense cloud as seed points go to `spirula train`, with a preset (`3dgs`, `in-the-wild`, `centered-object`, `360-camera`, `hdr`, `synthetic`, `meshing`), a quality (15 000 to 80 000 iterations), a reduction of the photos (suggested for about 4 megapixels each), the spherical harmonics degree and, if wanted, Spirula's web viewer while it trains (the browser opens on it, `http://localhost:7007/` unless another port is given, as soon as Spirula starts it). The run is in `<chunk data>/spirula/runs/<name>/`; `splat.ply` (the 3DGS PLY every viewer reads) becomes the product `splat` of the chunk. Progress (in the language of the program) and cancel in the window. |
| *Plugin > Spirula > Mesh dallo splat...* | `spirula mesh`: PLY, OBJ, glTF/GLB or STL (per-vertex colour or texture) beside the splat; a PLY also goes among the versions of the model (*Versioni*). |
| *Plugin > Spirula > Apri lo splat in Spirula Studio* | the splat in the viewer of Spirula Studio |
| *Plugin > Spirula > Installa Spirula Studio...* | the latest release of GitHub for this computer (macOS on Apple Silicon: the dmg, into `/Applications`; Windows and Linux x86_64: the zip with the executable, into `%LOCALAPPDATA%\Programs\Spirula Studio` or `~/.local/share/spirula-studio`), downloaded, checked with the SHA-256 GitHub publishes and installed. Offered by itself the first time a step needs Spirula and it is missing. |
| *Plugin > Spirula > Dove sta Spirula Studio...* | the executable: found by itself on the PATH, in `/Applications/Spirula Studio.app`, in the usual folders, or given here (also `PPG_SPIRULA`) |
| *File > Esporta > Splat (PLY 3D Gaussian Splatting)* | the splat in the survey coordinates of the chunk (its CRS, or its local frame): positions, rotations, scales and spherical harmonics moved from the frame Spirula trained it in; positions as doubles when the coordinates are large (projected systems) |
| *File > Esporta > Splat (PLY, sistema dell'addestramento)* | the splat as Spirula wrote it, in the frame of its training |

From the command line:

```bash
ppg plugin spirula install [--dir FOLDER]                  # the release for this computer
ppg plugin spirula where [--set /path/to/spirula]
ppg plugin spirula train scavo.ppg --preset in-the-wild --quality high --divisor 2 --viewer   # web viewer in the browser (--port, --no-browser)
ppg plugin spirula train scavo.ppg --quality low --no-masks --name prova -- --cap-max 500000
ppg plugin spirula mesh scavo.ppg --format ply,glb --color texture
ppg plugin spirula open scavo.ppg
ppg plugin spirula export scavo.ppg splat.ply                    # in the survey coordinates
ppg plugin spirula export scavo.ppg splat_spirula.ply --frame training
```

## Installing

* Spirula Studio: *Plugin > Spirula > Installa Spirula Studio* (or `ppg plugin spirula install`)
  fetches the release of your system from
  [its releases page](https://github.com/harry7557558/spirula-studio/releases) (Windows and Linux
  x86_64: a zip with `spirula`; macOS on Apple Silicon: `Spirula Studio.app`, with MoltenVK
  inside); you can also install it yourself and tell the plugin where it is. Intel Macs and ARM
  Windows or Linux have no build.
* The plugin, in the program: *Strumenti > Plugin...*, `spirula`, *Installa / aggiorna*,
  *Riavvia ora*. With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-spirula @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/spirula"
  ```

## Notes

* Spirula recentres and reorients the scene before training: the splat is in that frame, and
  `chunk.meta["splat"]["world_from_train"]` (from the `scene_transform.json` of the run) is the
  4 x 4 matrix that brings it back into the frame of the COLMAP model, the internal frame of the
  chunk; `chunk.transform` then places it in the world, as for the dense cloud. The export in
  survey coordinates applies both (`chunk.transform @ world_from_train`, a similarity `s R x + t`)
  to every splat: positions, rotations (`rot_0..3`, `w` first), log-scales (`scale_* + log s`) and
  the spherical harmonics of every degree in the file (`f_rest_*`, turned so that the colour seen
  along `R d` is the one seen along `d`; `f_dc_*` and the opacity stay). A run without
  `scene_transform.json` can only be exported in the frame of the training. Positions beyond
  10 km are written as doubles (a 32-bit float keeps 25 cm steps at 4 000 km).
* Spirula trains at the resolution of the photos it is given: 48-megapixel photos at full
  resolution need a lot of memory and time; the suggested reduction brings them to about
  4 megapixels. The quality sets Spirula's `--quality` macro (iterations and the cap of splats).
* Spirula prints its progress every hundred steps; the plugin reads it (in English, `--lang en`)
  and ends the process when the step is cancelled.
* Tests use a stand-in for `spirula` (no GPU needed): `pip install -e . pytest` in the
  environment of the program, then `pytest tests`.

Licence GPL-2.0-or-later. Author: Enzo Cocca. Spirula Studio is a separate program under the
GPL-3; the plugin only runs it.
