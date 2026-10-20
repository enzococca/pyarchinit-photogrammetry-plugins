# Spirula

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) that trains **3D Gaussian splats** of the aligned photos with
[Spirula Studio](https://github.com/harry7557558/spirula-studio) (GPL-3): one executable, Vulkan
compute on NVIDIA, AMD, Intel and Apple GPUs, no Python, CUDA or COLMAP to install.

| Where | What |
|---|---|
| *Plugin > Spirula > Addestra Gaussian splatting...* | The COLMAP model of the chunk (text, `sparse/0`), its photos (linked, any camera model: Spirula undistorts by itself), its masks and its dense cloud as seed points go to `spirula train`, with a preset (`3dgs`, `in-the-wild`, `centered-object`, `360-camera`, `hdr`, `synthetic`, `meshing`), a quality (15 000 to 80 000 iterations), a reduction of the photos (suggested for about 4 megapixels each), the spherical harmonics degree and, if wanted, Spirula's web viewer while it trains. The run is in `<chunk data>/spirula/runs/<name>/`; `splat.ply` (the 3DGS PLY every viewer reads) becomes the product `splat` of the chunk. Progress and cancel in the window. |
| *Plugin > Spirula > Mesh dallo splat...* | `spirula mesh`: PLY, OBJ, glTF/GLB or STL (per-vertex colour or texture) beside the splat; a PLY also goes among the versions of the model (*Versioni*). |
| *Plugin > Spirula > Apri lo splat in Spirula Studio* | the splat in the viewer of Spirula Studio |
| *Plugin > Spirula > Installa Spirula Studio...* | the latest release of GitHub for this computer (macOS on Apple Silicon: the dmg, into `/Applications`; Windows and Linux x86_64: the zip with the executable, into `%LOCALAPPDATA%\Programs\Spirula Studio` or `~/.local/share/spirula-studio`), downloaded, checked with the SHA-256 GitHub publishes and installed. Offered by itself the first time a step needs Spirula and it is missing. |
| *Plugin > Spirula > Dove sta Spirula Studio...* | the executable: found by itself on the PATH, in `/Applications/Spirula Studio.app`, in the usual folders, or given here (also `PPG_SPIRULA`) |
| *File > Esporta > Splat (PLY)* | a copy of the splat |

From the command line:

```bash
ppg plugin spirula install [--dir FOLDER]                  # the release for this computer
ppg plugin spirula where [--set /path/to/spirula]
ppg plugin spirula train scavo.ppg --preset in-the-wild --quality high --divisor 2 --viewer
ppg plugin spirula train scavo.ppg --quality low --no-masks --name prova -- --cap-max 500000
ppg plugin spirula mesh scavo.ppg --format ply,glb --color texture
ppg plugin spirula open scavo.ppg
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

* The splat is in the internal frame of the chunk (the frame of the COLMAP model): to place it in
  the world apply `chunk.transform`, as for the dense cloud.
* Spirula trains at the resolution of the photos it is given: 48-megapixel photos at full
  resolution need a lot of memory and time; the suggested reduction brings them to about
  4 megapixels. The quality sets Spirula's `--quality` macro (iterations and the cap of splats).
* Spirula prints its progress every hundred steps; the plugin reads it (in English, `--lang en`)
  and ends the process when the step is cancelled.
* Tests use a stand-in for `spirula` (no GPU needed): `pip install -e . pytest` in the
  environment of the program, then `pytest tests`.

Licence GPL-2.0-or-later. Author: Enzo Cocca. Spirula Studio is a separate program under the
GPL-3; the plugin only runs it.
