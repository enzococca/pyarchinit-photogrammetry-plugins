# Geophysics

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/geophysics/)**: the files to export, how to place a grid on the pegs, the filters, how to read the results.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 or later): the grids of a geophysical survey - magnetometry, resistivity, GPR time slices,
electromagnetic - beside the photogrammetric products of the same ground.

* **Import**: Surfer GRD (ASCII and binary, Surfer 6 and 7), ASCII XYZ / CSV (`x y value`, the
  exports of Geoplot, Bartington, Terrasurveyor; decimal commas with semicolons, tabs or spaces
  between the columns; the dummy values 2047.5, -9999, 32767, -32768, Surfer's blank and one of
  your own as no reading; readings closer along the traverses than the traverses are apart give a
  grid of the finer spacing with the rows between two traverses interpolated, or the readings
  averaged in a cell of your own, the cells centred on the readings), GeoTIFF and any raster, a
  folder of rasters or images as the slices of a GPR survey (in the natural order of their names).
  Every grid becomes a float32 GeoTIFF in the project (`<chunk data>/geophysics/`).
* **Georeferencing** of a grid surveyed in local metres: the corners of the survey grid as markers
  placed on the orthomosaic, or an origin and an azimuth, or typed pairs of local and world
  coordinates (a similarity by least squares, with the residual of every point); placed again, a
  grid starts from its local metres: the new placement replaces the old one.
* **Filters**: despike, destripe (zero-mean traverse), clip, low-pass (sigma 1 cell), high-pass
  (the residual of a Gaussian smoothing of sigma 10 cells: the small anomalies over the geology),
  fill of the gaps, resample - into a copy or in place, every filter recorded; the same defaults in
  the window and on the command line.
* **Overlay** on the orthomosaic or the DEM of the raster view (or, when the chunk has neither, on
  an empty frame the size of the grid), with the band (the slice of a GPR survey), the palette
  (greys, inverted greys, diverging, earth, heat) and the opacity; it stays over the next raster the
  view shows, until *Hide*.
* **Anomalies**: *Draw anomaly* draws a polygon on the view with the program's drawing, lent to
  the plugin (drag a box, or click the vertices and double click or Enter to close), with the grid,
  the band and a note, in the layer *Anomalie geofisiche* (drawn again on every raster the view
  shows); kept in the project, deleted with the program's *Select* tool.
* **Exports**: GeoTIFF (all the bands) and Surfer GRD (the band shown; a turned grid resampled
  north up) of a grid, a GeoPackage with the outlines of the grids and the anomalies (*File >
  Esporta*), the grids in the QGIS project of the program with their ramp.
* `ppg plugin geophysics import|list|georef|filter|export PROJECT.ppg ...` from the command line.

## Electrical resistivity tomography (ERT, 0.2.0)

The *ERT* tab of the pane takes the readings of a resistivity meter to a resistivity section in the
survey, inverted with [pyGIMLi](https://www.pygimli.org) (Apache-2.0):

1. **Readings**: a CSV/TXT table (`a, b, m, n` and `rhoa`, or `r`, or `u` and `i`; `err` when
   present; Syscal's `Spa.1-4`, `Rho`, `Dev`, `Vp`, `In` too; electrode numbers from 1 or positions in
   metres), pyGIMLi's unified data format (`.ohm`, `.dat`), and through pyGIMLi the other files of the
   instruments (RES2DINV, ABEM, Syscal); an optional table of the electrodes (`x, z` along the line or
   world `x, y, z`). *Synthetic profile* makes readings of a two-layer model with a block.
2. **Line in the survey**: drawn on the orthomosaic or DEM, from two markers or typed coordinates (two
   points or more: a bent line), or from the electrode coordinates of the file when they are in the
   system of the chunk; the first electrode's distance and the spacing, measured along the ground (the
   cable on a slope) or in plan; the heights of the electrodes from the DEM of the chunk, from the file
   or flat.
3. **Pseudosection and filters**: the readings as points in the 2D view (*Remove readings...*: draw an
   area around the bad ones); filters on the range, the error, the reciprocals and the spikes.
4. **Inversion**: pyGIMLi's ERT manager on a mesh that follows the topography (geometric factors
   computed on it), lambda, vertical weight, blocky model, robust data, relative and voltage errors;
   chi² and RMS; the parts the readings do not resolve blanked by the coverage or by the
   depth-of-investigation index of Oldenburg and Li.
5. **Section**: in the 2D view in its own frame (distance along the line x elevation, log colours
   with the legend in ohm m), in the 3D view as a vertical section under the line with the ground cut
   away on one side, the program's proxies of the units, the units and the cores of the
   geoarchaeology plugin; the line and the
   electrodes on the plan.
6. **Exports**: the section as GeoTIFF (u, v frame, with `.frame.json`), the cells of the mesh as CSV,
   the pixels as XYZ with world coordinates, figures PNG/SVG/PDF of the section and the
   pseudosection, the line and the electrodes as GeoPackage, the paradata (parameters, fit, versions)
   as JSON, the readings kept as `.ohm`.

pyGIMLi never runs in the program's process: *Install pyGIMLi* makes an environment of its own with
uv in the plugin's data folder (about 80 MB, plus about 60 MB for Python when uv has not it yet; `PPG_GEOPHYSICS_ENV` puts it elsewhere) and every
reading or inversion runs there, in a worker process. pyGIMLi 1.6 has builds for Windows x64, Linux
x86_64 and macOS 26 or later (Apple Silicon and Intel), not for Windows or Linux on ARM.

```bash
ppg plugin geophysics ert install
ppg plugin geophysics ert import survey.ppg line1.csv --name "Line 1" --spacing 1
ppg plugin geophysics ert place survey.ppg "Line 1" --markers E1,E48 --heights dem
ppg plugin geophysics ert filter survey.ppg "Line 1" --spikes 8 --max-error 5
ppg plugin geophysics ert invert survey.ppg "Line 1" --lam 20 --doi oldenburg
ppg plugin geophysics ert export survey.ppg "Line 1" exports/line1
```

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import survey.ppg survey.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef survey.ppg "Area 1" --origin 500100.0,4500800.0 --azimuth 95
ppg plugin geophysics filter survey.ppg "Area 1" destripe --copy
ppg plugin geophysics export survey.ppg "Area 1 destripe" area1.tif
```

Licence: GPL-2.0-or-later.
