# Geophysics

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/geophysics/)**: the files to export, how to place a grid on the pegs, the filters, how to read the results.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 or later): the grids of a geophysical survey - magnetometry, resistivity, GPR time slices,
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
* **Overlay** on the orthomosaic or the DEM of the raster view (or, with nothing shown, on a frame
  the size of the grid), with the band (the slice of a GPR survey), the palette (greys, inverted
  greys, diverging, earth, heat) and the opacity.
* **Anomalies**: *Draw anomaly* draws a polygon on the view (click the vertices, double click or
  Enter to close) with the grid, the band and a note, in the layer *Anomalie geofisiche*; kept in
  the project, deleted with the program's *Select* tool.
* **Exports**: GeoTIFF (all the bands) and Surfer GRD (the band shown; a turned grid resampled
  north up) of a grid, a GeoPackage with the outlines of the grids and the anomalies (*File >
  Esporta*), the grids in the QGIS project of the program with their ramp.
* `ppg plugin geophysics import|list|georef|filter|export PROJECT.ppg ...` from the command line.

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import survey.ppg survey.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef survey.ppg "Area 1" --origin 500100.0,4500800.0 --azimuth 95
ppg plugin geophysics filter survey.ppg "Area 1" destripe --copy
ppg plugin geophysics export survey.ppg "Area 1 destripe" area1.tif
```

Licence: GPL-2.0-or-later.
