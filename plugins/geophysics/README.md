# Geophysics

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 or later): the grids of a geophysical survey - magnetometry, resistivity, GPR time slices,
electromagnetic - beside the photogrammetric products of the same ground.

* **Import**: Surfer GRD (ASCII and binary, Surfer 6 and 7), ASCII XYZ / CSV (`x y value`, the
  exports of Geoplot, Bartington, Terrasurveyor), GeoTIFF and any raster, a folder of rasters or
  images as the slices of a GPR survey. Every grid becomes a float32 GeoTIFF in the project
  (`<chunk data>/geophysics/`).
* **Georeferencing** of a grid surveyed in local metres: the corners of the survey grid as markers
  placed on the orthomosaic, or an origin and an azimuth, or typed pairs of local and world
  coordinates (a similarity by least squares, with the residual of every point).
* **Filters**: despike, destripe (zero-mean traverse), clip, low-pass, high-pass (the residual of
  a Gaussian smoothing: the small anomalies over the geology), fill of the gaps, resample - into a
  copy or in place, every filter recorded.
* **Overlay** on the orthomosaic or the DEM of the raster view, with the band (the slice of a GPR
  survey), the palette (greys, inverted greys, diverging, earth, heat) and the opacity; the
  anomalies drawn with the polygon tool of the program in the layer *Anomalie geofisiche*.
* **Exports**: GeoTIFF and Surfer GRD of a grid, a GeoPackage with the outlines of the grids and
  the anomalies (*File > Esporta*), the grids in the QGIS project of the program with their ramp.
* `ppg plugin geophysics import|list|georef|filter|export PROJECT.ppg ...` from the command line.

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import survey.ppg survey.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef survey.ppg "Area 1" --origin 438100.0,4553800.0 --azimuth 95
ppg plugin geophysics filter survey.ppg "Area 1" destripe --copy
ppg plugin geophysics export survey.ppg "Area 1 destripe" area1.tif
```

Licence: GPL-2.0-or-later.
