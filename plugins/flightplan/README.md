# Flight plans

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.4 or later):
drone flight plans designed before going to the field.

Work in progress (0.1.0 not released yet). The planner - serpentine grids, rings, routes given point by point,
legs that fit one battery, the camera profiles - is ported from the author's Photo3D planner.

## Command line

```
ppg plugin flightplan cameras
ppg plugin flightplan design PROJECT.ppg --bbox W,S,E,N --goal orto --value 2 [--terrain] [--takeoff LAT,LON]
ppg plugin flightplan show PROJECT.ppg
ppg plugin flightplan coverage PROJECT.ppg [--cell M] [--low N]
ppg plugin flightplan export PROJECT.ppg --format kmz-zip --out volo.zip
ppg plugin flightplan djifly PROJECT.ppg --zip pronte.zip --missions UUID,UUID
ppg plugin flightplan design --coords "lat,lon;lat,lon;lat,lon" --plan piano.json   # without a project
```

The plans of a project are kept in its chunk (`list`, `use ID`, `remove ID`); the predicted coverage (photos per
cell and GSD on the chunk's DEM) is written into the chunk's folder. Every export first recomputes the heights of
the blocks that follow the terrain and changed since.

Licence: GPL-2.0-or-later.
