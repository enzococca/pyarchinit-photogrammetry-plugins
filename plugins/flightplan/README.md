# Flight plans

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.4 or later):
drone flight plans designed before going to the field, and checked after the flight.

- **Design**: the area (drawn on the 2D view, from a layer, a KML/GeoJSON/GeoPackage file or coordinates, with a
  margin), the camera (a list of drone cameras, or one from a photo's EXIF, from the chunk's calibrated sensor, or typed
  by hand - sensor values are never invented), a goal (orthophoto, DEM, full 3D model, close inspection, quick survey,
  GSD by hand) or a height, overlaps, speed, direction, double grid, obliques, a shot per point or by interval. The plan:
  blocks and legs that fit a battery, shots, minutes, batteries, GSD, warnings.
- **Terrain following** on the project DEM (the public AWS terrain tiles where it ends, or instead of it): every point
  its own height above the takeoff, per-leg takeoffs corrected by their ground.
- **Predicted coverage** on the chunk's DEM: photos per cell and GSD per cell, low-coverage cells, an RGBA map on the
  2D view. No occlusion.
- **GNSS targets** from a CSV (degrees, UTM, any EPSG) or from the chunk's markers, as descents with a ladder of photos.
- **Exports** for the drones and the GIS, a flight sheet in Excel and HTML, one map per leg over the chunk's
  orthophoto, the DJI Fly waypoint folder filled with the legs.
- **After the flight**: the planned shots against the photos of the chunk (matched, missing, extra, repeated, offsets,
  height difference from DJI's XMP) and the coverage the aligned photos give against the predicted one.

Everything is in the **Flight plans** pane (docked on the right, nine steps with their state), in a group of the tool
rail of the 2D view, in the workspace tree and on the command line; *File > Export* has five of the formats (the DJI
KMZ, QGroundControl, Litchi, GeoPackage and the Excel sheet). The user guide is on the
[program's site](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/flightplan/).

The planner - serpentine grids, rings, routes given point by point, legs that fit one battery, the camera profiles, the
DJI KMZ and the DJI Fly copy - is ported from the author's Photo3D planner. A plan JSON of that planner loads as it is.

> **Airspace, permits, obstacles and weather are the pilot's.** The plugin computes a route; checking that the flight
> is allowed and safe is the pilot's job. Look at the route in the flight app before taking off.

## Heights in the files

All drone files carry heights **above the takeoff point**, one per waypoint (a plan that follows the terrain keeps its
per-point heights in every drone format), and each file declares it:

| Format | Reference written in the file |
|---|---|
| DJI WPML KMZ (DJI Pilot 2, FlightHub 2, DJI Fly) | `heightMode` / `executeHeightMode` `relativeToStartPoint`, `coordinateMode` `WGS84`, `executeHeight` per point |
| QGroundControl `.plan` | items frame 3 (`MAV_FRAME_GLOBAL_RELATIVE_ALT`), `AltitudeMode` 1, `globalPlanAltitudeMode` 1 |
| Mission Planner `.waypoints` | frame 3; home row at the takeoff with altitude 0 |
| Parrot `.mavlink` | frame 3 |
| Litchi CSV | `altitudemode` 0 (above the takeoff) |
| KML | `altitudeMode` `absolute` = takeoff ground + height above the takeoff; takeoff height and its source written in the file (0 with a note when unknown) |
| GeoJSON | z = height above the takeoff, `riferimento_quote` = `decollo` |
| GeoPackage (chunk's system) | `quota_decollo_m` above the takeoff; Z = takeoff ground + it, in the DEM's height system; `riferimento_quote` column |

The drone never reads the KML, GeoJSON or GeoPackage.

## Proven and not proven

- **Proven on the real drone**: the DJI Fly KMZ on the DJI Mini 4 Pro (the planner's file).
- **Not yet imported in the real apps**: the WPML KMZ in DJI Pilot 2 and FlightHub 2, Litchi, QGroundControl, Mission
  Planner, Parrot, and the area KML for the DJI apps (experimental). In particular the per-point heights of
  QGroundControl, Mission Planner, Parrot and Litchi from a terrain-following plan have not been tried in those apps.
  The files follow each format's specification and are checked by the tests: **try them in the real app before
  flying.**
- **Not yet tried on a real controller**: the copy into DJI Fly on Windows through MTP and PowerShell.
- **Not yet tried on a real flight**: the coverage obtained after the flight (tested on synthetic data).

## Limits

- No occlusion in the coverage, predicted or obtained (walls and trees do not hide the ground); cells outside the DEM
  are flat at the takeoff; cells outside the reconstruction are not measured.
- Without an orthophoto in the project the maps of the legs are on grey (no third-party map tiles).
- A chunk in a local system: area only from a file or from degrees, no drawing on the view, no coverage.
- Close-up shooting (a trench, a wall face) is not in 0.1.0.
- The comparison and the coverage become "To do again" when the plan or the photos change, after a new alignment, a new
  DEM or new positions.

## Command line

```
ppg plugin flightplan cameras
ppg plugin flightplan design PROJECT.ppg --bbox W,S,E,N --goal orto --value 2 [--terrain] [--takeoff LAT,LON]
ppg plugin flightplan show PROJECT.ppg
ppg plugin flightplan terrain PROJECT.ppg [--detail fine] [--tiles] [--off]
ppg plugin flightplan targets PROJECT.ppg --csv targets.csv [--epsg N]      # or --markers
ppg plugin flightplan coverage PROJECT.ppg [--cell M] [--low N]
ppg plugin flightplan export PROJECT.ppg --format kmz-zip --out volo.zip
ppg plugin flightplan djifly PROJECT.ppg --zip pronte.zip --missions UUID,UUID
ppg plugin flightplan compare PROJECT.ppg [--plan-id ID] [--max-distance M]
ppg plugin flightplan list PROJECT.ppg                                      # use ID, remove ID
ppg plugin flightplan design --coords "lat,lon;lat,lon;lat,lon" --plan piano.json   # without a project
```

Formats: `kmz`, `kmz-zip`, `djifly-zip`, `plan`, `waypoints`, `mavlink`, `litchi`, `kml`, `geojson`, `gpkg`,
`area-kml`, `sheet-xlsx`, `sheet-html`, `maps-zip`. The plans of a project are kept in its chunk; the coverage, the
comparison and the DJI Fly backups in `<chunk data>/flightplan/`. Every export first recomputes the heights of the
blocks that follow the terrain and changed since. The network is used only for the public terrain tiles, on request.

Licence: GPL-2.0-or-later.
