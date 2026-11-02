# Masonry

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 or later):
walls, buildings and stone towers recorded by their faces.

* **Faces**: a wall (a building, a tower) is recorded by its faces, each an elevation of the program - on the vertical
  plane through two points traced on the plan (`v` is the height), on a cylinder for a round tower, a well or an apse
  (three points on the plan, or the circle fitted to a horizontal slice of the model; seen from outside or inside), or
  one plane per side of a polygonal tower (the slice's outline simplified into a polygon). The faces are listed with
  their size, resolution and the drawings they hold.
* **Shape**: on the model's surface within the face's window, the plane (or the cylinder) fitted by least squares with
  Huber's weight; the **lean** (degrees, cm/m, the azimuth its top goes towards, towards the viewer or away); the
  **bulge** map - the distance of the surface from the fit, signed towards the viewer, in cm, on the elevation's grid -
  shown over the elevation with a diverging palette and a legend; **horizontal sections** at chosen heights stacked in
  one DXF and one SVG, with the offsets between them at chosen stations; the **axis of a round tower** from the circles
  fitted at several heights (inclination, direction, out-of-round per height); **two surveys** of the same wall (two
  chunks already aligned) compared on the face: the difference as a raster, its median and 90th percentile.

The measures say how a wall stands; they are not a verdict on it. The plugin writes only under
`<chunk data>/masonry/`; the program's elevations keep their format.

Command line: `ppg plugin masonry face|shape|list|remove`.

License: GPL-2.0-or-later.
