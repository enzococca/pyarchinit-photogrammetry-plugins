# Masonry

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 or later):
walls, buildings and stone towers recorded by their faces.

* **Faces**: a wall (a building, a tower) is recorded by its faces, each an elevation of the program - on the vertical
  plane through two points traced on the plan (`v` is the height), on a cylinder for a round tower, a well or an apse
  (three points on the plan, or the circle fitted to a horizontal slice of the model; seen from outside or inside), or
  one plane per side of a polygonal tower (the slice's outline simplified into a polygon); or an elevation the
  program already made (*Existing elevation...*, `--existing`), taken as it is with its patches and drawings, also one
  developed on the tower's axis. The faces are listed with their size, resolution and the drawings they hold.
* **Shape**: on the model's surface within the face's window, the plane (or the cylinder) fitted by least squares with
  Huber's weight; the **lean** (degrees, cm/m, the azimuth its top goes towards, towards the viewer or away); the
  **bulge** map - the distance of the surface from the fit, signed towards the viewer, in cm, on the elevation's grid -
  shown over the elevation with a diverging palette and a legend; **horizontal sections** at chosen heights stacked in
  one DXF and one SVG, with the offsets between them at chosen stations; the **axis of a round tower** from the circles
  fitted at several heights (inclination, direction, out-of-round per height); **two surveys** of the same wall (two
  chunks already aligned) compared on the face: the difference as a raster, its median and 90th percentile.

* **Stones and courses**: on the stones drawn on a face (the program's Stones tool on the face's elevation), per stone
  its width, height, area and best-fit ellipse (elongation, orientation) and rectangularity; the mortar share (1 - the
  stones' union over the wall's area: the face's bounds or a polygon drawn on it; a stone drawn twice counts once);
  the courses (each stone joined to its neighbour on the right sharing at least half its height: sloping courses stay
  courses; mean height, count, thickness, slope); tables in CSV and Excel, a GeoPackage, the measures added to the
  program's export of the face. Stones smaller than four pixels of the face are left out and named.
* **Courses and joints from a classified drawing** (0.1.3, with the program 1.7.7 or later; the button is hidden on
  an older program): the lines drawn on the face (or a GeoPackage, GeoJSON or DXF in its `(u, v)`) are classified by
  the program along the face's horizontal - lines along it are the **beds**, a pair of lines along it (also a band
  traced as a polygon) a **course** of known thickness, lines across it the **joints**. A course lies between a bed
  and the bed directly above it; its stones are the cells split by the joints that cover at least half of its height.
  A check plate shows the courses and the stones proposed; only then are the stones added to those of the face (or put
  in their place), skipping those that overlap a stone already there by more than half - taking the same drawing
  again adds nothing. One step of *Annulla*. The outlines of the face's USM are left out unless asked.
* **Type suggested**, never a verdict: *coursed* when at least 70 % of the stones sit in courses of 3 or more stones
  sloping less than 2°; *regular* when the courses' thickness varies little (median absolute deviation / median < 20 %);
  *squared* when the stones' median rectangularity is over 0.8, *elongated* when their median elongation is 2 or more.
* **USM**: the units drawn on the face; relationships proposed by their geometry - *si appoggia a* (a unit above
  another touching it along at least 20 % of its width), *taglia* (a unit that interrupts a course of another),
  *copre* (a unit above another along its whole top and width), *riempie* (a unit enclosed by another: a blocked
  opening) - each with its evidence, written only when the user ticks it; the rules read shapes and cannot tell a
  cut from a straight-joint abutment;
  *Salva in PyArchInit*: the US forms, the relationships both ways and the face's drawings as media.

* **Outputs**: per face a plate at scale (SVG and PDF: the orthophoto at the largest standard scale that fits A3, the
  stones, courses, USM, sections and bulge contours, a scale bar and a title block with the measures), a DXF with the
  layers PIETRE, CORSI, USM, SEZIONI, BORDO, QUOTE, the GeoPackages, the tables and the rasters; for the wall an IFC4
  `IfcWall` - the slab between two surveyed faces (looking within 15° of opposite ways, less than 3 m apart), or
  behind each other face by a thickness given (each side of a polygonal tower) - with
  `Pset_PyArchInit_Masonry` and the USM as `IfcBuildingElementPart`; the face opened in QGIS, unfolded.
  From 0.1.2 the wall and its USM keep their GlobalIds from one export to the next, by the rule the conservation
  plugin shares, so its surface features adhere to the same wall.

The proof through the program's pipeline (a synthetic wall photographed, aligned, densified, meshed; stones found by
SAM 2.1): `proof/make_proof.py`.

The measures say how a wall stands; they are not a verdict on it. The plugin writes only under
`<chunk data>/masonry/`; the program's elevations keep their format.

Removing a wall or a face can remove the elevations of its faces too: with program 1.10.0 the drawings kept with those
elevations (the stones of the face, its units, spot heights and points) go with them, and the question says how many
beforehand (`remove --elevations` says it after).

Command line: `ppg plugin masonry face|shape|stones|classified|usm|plate|export|ifc|list|remove`.

License: GPL-2.0-or-later.
