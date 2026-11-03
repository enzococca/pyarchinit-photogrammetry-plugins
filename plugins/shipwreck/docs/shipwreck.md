<!-- Sections to add to the program's guide docs/plugins/shipwreck.md for version 0.2.0 of the plugin (needs
pyarchinit-photogrammetry 1.7.0 or later). Suggested place: after "The timbers", before "Command line"; the command
lines go into "Command line". -->

## Clinker strakes

Step 7 of the pane, **Strakes (clinker)**, reads the planking of a clinker hull on the sections: where each strake laps
over the one below, and from that the strakes themselves. It needs the ship frame and the sections at the stations
(steps 1 to 3); a station every metre or less is a good start.

| Button | What it does |
|---|---|
| *Find the strakes...* | The lands on every station's section, the land lines along the hull, the layer "Fasce" in the raster view and the drawings, `lands.dxf` and `lands.gpkg`. |
| *Strake model...* | A strake between two land lines: widths, length, thickness and lap, a mesh per strake, the tables. |
| *Into the catalogue* | A `plank` record of the timber catalogue per strake, its outline in plan, noted *strake model (hypothesis)*. |
| *For the Clinker Boat Tool...* | The files for Fröhlich and Nielsen's Houdini asset, in a folder you choose. |
| *Planks from Houdini...* | The planks that asset exports, back in the catalogue with their distance from the scan. |
| *Open the folder* | `<chunk data>/shipwreck/clinker`. |

### How the lands are found

On each station the plugin takes the **raw** section (never the water-corrected one) and splits it at the keel: port
(y > 0) and starboard. Each side is read from the rabbet, the top of the keel's side, out to the sheer, and ends where
the profile turns back round the sheer or bends into sediment, a trench or the seabed. A moving average over 12 cm of
the profile is the hull "without planks"; the raw profile's offset from it is a saw tooth on a clinker hull, rising
gently inside a strake and dropping by the step at each land. A land is where that offset changes by more than the
**least step** within 3 cm; two lands are at least the **least strake width** apart. Each land has its position along the
section from the rabbet and its step in millimetres.

Across the stations, the count of lands found most often is the hull's on that side. A station with another count (a
stealer, a drop strake, an eroded sheer, damage) is **uncertain**, and so is a station whose rabbet or garboard lies
more than 2 cm from the curve through its neighbours (looked for with five stations or more on a side: the curve is
a parabola through four neighbours, and with fewer a true jump is not caught). The lands of an uncertain station are matched to the lines by
their position along the section; a land on no line is listed. Nothing is faired or filled: a side with a gap, a loose
piece or no readable keel gives no lands and is listed with the reason.

| Parameter | Default | When to change it |
|---|---|---|
| Least step (mm) | 8 | Lower for thin planking or a worn hull, higher if the noise of the model makes false lands. |
| Least strake width (m) | 0.08 | Lower for very narrow strakes. |
| Half width of the keel (m) | 0.10 | How far from the keel's middle its side may reach. |
| Least width of the sole (m) | 0.02 | Lower it when a narrow, worn keel leaves the sides not read at every station. |

The pane says how many strakes each side has, which stations are uncertain and why, the lands on no line and the sides
not read. When the sections are cut again after the lands were found, the pane says the lands must be found again, and
the land lines are drawn dashed in the raster view.

### The strake model: a hypothesis

The lands are measured. The land lines that join them, and the strakes between them, are **a hypothesis from the
sections, not a measure**: every file says so (*modello delle fasce: ipotesi dalle sezioni, non una misura*), and the
model is never mixed with the sections.

For each side, strake 1 is the garboard, between the rabbet and the first land line; the last strake runs to the sheer.
Per strake: its width at every station (the length along the section between its two lines) with an uncertainty in
millimetres, nothing where the station is uncertain or not read; its length along the upper line; its thickness (the
median step of the land where it laps over the one below, or the value you give); the lap you give (20 mm by default).
The strakes are named `B1, B2, ...` on port (babordo) and `T1, T2, ...` on starboard (tribordo). Files in
`<chunk data>/shipwreck/clinker/`: `strakes.obj` (ship coordinates, one object per strake), `strakes.ply` (the chunk's
coordinates, double precision), `strakes.csv` and `strakes.xlsx`.

The model is refused when a side with lands has fewer than three stations with a width: it would be all interpolation.
When the lands are found again, the files of the earlier model (and their copies in the last folder exported for the
Clinker Boat Tool) are renamed `.old`, its strakes are taken out of the timber catalogue, and the pane says so. If a
file cannot be renamed (open in another program), nothing changes and the message names it.

### The Clinker Boat Tool

The Clinker Boat Tool by Fred Fröhlich and Sigurd Solhaug Nielsen is a Houdini Digital Asset that builds a clinker boat
plank by plank from a scan. *For the Clinker Boat Tool...* writes `clinker_boat_tool/` in the folder you choose, in the
ship frame turned so that Y is up (Houdini's up axis) and the keel runs along X: `hull.obj`, `sections.obj`,
`stations.json`, `frame.json`, `lands.obj` and `strakes.obj` with the strake tables when they exist, and a
`README.txt`. In Houdini, with the asset's six steps:

1. **Import the scan**: `hull.obj`; `sections.obj`, `lands.obj` and `strakes.obj` as references.
2. **Calibrate and align**: already done, in metres with the keel along X and Y up.
3. **Define sections**: the stations' x are in `stations.json`; `sections.obj` shows the cuts there.
4. **Generate the plank layout**: read widths, thickness and lap per strake from `lands.obj` and `strakes.xlsx`.
5. **Bend and align** the planks on the scan.
6. **Presets and export**: export the planks as OBJ, one object or group per plank, under a name of your own
   (`planks.obj`): never `strakes.obj` or `lands.obj` in the export folder, which a new detection sets aside.

*Planks from Houdini...* reads that OBJ back: each plank becomes a `plank` record of the catalogue (its outline in plan,
its side from the mean y or from B/T in its name), with the distance of its vertices from the scan, median and 90th
percentile in millimetres, in `<file>_scostamenti.csv` and `.xlsx`. Importing the same file again replaces its planks,
never yours. A group that holds all the others (such as *all*) is not a plank and is left out, whether or not it also
holds the faces outside any group.

The plugin writes files that the asset's own steps take in, and reads back the planks it exports: nothing of the asset
is copied or used. Please cite:

- Fröhlich, F., Nielsen, S. S. *From Photogrammetry to Construction Knowledge: 3D Documentation and Parametric
  Reconstruction of Historic Clinker Boats*. CC BY 4.0. [10.5281/zenodo.23422398](https://doi.org/10.5281/zenodo.23422398)
- Fröhlich, F., Nielsen, S. S. *Clinker Boat Tool* (Houdini Digital Asset). MIT License.
  [10.5281/zenodo.17488549](https://doi.org/10.5281/zenodo.17488549)

### Limits of the strakes

- On sections of a mesh noisier than about 0.5 mm the widths are good to about a centimetre. At 1 mm of noise about one
  side in fifteen may get a wrong count of lands; it is flagged as an uncertain station.
- The jumps are looked for only with five stations or more on a side. A jump of 2-2.5 cm of the rabbet at an end
  station is not caught. Three or more adjacent stations shifted together,
  or a gradual step, are not flagged.
- A permanent step of the keel (a broken or displaced hull, a scarf) flags every station on one side of it. Above about
  3 mm of scatter on the rabbet a station may be flagged wrongly.
- Without a sole, a V of planking steeper than 80° within the keel's half width is read as keel until it flattens. A
  keel side in line with the garboard is a side not read (`keel_unclear`).
- A hull seen from inside (the planking under the frames, as on many excavated wrecks) shows the inner face: the keel is
  hidden by the keelson and the reading per station gives uncertain stations. The next version will trace the lands on
  the hull's relief.

<!-- Lines to add to "Command line": -->

```bash
ppg plugin shipwreck clinker detect PROJECT.ppg [--min-step MM] [--min-strake M]   # prints the uncertain stations
ppg plugin shipwreck clinker strakes PROJECT.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROJECT.ppg --out FOLDER
ppg plugin shipwreck clinker import PROJECT.ppg --obj PLANKS.obj [--side-from y|name] [--no-deviation]
ppg plugin shipwreck clinker status PROJECT.ppg                                     # says when the lands are stale
```
