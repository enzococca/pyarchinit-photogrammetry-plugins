# Powerlines

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/powerlines/)**: the flight and the cloud the detection needs, the parameters, how to read wires and poles.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) that finds the **overhead powerlines** in the dense cloud of a chunk: every
conductor as a catenary curve with its length, sag, clearance from the ground and distance from
the nearest obstacle (vegetation, buildings), the poles where the conductors end, the points of
wires and poles classified (ASPRS classes 14 and 15). Nothing to install beyond the program:
numpy, SciPy, shapely and ezdxf come with it.

| Where | What |
|---|---|
| *Plugin > Linee elettriche > Rileva linee elettriche...* | The detection on the dense cloud of the active chunk, in the background, with the parameters below. The conductors (layer *Linee elettriche*) and the poles (layer *Pali*) are drawn on the orthomosaic and DEM views with their attributes; a summary gives wires, lines, total length, lowest clearance and nearest obstacle. Wire points take class 14, pole points class 15 (3D view: colours by class). |
| *... > Mostra le linee elettriche* | The layers again, for example after the project is reopened. |
| *... > Elenco delle linee elettriche...* | A table: id, line, length, sag, clearance, obstacle, azimuth, points; the poles with their height and conductors. |
| *... > Dimentica le linee elettriche* | Removes the result; the points the plugin classified take back the class they had. The classes 14 and 15 of an imported lidar or of another tool, and the labels given by hand, stay. |
| *File > Esporta > Linee elettriche (GeoPackage)* | Tables `linee_elettriche` (LINESTRING Z, a vertex every metre, the attributes of the table) and `pali` (POINT Z at the ground), in the CRS of the chunk. An existing GeoPackage keeps its other tables; a file that exists and is not a GeoPackage is never deleted: the export stops with an error. |
| *File > Esporta > Linee elettriche (DXF)* | 3D polylines on the layer `LINEE_ELETTRICHE`, the poles as vertical lines on `PALI`, labels `L1`, `P1`... on `ETICHETTE`; metres, world coordinates. |
| *File > Esporta > Linee elettriche (CSV)* | One row per conductor with its attributes and ends; the poles in `<name>_pali.csv` beside it. |

From the command line:

```bash
ppg plugin powerlines detect scavo.ppg [--chunk NAME] [--min-height 4] [--radius 1] [--linearity 0.85] \
    [--gap 2] [--min-length 5] [--min-points 10] [--merge-gap 15] [--no-classify] [--export linee.gpkg]
ppg plugin powerlines list scavo.ppg
ppg plugin powerlines export scavo.ppg linee.dxf      # .gpkg, .dxf or .csv
ppg plugin powerlines clear scavo.ppg
```

## Installing

* In the program: *Strumenti > Plugin...*, `powerlines`, *Installa / aggiorna*, *Riavvia ora*.
* With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-powerlines @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/powerlines"
  ```

## How it works

The detection works on the points of the dense cloud, in the world frame of the chunk (metres,
z up: a georeferenced or levelled survey).

1. **Height above the ground** of every point: from the ground points when the cloud is
   classified (class 2 on at least 2 % of the points, for example with the ground
   classification of the terrio plugin), otherwise from the progressive morphological filter
   of the program (cells of 1 m, windows up to 40 m).
2. **Candidates**: points above the minimum height whose nearest points (within the radius)
   lie along a line - the direction of the point - with nothing beside the line (a roof has
   points beside every point, a wire has none nearer than the next conductor) and nothing in
   the air 0.3-3 m under it (a roof edge has the wall under it, a branch the trunk, a wire hangs
   free). The direction may not be steeper than 37 degrees.
3. **Wires**: candidates linked when each lies within the gap and within 25 cm of the line of
   the other, so parallel conductors are never linked; a run of points that crosses a pole is
   cut where the next span begins (both spans sag away from the pole: two parabolas fit the run
   far better than one); groups with enough points and length are kept.
4. **Catenary** `z = z0 + a (cosh((s - s0) / a) - 1)` along the direction of the wire in plan,
   fitted by robust least squares (the lowest point may lie outside the piece seen). Pieces of
   one wire separated by a gap of the reconstruction are joined when they lie on the same line,
   the gap is at most the merge gap, and the catenary of one reaches the other with the same
   slope.
5. **Poles** where the ends of several wires meet within 3 m, away from the border of the
   survey (where the cloud simply ends), confirmed by a thin vertical structure of points when
   the cloud has one. Thin poles often leave no points in photogrammetry: a pole without points
   is kept only where at least two conductors end together, and is placed from their ends.
6. **Clearance and obstacles**: along every wire (a sample every metre) the smallest height
   above the ground, and the distance from the nearest point of the cloud that is neither a
   wire nor a pole and stands at least 0.5 m above the ground.

| Parameter | Default | |
|---|---|---|
| Minimum height above the ground | 4 m | No cable is looked for below it: lower the value for low lines, raise it to leave out fences, walls and vehicles. |
| Neighbourhood radius | 1 m | The nearest points give the direction of a point: keep it below the distance between parallel conductors. |
| Minimum linearity | 0.85 | `(l1 - l2) / l1` of the points along the direction: 1 is a perfect line; lower it for noisy clouds. |
| Largest distance between points of a cable | 2 m | Raise it for sparse reconstructions. |
| Shortest cable, fewest points | 5 m, 10 | Shorter or thinner groups are not wires. |
| Largest gap bridged | 15 m | Pieces of one cable separated by a gap of the reconstruction are joined up to this gap. |
| Classes 14 and 15 in the cloud | on | Wire and pole points classified; the points of a previous run of the plugin take back their class first. Only the points the plugin finds change: the classes 14 and 15 already in the cloud (an imported lidar) stay. |

The result stays in the project (`chunk.meta["powerlines"]`): every wire with its catenary
(`origin`, `u`, `a`, `s0`, `z0`, `s_min`, `s_max`), its vertices every metre, length (arc
length), sag (largest distance below the chord), clearance, obstacle distance, azimuth and
number of points; every pole with its position, ground height, height (top of its points, or of
the highest attachment), conductors, lines and whether the cloud holds its structure.
"Lines" are groups of parallel conductors side by side over the same span.

The points the plugin classifies are remembered, with the class they had, in
`<chunk data>/powerlines/classified.npz` with a fingerprint of the dense cloud: a new detection
and *Forget* give back their class only to them, while they still hold the plugin's class and
were not labelled by hand. When the cloud changed in between (built again, cleaned, cropped) they
cannot be matched and their classes stay, as do those given by version 0.1.0 (which did not
remember its points); the summary says so.

The classes 14 and 15 are not left out of the DEM and of the model by themselves (the program
leaves out only the noise, 7 and 18): to build a DEM or a model without wires and poles, untick
classes 14 and 15 in the list of classes of *Build DEM...* or *Build model...* (built from the
dense cloud).

## Limits

* The detection works on the points: the cables must be in the dense cloud. In photogrammetry
  thin cables far from the camera are not reconstructed: fly close (inspection flights at
  20-30 m) and build the depth maps at high quality. A detection on the photos themselves is a
  possible later step.
* Conductors closer than half a metre (bundles) are not separated.
* A wire needs free air 0.3-3 m under it: cables lying on a roof or along a wall are not found.
* The heights are those of the cloud: without ground control points the clearance carries the
  error of the survey.

## Tests

A synthetic survey (a sloping ground with trees, a building, six poles, a line of three
parallel conductors over three spans - one with a gap in the reconstruction - and a lower line
crossing under it): wires, catenaries, poles, clearance, obstacles, classes, the exports, the
layers of the window, the command line and the plugin in the program.

```bash
pip install -e plugins/powerlines pytest
python -m pytest -q plugins/powerlines/tests
```

## Licence

GPL-2.0-or-later, as the program.
