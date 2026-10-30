# Geoarchaeology

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/geoarchaeology/)**: the data to collect, how to write the CSV tables, the steps in the program, how to read the results.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 or later): the logs of hand augers, mechanical cores and trench sections turned into
surfaces and a 3D model of the deposits, on the same ground as the photogrammetric survey (the
DEM and the orthomosaic of the chunk).

* **Boreholes** from CSV: a collar table (id, x, y in the reference system of the chunk, z of the
  top, depth) and an interval table (id, from and to in metres below the collar, the unit - a US
  number or a lithological unit -, description, colour, lithology), or one table with both. Column
  names in Italian or English as they are written in the field (`Sigla sondaggio`,
  `Quota s.l.m. (m)`, `Codice US`, `US n.`, `Da (cm)`... or `Hole ID`, `Elevation (m a.s.l.)`,
  `From (cm)`, `Context No.`...), units of length converted; a name that only guesses its field
  (`codice`, `h`, `tetto`/`letto` holding elevations, told apart from depths by the values) is
  taken when nothing clearer is there, and reported. Comma, semicolon or tab separated; numbers
  read in the context of the file (decimal point or comma, thousands separators); a file separated
  by commas with unquoted decimal commas put back together where that can be done in one way only.
  Intervals may also be given by their elevations (`quota_tetto`, `quota_letto`). Colours as
  Munsell notations (also `2,5Y 5/2`), `#rrggbb`, `r,g,b` or words in Italian or English
  (`bruno rossastro`, `light yellowish brown`). Unit codes compared in one form (`US 1`, `US1`,
  `us 1` and, when one kind has that number, `1`), shown as written most often. Collars without z
  take it from the DEM; those with one are compared with it. Overlaps, gaps, units out of order,
  rows and colours left out are reported, with the lines of the files.
* **Order of the units** from the logs (a unit above another in a borehole lies above it, the
  disagreements to the majority), or set by hand.
* **Stratigraphic columns**: one borehole with its descriptions, or a panel of several on a
  common elevation axis with the tops of the same unit joined; units in their colour (the Munsell
  colour of the log when given) with the hatch of the lithology named (clay, silt, sand, gravel,
  stones, mudbrick, ash, peat, organic: the lithology column first, then the description, the head
  noun deciding - `ghiaia sabbiosa` is gravel, `sandy gravel` too); SVG and PNG with matplotlib,
  a panel in a file of its own (`colonne_S1-S2-S3`).
* **Contact surfaces by kriging**, written in NumPy: for the top of every unit the experimental
  semivariogram, a spherical, exponential or Gaussian model with nugget fitted by weighted least
  squares (Cressie's weights), ordinary kriging or universal kriging with a linear drift (a
  contact dipping across the site). With `auto` a model or a drift does not win by a negligible
  leave-one-out gain: among those within 5 % of the least error the steadiest is taken (spherical,
  exponential, without a drift before with one, Gaussian last), and every model tried is reported
  with its error. The kriging variance is the second band; inverse distance weighting below 5
  boreholes. A borehole that ends above a contact keeps the contact below its bottom. On the grid
  of the DEM (or a cell given), NaN outside the hull of the boreholes plus a margin.
* **Each contact in its own mode**: by *elevation* (a contact with a shape of its own: the natural
  soil under a tell, a buried floor) or by *depth* below the ground, hung from the DEM (a contact
  that follows the ground: the base of the topsoil, layers draped over a mound). The default
  `auto` takes depth when the depths of the contact vary across the boreholes less than half as
  much as its elevations (their scatter about a plane), elevation otherwise; the mode can be set
  for some units (`--mode auto,US2=depth,US5=elevation`, the *Per unità* field of the dialog,
  `modes={"US 2": "depth"}`). The registry tells the mode of every contact and why.
* **Consistent stack**: where two contacts cross, the one with the larger kriging variance
  yields (in every column the variance-weighted regression that keeps the contacts in order: a
  well determined natural soil is not pushed down by an uncertain contact above it); none rises
  above the ground (the DEM). A unit missing from a log stays missing around it: the boreholes
  vote for its presence by inverse distance, and where those without it win and the two contacts
  cannot tell a thickness from none (under two standard deviations of their kriging) its top and
  its base are one surface; no spurious thin units between the boreholes. The cells moved are
  counted (lowered to the ground, lowered by a contact above, raised by one below) and the
  correction is the third band (positive lowered, negative raised). **Thickness** (isopach)
  rasters between consecutive contacts, with the volume summed over the cells with a DEM (those
  where the ground is the surface of the collars are reported apart) and where the unit is
  missing.
* **Sections** along a trace drawn on the orthomosaic or DEM (the program's drawing, lent to the
  plugin: two or more vertices, double click or Enter to end), typed, or through the boreholes
  chosen: the profile of the ground and of every
  contact, the units filled between them, the boreholes within a distance projected onto the
  trace with their logs and their offset; at a scale (chosen to fit an A0 sheet when not given,
  as the program's own sections) with a vertical exaggeration; SVG, PNG, DXF (section
  coordinates, units as hatches of their colour) and CSV of the profile. Where matplotlib cannot
  write SVG or PDF (the desktop application up to 1.3.0 carries only its PNG backend), sections and
  columns are written as PNG only, and the user is told.
* **3D model of the units**: a closed solid per unit between its top and the top of the unit
  below (the last one down to a floor: the deepest bottom of the boreholes, kept a margin - 5 %
  of their height - below the lowest top of the last unit, so that its solid has no hole), built
  on the grid as the program builds the proxies of the stratigraphic units; a unit whose code
  matches US drawn in the chunk is clipped to their polygons. A PLY per unit (double precision,
  coloured), one OBJ with an object and a material per unit, the volumes per unit in a CSV (on
  the cells with a DEM, the others in their own columns). At the end the units are shown together
  in the program's 3D view, each in its colour (`unita_vista.ply`, optionally set apart
  vertically); the pane shows them again, opens the folder of the model, or makes the units the
  model of a new chunk of the project (*Apri come modello di un nuovo chunk*), where the *Taglia*
  button of the program's 3D view cuts them and closes every unit on the cut in its colour.
* **The implicit model, on request** (LoopStructural, MIT, in a process of its own; see below): the ground as
  scalar fields in 3D instead of surfaces z(x, y). What it adds: cuts and their fills, as excavation records (a pit
  or a ditch crosses the layers, with the outline of the US drawn, a floor through every base measured, its
  volume as a range between a flat, bowl and V floor), undercut walls from rows marked `wall`, contacts seen in
  few boreholes shaped by the layers around them, contacts and orientations from CSV files, the program's
  spot heights carrying a US. Surfaces, solids, volumes beside the kriging's, sections, exports and QGIS go through
  the same products. Kriging stays the default: on plain layers it is as good or better, and the leave-one-out
  of the two methods is shown side by side. Tab *Implicito* of the pane.
* The *Geoarcheologia* pane (with the program 1.3.4 a pane of its own beside the workspace, from *Vista > Pannelli*; before, a tab of the project dock): boreholes, surfaces and sections in three
  tables, the boreholes as a layer of points of the view (id, z, depth, units), a surface (its
  elevation, variance or correction - this one on a diverging ramp around zero, lowered and raised
  cells apart) or a thickness over the orthomosaic or DEM with palette and opacity, a section
  through the boreholes in the order they were picked, the section traces as a layer (boreholes
  and traces drawn again on every raster the view shows), the
  exports, the QGIS project (surfaces, boreholes and sections; also for a chunk in a local
  system). *File > Esporta > Sondaggi e superfici geoarcheologiche (GeoPackage)*.
* `ppg plugin geoarchaeology import|list|order|surfaces|section|model|model-chunk|logs|implicit|implicit-data|implicit-status|implicit-install|export|remove PROJECT ...`;
  `ppg plugin geoarchaeology help` prints the options of every command.

```bash
pip install "ppg-geoarchaeology @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geoarchaeology"
ppg plugin geoarchaeology import scavo.ppg collari.csv intervalli.csv
ppg plugin geoarchaeology surfaces scavo.ppg --model auto --trend auto --mode auto
ppg plugin geoarchaeology surfaces scavo.ppg --mode "auto,US2=depth,US5=elevation"
ppg plugin geoarchaeology section scavo.ppg --through S1,S4,S7 --exaggeration 5
ppg plugin geoarchaeology section scavo.ppg --coords "500100,4500800;500180,4500860"
ppg plugin geoarchaeology logs scavo.ppg S1 S2 S3 S4
ppg plugin geoarchaeology model scavo.ppg
ppg plugin geoarchaeology model-chunk scavo.ppg
ppg plugin geoarchaeology export scavo.ppg geoarcheologia.gpkg
```

A collar table and an interval table:

```text
sondaggio;x;y;quota;profondita
S1;500102,5;4500803,1;101,25;4,2
S2;500130,0;4500811,7;;3,6

sondaggio;da;a;us;descrizione;colore
S1;0;0,3;1;humus;10YR 3/2
S1;0,3;1,6;2;limo sabbioso;10YR 5/4
S1;1,6;4,2;3;argilla grigia;2.5Y 5/2
```

Everything lives in `chunk.meta["geoarchaeology"]` and in `<chunk data>/geoarchaeology/`
(`sources/` the tables read, `sondaggi.gpkg` boreholes, intervals and sections, `surfaces/`,
`sections/`, `logs/`, `model/`, `implicit/`). No dependency beyond what the program carries (NumPy, SciPy,
rasterio, shapely, ezdxf, matplotlib with its interface; colour-science for the Munsell colours
when present).

## How LoopStructural runs

LoopStructural and what it needs (pandas, scikit-image, scikit-learn) are never imported by the program: each model
runs in a process of its own (`implicit_worker.py`, started by path), with data in a job folder and results swapped in
only at the end, so a failure or a cancel leaves the present model as it was. The Python that runs it is

* the Python of the program, when LoopStructural is installed there;
* otherwise an environment the plugin makes on request with [uv](https://github.com/astral-sh/uv) (Python 3.12, about
  300 MB, in the plugin's data folder, `PPG_GEOARCH_LOOP_ENV` to put it elsewhere): the *Installa LoopStructural* button
  or `ppg plugin geoarchaeology implicit-install`. The packaged application works this way.

VTK is left out (`--no-deps`): LoopStructural uses it only for drawing, and the worker puts a stub in the place of pyvista.

```bash
ppg plugin geoarchaeology implicit-install
ppg plugin geoarchaeology implicit-data scavo.ppg --contacts contatti.csv --orientations giaciture.csv
ppg plugin geoarchaeology implicit scavo.ppg
ppg plugin geoarchaeology implicit-status scavo.ppg
```

Licence: GPL-2.0-or-later.
