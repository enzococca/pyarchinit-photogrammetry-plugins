# Landscape through time (Landlab)

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/landscape/)**: the DEM to start from, the scenario step by step, how to read the results.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 or later): how the surveyed surface changes over the years through **soil creep** (hillslope
diffusion) and **running water** (erosion and deposition), simulated with
[Landlab](https://landlab.csdms.io) (CSDMS, MIT licence) on the DEM of the chunk. How a tell or a
mound wears down, where the colluvium gathers at the foot of a slope or in a hollow, how much of an
excavated or exposed surface goes in ten or a hundred years.

**Scenarios, not predictions.** The result depends on the parameters, which are known within an
order of magnitude at best: read it as "what these processes, at these rates, would do to this
surface". Compare it with the DEM of a later phase to calibrate the parameters. Time runs forward
only: an older surface cannot be rebuilt by running the model backwards (diffusion backwards is not
physical, it amplifies every error) and the plugin does not offer it. Not simulated: tectonic uplift,
rain that changes over time, vegetation, wind, lateral migration of channels, the sea, what lies
under the surface.

## The processes

| Process | Landlab component | Parameters |
|---|---|---|
| **Soil creep** (hillslope diffusion, Culling 1963) | `LinearDiffuser`: the flux of soil is `D` times the slope | `D` (m²/yr) |
| **Running water** (SPACE, Shobe et al. 2017) | `SpaceLargeScaleEroder` (then `Space`, then `ErosionDeposition` when missing): erosion of a layer of loose soil and of the material below, transport and deposition of the sediment | `K_sed`, `K_br`, `F_f`, `phi`, `H_star`, `v_s`, `m_sp`, `n_sp`, runoff, thickness of the soil |
| Flow routing | `PriorityFloodFlowRouter` (richdem) when the data reach the frame of the grid, else `FlowAccumulator` + `DepressionFinderAndRouter`: D8, depressions filled; discharge = drained area x runoff | runoff (m/yr) |

Either process can be switched off (not both). The cells without a value are closed. The edge of the
data is **open** (water and sediment leave there, as on a slope that goes on beyond the survey) or
**closed** (nothing leaves; with running water the lowest edge cell beside a cell inside stays open as
the one outlet - one for each group of cells inside the edge: the islands a DEM often has beside its
main body, and the parts joined to it only by a neck one or two cells wide).

## Presets

The values are the plugin's defensible starting points, in the upper part of the published ranges
for loose, unvegetated material. They are **indicative**: calibrate them with a later DEM when you
have one.

| Preset | Years | D (m²/yr) | K_sed | K_br | F_f | phi | H* (m) | v_s (m/yr) | runoff (m/yr) | soil (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| *Tell in mattoni crudi* (mudbrick tell) | 1000 | 0.01 | 5e-4 | 1e-4 | 0.5 | 0.3 | 0.5 | 1 | 0.2 | 0.5 |
| *Versante in terra* (earth slope) | 1000 | 0.005 | 1e-3 | 2e-4 | 0.2 | 0.3 | 0.5 | 1 | 0.3 | 0.5 |
| *Scavo esposto* (exposed excavation) | 10 | 0.02 | 2e-3 | 5e-4 | 0.3 | 0.3 | 0.2 | 1 | 0.3 | 0.2 |
| *Personalizzato* (custom) | any value; changing a value of a preset makes it custom ||||||||||

`m_sp` = 0.5 and `n_sp` = 1 in all of them (the usual stream-power exponents).

### Where the values come from

* **Hillslope diffusivity D.** The diffusion model of soil creep is Culling (1963, *Journal of
  Geology* 71). Measured values on soil-mantled hillslopes lie mostly between 1e-4 and 1e-2 m²/yr:
  Roering, Kirchner and Dietrich (1999, *Water Resources Research* 35) found about 3e-3 m²/yr in the
  Oregon Coast Range; the degradation of fault scarps and old shorelines gives about 1e-3 m²/yr
  (Hanks et al. 1984). Bare, loose material moves faster than vegetated soil: the presets put a
  mudbrick tell at 1e-2 and a fresh, unprotected excavation at 2e-2, above the natural range on
  purpose; a grassed earth slope at 5e-3.
* **Erodibility K_sed and K_br.** In SPACE (Shobe, Tucker and Barnhart 2017, *Geoscientific Model
  Development* 10) the soil is eroded at `K_sed Q^m S^n (1 - exp(-H/H*))` and the material below at
  `K_br Q^m S^n exp(-H/H*)`. Their units depend on `m`, `n` and on the discharge (m³/yr here), so a
  value is comparable only with the same exponents and units; the loose soil is more erodible than the
  material below (`K_sed` above `K_br`, by 5 in the presets). There is no general table of them:
  the examples that come with Landlab's SPACE use anything from 1e-5 to 1 (the component's defaults
  are 0.02), set for catchments that evolve over thousands of years. The presets stay low (5e-4 -
  2e-3) because the slopes of a tell are short and the mudbrick is cohesive. These two are the least
  known values: calibrate them.
* **F_f**, the fraction of the eroded material below the soil so fine that it leaves in suspension
  without settling (0-1): high for mudbrick and silty fills (0.5), lower for coarser earth.
* **phi**, the porosity of the deposited sediment: 0.3 is a common value for sand and gravel
  (0.2-0.5 for natural sediments).
* **H\***, the thickness of soil at which the water stops reaching the material below
  (`exp(-H/H*)`): 0.5 m in the presets, 0.2 m for an excavation floor where little loose soil lies.
* **v_s**, the effective settling velocity (m/yr): with v_s much larger than the runoff the sediment
  settles close to where it eroded (transport-limited), much smaller it is carried away
  (detachment-limited); Shobe et al. (2017) discuss that transition. 1 m/yr with 0.2-0.3 m/yr of runoff
  sits between the two.
* **Runoff**, the rain that runs off the surface in a year, uniform (m/yr): 0.2-0.3 m is a
  semi-arid to Mediterranean climate after infiltration.

## The result on the DEM

The simulation runs on a grid of its own: the DEM averaged to a coarser cell (by default its
resolution coarsened so that the longer side has at most 500 cells; above 1000 cells a side the pane
warns). The **change** the simulation made - dz, and each snapshot minus the start - is interpolated
back onto the grid of the DEM (bilinear) and **added to the DEM**: the detail the coarser simulation
could not see (stones, walls, the edges of trenches) stays in the final surface. The cells of the DEM
out of reach of the interpolation, along the outermost edge of the data, get dz 0. The drainage area
comes back by nearest neighbour. Every layer is a float32 GeoTIFF with NaN as nodata, on the grid, in
the frame and with the CRS of the DEM; `dz.tif` is written as the difference of the two float32
surfaces, so `final - DEM = dz` holds in the files.

## The volumes

`balance.json` of each run, and the pane:

* **bulk** volumes from dz: `eroded_m3`, `deposited_m3`, `net_m3` - what the map shows, pores
  included;
* **solid** volumes, per cell `(1 - phi) dH + d(bedrock)`: `eroded_solid_m3`, `deposited_solid_m3`
  (the soil is porous, the material below is not; with creep alone solid = bulk);
* what leaves: `bedload_out_m3` and `wash_load_m3` (the fines F_f), their sum `fluvial_out_m3`, and
  `diffusion_out_m3` / `diffusion_out_solid_m3` (creep across open edges; the solid one also counts
  the pores gained when creep turns the material below into soil);
* `closure_m3` = solid eroded - solid deposited - everything out, and `closure_percent` of the eroded
  solid: about 0 when the books close (it is checked in the tests, in every combination of edges,
  F_f and phi). None with `Space` or `ErosionDeposition`, which do not report their outflow.

Bulk eroded and deposited differ when sediment is deposited (pores) or leaves: compare the solid
ones.

## How long it takes

Before a run the pane shows an estimate: the sub-steps of the diffusion (exact: no step longer than
0.14 dx²/D), an upper bound of the steps of the running water (each step shorter than a CFL-like
limit, from the fastest incision possible: the whole grid draining to one cell) and the seconds, from
costs measured on an Apple M4 Pro scaled by the cells. Above 10 minutes it warns, above 2 hours it
refuses (`--force` on the command line). The estimate is a ceiling: the steps of the real run follow
the real discharge and are usually far fewer. A larger cell shortens a run a lot (the diffusion
steps go with the square of the cell).

## Where

* the **Paesaggio** (*Landscape*) pane, on the right beside *PyArchInit*
  (*Vista > Pannelli*): the DEM to start from (the active one or a kept version), the preset, the
  years, the snapshots, the cell, the edge, the step, the two processes with their parameters
  (advanced ones of the water behind a tick), the estimate, *Run*; the runs of the chunk in a table,
  their layers over the 2D view (final surface, erosion and deposition with a diverging palette,
  drainage area, snapshots, differences), opacity, *Add as a version of the DEM*, *Compare with
  another chunk...*, *Export...*, *Remove*;
* the runs in the workspace tree, in the QGIS project of the program, in the raster calculator
  (program 1.4.0 or later); *File > Esporta > Paesaggio nel tempo (GeoTIFF)*;
* *Plugin > Paesaggio nel tempo (Landlab)* in the menu;
* the command line:

```bash
ppg plugin landscape status
ppg plugin landscape install                        # the environment of the plugin (uv)
ppg plugin landscape estimate scavo.ppg --preset mudbrick_tell --years 1000
ppg plugin landscape run scavo.ppg --preset mudbrick_tell --years 1000 --snapshots 100,500 --name "Tell, 1000 anni"
ppg plugin landscape run scavo.ppg --preset custom --years 50 --D 0.02 --no-water --boundary closed
ppg plugin landscape run scavo.ppg --preset earth_slope --cell 2 --K-sed 0.002 --runoff 0.4
ppg plugin landscape list scavo.ppg
ppg plugin landscape show scavo.ppg r20261030-101500
ppg plugin landscape version scavo.ppg r20261030-101500       # the final surface among the versions of the DEM
ppg plugin landscape compare scavo.ppg r20261030-101500 --with "Fase 2"
ppg plugin landscape export scavo.ppg r20261030-101500 paesaggio.tif            # every layer, one band each
ppg plugin landscape export scavo.ppg r20261030-101500 dz.tif --layer dz
ppg plugin landscape remove scavo.ppg r20261030-101500
```

`--chunk LABEL` works on another chunk than the active one; `--source` starts from a kept version
of the DEM; `--dt` sets the longest step; `--json` prints the whole entry. The parameters of the
scenario: `--D --K-sed --K-br --runoff --soil-depth --F-f --phi --H-star --v-s --m-sp --n-sp`,
`--no-diffusion`, `--no-water`.

The active DEM is never replaced: *Add as a version of the DEM* puts the final surface among the
kept versions (*Paesaggio dopo 1.000 anni (...)*), and contours, sections, volumes and relief use it
when you make it active. *Compare* gives the final surface minus the DEM of another chunk (a later
phase) - median, NMAD and 95th percentile - next to the same numbers for the starting DEM: the
parameters come closer when the difference shrinks.

## How Landlab runs

Landlab and its compiled dependencies (richdem, netCDF4) are never imported by the program: each
run happens in a process of its own (`ppg_landscape/worker.py`, the simulation in
`ppg_landscape/engine.py`). The plugin writes the DEM at the simulation cell for it, the worker
writes the arrays back, and the two speak through lines of JSON (progress, result, error);
*Cancel* stops it at once and leaves nothing behind. The worker runs with:

* the Python of the program, when Landlab is installed there (`pip install landlab`);
* otherwise an environment the plugin installs on request with
  [uv](https://github.com/astral-sh/uv) - Python 3.12 and Landlab 2.x from PyPI, about 500 MB, in
  the plugin's folder (`PPG_LANDSCAPE_ENV` to put it elsewhere): this is how the packaged
  application works, which has no pip. The pane offers it the first time.

Landlab has builds for macOS (Apple Silicon; Intel with macOS 15 or later), Linux (x86_64 and
aarch64) and Windows x64; not for Windows on ARM, where the plugin says it is not available.

## Installing

* In the program: *Strumenti > Plugin...*, `landscape`, *Installa / aggiorna*, *Riavvia ora*.
* With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-landscape @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/landscape"
  pip install landlab        # optional: else the plugin installs its own environment
  ```

Tests: `pip install -e . pytest` in the environment of the program, then `pytest tests`. The
simulations themselves run where Landlab is importable (`<env>/bin/python -m pytest tests` with an
environment that has Landlab, numpy and pytest), and the tests of the real worker take the Python
named by `PPG_TEST_LANDLAB`; the others use a worker that speaks the same protocol without Landlab.

## References

* Culling, W. E. H. (1963). Soil creep and the development of hillside slopes. *The Journal of
  Geology* 71(2), 127-161.
* Hanks, T. C., Bucknam, R. C., Lajoie, K. R., Wallace, R. E. (1984). Modification of wave-cut and
  faulting-controlled landforms. *Journal of Geophysical Research* 89(B7), 5771-5790.
* Roering, J. J., Kirchner, J. W., Dietrich, W. E. (1999). Evidence for nonlinear, diffusive
  sediment transport on hillslopes and implications for landscape morphology. *Water Resources
  Research* 35(3), 853-870.
* Shobe, C. M., Tucker, G. E., Barnhart, K. R. (2017). The SPACE 1.0 model: a Landlab component for
  2-D calculation of sediment transport, bedrock erosion, and landscape evolution. *Geoscientific
  Model Development* 10, 4577-4604.
* Barnhart, K. R., et al. (2020). Short communication: Landlab v2.0: a software package for Earth
  surface dynamics. *Earth Surface Dynamics* 8, 379-397.

Licence: **GPL-3.0-or-later** (richdem, which Landlab uses for the flow routing, is GPL-3; the
program and most plugins of this repository are GPL-2.0-or-later, which allows it). Landlab is
MIT-licensed. Author: Enzo Cocca.
