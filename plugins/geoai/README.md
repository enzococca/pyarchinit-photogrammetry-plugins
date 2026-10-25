# GeoAI

GeoAI 0.1 learns pixel classes from **reviewed archaeological unit drawings** and applies the
trained model to another orthomosaic. Its results are proposals: check them on the view, delete
incorrect polygons, then convert the remaining or selected proposals into units or export a
GeoPackage. It does not infer stratigraphic relationships.

Requires pyarchinit-photogrammetry 1.5.0. GeoAI can be installed independently; when SamGeo is also
installed, both use the `geoai-env` environment and preserve each other's installed features.
PyTorch and GeoAI run in a separate process, never alongside the application's pycolmap runtime.
`Plugin > GeoAI > Install GeoAI...` installs the ML dependencies on request. A GPU is recommended;
CPU is suitable for small trials. CUDA and Metal support depend on the installed PyTorch/GeoAI stack.

## Desktop

1. Open `Plugin > GeoAI > GeoAI pane` with the orthomosaic and checked unit drawings.
2. Pick a unit field, such as `tipo` or an attribute. In the table, assign each value a class name
   and a unit type (`US`, `USM`, ...). Values with the same class name are grouped. An empty class
   excludes that value from training; its pixels are unknown, not background.
3. **Draw reviewed area**: explicitly outline the area that has been completely annotated. Ground
   without units *inside that area* becomes background. Outside it, unmapped units, conflicting
   labels and raster nodata are ignored. Do not enclose unreviewed ground.
4. **Prepare dataset...** chooses a new folder. The raster's first three bands must be RGB uint8 on
   a square, north-up metre grid. Tiles default to 512 px. At least four tile columns across the
   reviewed area are needed: the right-hand part is validation, with a complete unused tile column
   separating it from training. Each class, including background, must occur in both sets.
5. **Train model...** selects that dataset and a new output folder (conventionally `wall.ppgseg`).
   A U-Net/ResNet-34 is created through `geoai.train.get_smp_model`, initially with ImageNet encoder
   weights. Input normalization is fixed at uint8/255. Training ignores unknown pixels; validation
   reports IoU per class, mean IoU and a confusion matrix. The checkpoint with best validation mean
   IoU is kept. Validation is used for model selection, not an independent final test estimate.
6. **Apply model...** selects a model for the active chunk. Its raster must have the same resolution
   and RGB encoding as training. Resample beforehand if needed. The model is applied in tiles with
   overlapping context; output is a class GeoTIFF and a GeoPackage of proposals.
7. Review the cyan proposals. The application's Select/Delete removes proposals from the GeoPackage.
   **US** converts selected proposals, or all remaining ones if none are selected. Existing conversions
   are detected from the actual unit drawings, so Undo works. **GeoPackage...** exports the reviewed
   proposals. The class raster remains the original model output when proposals are deleted.

The model folder contains `model.json`, `weights.pt` and `history.json`. Keep the directory together
when transferring a model. Metadata records classes, unit types, bands, resolution, normalization,
spatial split, package versions and metrics. Loading uses a weights-only state dictionary.
Dataset folders are immutable; failed preparation/training has no ready manifest and requires a new
output folder. Cancellation stops the worker; a partial output may remain for inspection.

## Command line

```sh
ppg plugin geoai install
ppg plugin geoai status
ppg plugin geoai prepare excavation.ppg classes.json dataset
ppg plugin geoai train dataset walls.ppgseg --epochs 20 --batch-size 2 --device cuda
ppg plugin geoai apply another.ppg walls.ppgseg --min-area 0.01
ppg plugin geoai export another.ppg RUN_ID reviewed.gpkg
```

`classes.json` uses chunk coordinates, for example:

```json
{
  "field": "tipo",
  "mapping": {"USM": "masonry", "US": "soil"},
  "kinds": {"masonry": "USM", "soil": "US"},
  "reviewed_area": "POLYGON ((0 0, 50 0, 50 50, 0 50, 0 0))",
  "tile_size": 512,
  "validation_fraction": 0.2
}
```

The example area must be replaced with the actual fully reviewed area of your project.

## Scope and validation

This first version prepares one chunk at a time and learns RGB classes. Height bands, combining
multiple chunks and change detection between dates are subsequent steps. Spatial separation reduces
leakage but does not establish transfer to a different site: test on a separate survey before relying
on a model. A classification is a material/appearance hypothesis, not a stratigraphic interpretation.

Synthetic tests cover class mapping, unknown and conflicting labels, nodata, spatial separation,
worker isolation, fake training/application, polygons across tile seams and conversion attributes.
`PPG_GEOAI_FAKE=1` is exclusively a test engine; its models are explicitly labelled and refused in
normal mode. Real GPU training and archaeological model quality require validation on actual data.

API reference: [GeoAI training](https://opengeoai.org/train/). Plugin code: GPL-2.0-or-later;
third-party software and pretrained weights retain their respective licenses. This plugin does not
need SAM 3 weights or a SAM 3 access token.
