# Underwater

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/underwater/)**: the photos and
frames to use, the colours, the depths. *[Italiano](README.it.md)*

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 or later)
for surveys under water - wrecks, submerged structures, finds on the sea bed - with the photos of a diver or the
images of an ROV. It needs nothing beyond the program (numpy, SciPy, OpenCV, Pillow, pycolmap).

| Where | What |
|---|---|
| **1. Photos and frames** | |
| *Fotogrammi da un video...* | GoPro or ROV video: the program's own extraction (a frame each time the view has moved, the sharpest of the moments after), with the time of every frame (the creation time written in the MP4/MOV, or one given, plus the time in the video) |
| *Fotogrammi da una sequenza...* | A folder of images at 10-30 a second (an ROV): every frame matched with the one before (ORB points, a similarity with RANSAC), the frames of a stretch chained; the next frame kept is the sharpest of those that moved by half to all of `1 - overlap` of the width since the last one, frames much less sharp than their neighbours left out; an ROV hovering gives no new frames, a jump starts a new stretch. The frames stay where they are and become photos with their time (a table of the times - name or number, time - or the file names, or the EXIF) |
| *Calibrazione (Kalibr)...* | The calibration of an ROV camera from Kalibr (radtan: OPENCV, equidistant: OPENCV_FISHEYE), the principal point moved by half a pixel to COLMAP's convention, held fixed in the alignment |
| *Valuta le foto* | Three scores per photo: **quality** (the program's sharpness over that of the sharpest photos of the chunk: blur), **contrast** (spread of the lightness, 2nd-98th percentile: milky water, too dark) and **veil** (the median of the dark channel of green and blue over the brightest light: haze, backscatter). Thresholds in the pane (0.5, 0.12, 0.6), the flagged photos in a table with why; *Escludi le segnate* disables them (the alignment and the products leave them out), *Riattiva le escluse* brings them back; a CSV table |
| **2. Colours** | |
| *Ripristino fisico (Sea-thru)* | With the **range of every pixel** from the program's depth maps (*Flusso di lavoro > Mappe di profondità*; the depth maps are of the undistorted photos: every pixel of the photo is taken to its ray with the camera model and to the depth map, `range = z * sqrt(1 + x² + y²)`, in metres when the chunk is) the image formation model of Sea-thru (Akkaynak and Treibitz, CVPR 2019), written here from the paper: the **backscatter** of every channel fitted to the darkest pixels of every range bin, `B(z) = B∞ (1 - e^(-βB z))`, subtracted; the **illuminant** - the local average among pixels at a similar range - fitted by `log E = w + e_photo - βD(z) z`; where photos see the same scene, the **points seen twice** at two ranges give the differences of attenuation between the channels free of the scene's colours and of the exposures (`log D_i - log D_j = (e_i - e_j) - β (z_i - z_j)`), the illuminant only that of green; the colours `J = D e^(βD z)`, the colour of the light from the **grey edges** (van de Weijer et al. 2007) of the restored photos. The coefficients **once for the survey** (the photos of a dive get the same correction: no seams of colour in the orthomosaic and the texture; the exposure and the strength of the backscatter of every photo stay its own) or **per photo** |
| *Correzione rapida* | No ranges: before the alignment, or without depth maps. Red (and, in green water, blue) compensated with the green (Ancuti et al., IEEE TIP 2018), grey-world white balance, a stretch of the lightness, optional CLAHE (local contrast: it can make the alignment of dark, turbid photos possible). Means and gains from a sample of the survey together |
| *Anteprima*, *Correggi le foto* | Before and after of one photo; then copies of all of them, same names, size and format, the EXIF kept, in `<chunk data>/underwater/colour/` (the originals are never touched) |
| *Usa le foto corrette*, *Usa le originali* | The cameras pointed to the copies or back to the originals (what the program's *Ricollega le foto* does, both ways): the orthomosaic, the texture, the depth maps and a new alignment read the photos where the cameras point |
| **3. Depth instead of GPS** | |
| *Importa il registro...* | CSV and text tables (Unix time in s/ms/us/ns, ISO dates, clock times with a date column, seconds from the start; depth in m or ft, or pressure in bar, mbar, dbar, kPa, Pa, psi turned into depth with the density of sea or fresh water), dive computer exports (Subsurface: date, time, sample time, sample depth), UDDF, the depth file of AQUALOC. Matched to the photos by time (the frames' own times, the file names, the EXIF with sub-seconds and time zone) plus a **time offset** |
| *Stima lo scarto* | The time offset with which the depths agree best with the aligned cameras (the levelling solved for every offset tried, coarse then fine) |
| *Metti in bolla e in scala* | `v · c + b = -depth` for every camera centre `c` (internal frame): linear, solved with Huber weights; `|v|` is the scale, `v/|v|` the vertical, `b` the height. The chunk in a local frame in metres, **Z = -depth** (the surface at 0), X, Y and the heading kept as they were; the residual of every camera, the outliers, the uncertainty of the vertical. When the depth changes too little to give the scale (a survey at constant depth) the control scale bars give it, or *Scala: dalle barre di scala* |
| *Posizioni note (USBL)...* | X, Y of some cameras by photo name or by time (interpolated): after the levelling a horizontal similarity gives the heading and the position, in their CRS |

The program's camera references are positions in three coordinates with one accuracy for all three: a
reference with only Z cannot be held there, so the levelling is done by the plugin and kept in
`chunk.meta["underwater"]["levelling"]`. A new alignment resets the transform of the chunk: the pane and the
workspace say *Bolla da rifare*, one click levels it again.

From the command line:

```bash
ppg plugin underwater frames-sequence scavo.ppg raw_data/images_sequence_6 --times raw_data/img_sequence_6.csv --overlap 0.9
ppg plugin underwater calibration scavo.ppg archaeo_camera_calib.yaml
ppg plugin underwater score scavo.ppg --quality 0.5 --contrast 0.12 --veil 0.6 --exclude
ppg plugin underwater quick scavo.ppg --clahe --alpha 0 --no-balance --use     # before the alignment
ppg plugin underwater restore scavo.ppg --per survey --use                       # after the depth maps
ppg plugin underwater restore scavo.ppg --depth-folder depthMaps --fov 76       # depth images in metres made elsewhere (Metashape)
ppg plugin underwater use scavo.ppg original
ppg plugin underwater depth scavo.ppg depth_sequence_6.csv --offset 0
ppg plugin underwater offset scavo.ppg --span 900
ppg plugin underwater level scavo.ppg [--scale bars] [--positions usbl.csv --crs 32633]
ppg plugin underwater status scavo.ppg
ppg plugin underwater csv scavo.ppg foto.csv
```

## Tried on real data

**AQUALOC**, archaeological site, sequence 6 (M. Ferrera, V. Creuze, J. Moras, P. Trouvé-Peloux, *AQUALOC: An
Underwater Dataset for Visual-Inertial-Pressure Localization*, IJRR 2019): 3385 monochrome frames 968 x 608 at
20 per second over an amphora wreck at 387-391 m, the pressure depth of the ROV, a COLMAP trajectory as ground
truth. 139 frames chosen in 8.5 s (overlap 0.9); aligned by the program with the Kalibr calibration held fixed:
63 of 139 with the original frames, **136 of 139** with the quick correction (CLAHE) used for the alignment
(sandy stretches with little texture). Levelled and scaled by the depths: residual of the camera depths
**2.3 cm RMS** (max 6.1 cm, depth range 3.9 m), standard error of the scale 0.5 %, vertical within **0.18°** of
the vertical the depths give to the ground truth; after a similarity, the camera positions are within **5.8 cm
RMS** of the ground truth over 38 m. The ground truth itself needs a scale of 1.065 to fit the same depths (our
scale is the depths'). A camera clock shifted by 7.3 s: the offset found is 7.33 s.

**Sea-thru D3** (D. Akkaynak, T. Treibitz, CVPR 2019): RAW photos of a coral head in the Red Sea at 1-3 m with a
colour chart, and depth maps in metres made with Metashape. Developed with the camera white balance and no colour
matrix (one that clips the red to zero leaves nothing to restore), 23 photos restored with their depth maps (fitted
on the 23 together, 2.9 minutes): the grey scale of the chart (5 patches at 0.9 m) has a mean chroma (CIELAB a*b*)
of **3.4** (0.9-4.7) against 16.9 in the photos as developed and 6.1 with the quick correction; one model per photo
3.2. Attenuation found (per metre): red 0.149, green 0.120, blue 0.094 (the illuminant alone, before the points seen
twice, gave blue 0.51 and a chart at 19). The same points seen from ranges at least 25 % apart (91 matches, median
0.38 m apart) differ by a median ΔE a*b* of 1.56 in the developed photos, 1.74 restored, 1.41 with the quick
correction: in D3 the camera stays 1-3 m from the scene and the colours hardly change with the range; the gain is in
the colour of the light.

## References

- D. Akkaynak, T. Treibitz, *Sea-thru: A Method for Removing Water From Underwater Images*, CVPR 2019 (the model
  of the restoration; the code here is written from the paper).
- C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, P. Bekaert, *Color Balance and Fusion for Underwater Image and
  Video Enhancement*, IEEE Transactions on Image Processing 27(1), 2018 (the red compensation).
- J. van de Weijer, T. Gevers, A. Gijsenij, *Edge-Based Color Constancy*, IEEE Transactions on Image Processing
  16(9), 2007 (the colour of the light from the edges).
- P. Drews Jr., E. do Nascimento, F. Moraes, S. Botelho, M. Campos, *Transmission Estimation in Underwater Single
  Images*, ICCV Workshops 2013 (the underwater dark channel of the veil score).
- AQUALOC and Sea-thru (above): used for the tests on this computer only; they are not distributed with the plugin.

Licence: GPL-2.0-or-later, like the program the plugin runs in.
