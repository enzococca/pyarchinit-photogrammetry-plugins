# Conservation

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 or later):
decay mapping on the program's elevations.

* **Vocabulary**: the terms of the ICOMOS-ISCS *Illustrated glossary on stone deterioration patterns* (2008) are the
  codes (`ISCS.CRACK`, `ISCS.CRACK.HAIR_CRACK`, `ISCS.ALVEOLIZATION`, ...); the 27 terms of UNI 11182:2006 are their
  Italian face, each mapped onto the ICOMOS-ISCS term or terms it corresponds to (the mapping is the plugin's: no
  official table exists); masonry and structural symptoms that neither glossary has (out of plumb, bulging, loss of
  mortar, displaced stones, open joint) are local terms, flagged as such; the interventions follow the families of
  EN 15898. Names are the glossaries' own, the one-line definitions the plugin's paraphrases with their source. A
  project can add its own terms.
* **Legend**: the colours and hatches are the plugin's defaults - no standard prescribes them - and can be changed per
  project; they are written as QGIS styles too.
* **Campaigns**: a survey of a face (an elevation of the program) on a date, with the surveyor, the method and the
  EN 16096 condition and urgency classes the surveyor chooses; the recommendation class of the whole object once per
  date.
* **Features**: areas of decay, cracks as lines with their opening, treatments, notes - drawn on the elevation in its
  (u, v) metres, with their extent, intensity, the photos that see them and the stratigraphic units they lie on; in the
  world they are lifted on the surface through the elevation's relief.

The plugin writes only `chunk.meta["conservation"]` and files under `<chunk data>/conservation/<elevation>/`.

Licence: GPL-2.0-or-later.
