# Architectural elements

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry): parametric
architectural elements made from the photogrammetric model.

* **Column by revolution**: draw the region in plan; the axis and the profile are measured on the model (the median
  radius per height and its spread), the solid is a spline through K knots fitted to it, or the profile as measured.
  Base, shaft and capital can be separate elements on the same axis.
* **Moulding, cornice, string course**: trace the path; the section is cut from the model by the plane normal to the
  path, closed against the wall; the solid is the section swept along the path with mitred corners.
* **Arch and barrel vault**: on a section of the project or a vertical plane through two points; the intrados is fitted
  as a circle (semicircular or segmental), a pointed arch, an ellipse or kept as measured, the type chosen by the fit.

Every element is written as STEP (CadQuery, OpenCascade) and as a parametric IFC solid, and always carries its **fit**:
the median, 90th percentile and maximum distance of the model's points from the solid. The solid is an interpretation
of the measured model; the fit says how far.

CadQuery runs in an environment of its own that the plugin installs on request (about 900 MB; not on Windows on ARM),
never in the program's process. Command line: `ppg plugin architecture column|moulding|arch|list|show|export|remove|install|status`.

Licence: GPL-2.0-or-later. CadQuery and OCP are Apache-2.0, OpenCascade LGPL-2.1 with an exception, VTK BSD.
