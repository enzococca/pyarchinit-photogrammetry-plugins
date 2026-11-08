"""Flight plans (Piani di volo) - a plugin of pyarchinit-photogrammetry: a drone survey designed before going
to the field - area, camera, ground resolution and height, overlaps, strips, shots, time, batteries, terrain
following, the coverage predicted on the DEM - and the files the drones read.

The planner is ported from the author's Photo3D planner. Its plans are JSON dicts of blocks
(:mod:`ppg_flightplan.plan`) on WGS84 latitude and longitude, heights above the takeoff point; the plugin
converts to the chunk's coordinate system only at the boundary (the area drawn on the view, the shots shown
back on it).
"""
from __future__ import annotations

from pyarchinit_photogrammetry.plugins import Plugin

__version__ = "0.1.0"


class FlightPlanPlugin(Plugin):
    name = "flightplan"
    title = "Piani di volo"
    version = __version__
    description = ("Piani di volo per il rilievo da drone: area, camera, GSD e quota, sovrapposizioni, strisciate, "
                   "scatti, tempi e batterie, volo che segue il terreno, i file per i droni")

    def __init__(self):
        self.context = None

    def activate(self, context) -> None:
        self.context = context
