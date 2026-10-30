"""Landscape through time (Landlab) - a plugin of pyarchinit-photogrammetry: how the surveyed
surface changes over the years through soil creep (hillslope diffusion) and running water (erosion
and deposition with SPACE), simulated with Landlab (MIT, CSDMS) on the DEM of the chunk.

Landlab is never imported by the program: it runs in a process of its own (:mod:`.worker`, the
simulation in :mod:`.engine`) with this Python when Landlab is installed here, else in an
environment the plugin installs with uv (:mod:`.environment`); :mod:`.runner` starts the worker
and relays its progress.
"""
from __future__ import annotations

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, t

__version__ = "0.1.0"


class LandscapePlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Come cambia nel tempo la superficie rilevata: creep del suolo e acqua che scorre "
                             "(erosione e deposito) simulati con Landlab sul DEM del chunk",
                             "How the surveyed surface changes over time: soil creep and running water (erosion "
                             "and deposition) simulated with Landlab on the DEM of the chunk")
        self.context = None

    def activate(self, context) -> None:
        from . import environment

        self.context = context
        environment.configure(context)
