"""The tests read the Italian texts of the plugin, whatever the language of the system, and
never touch the network: the STAC catalog is a fake opener, the scenes small local GeoTIFFs."""
import os

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
os.environ["PPG_SENTINEL2_CLIENT"] = "stdlib"  # never pystac-client (it would go to the network)

import pyarchinit_photogrammetry  # noqa: E402,F401 - first: it points rasterio at its own PROJ data
