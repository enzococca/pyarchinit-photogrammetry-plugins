"""The tests read the Italian texts of the plugin, whatever the language of the system, and never touch
the network."""
import os

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")

import pyarchinit_photogrammetry  # noqa: E402,F401 - first: it points rasterio at its own PROJ data
import pytest  # noqa: E402

from planfixtures import reference_plan  # noqa: E402


@pytest.fixture
def plan() -> dict:
    return reference_plan()
