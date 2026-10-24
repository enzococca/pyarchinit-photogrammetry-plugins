"""The tests read the Italian texts of the plugin, whatever the language of the system."""
import os

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
