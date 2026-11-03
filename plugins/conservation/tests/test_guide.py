"""The guide reproduces the plugin's legend: the tables between its markers are the ones the vocabulary gives today
(``python scripts/legend_table.py --write docs/conservation.md docs/conservation.it.md`` writes them again). The
guide lives in the program's documentation once published: the test runs where the guide is next to the plugin."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("legend_table", HERE / "scripts" / "legend_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, lang", [("conservation.md", "en"), ("conservation.it.md", "it")])
def test_the_guide_reproduces_the_legend(name, lang):
    path = HERE / "docs" / name
    if not path.exists():
        pytest.skip(f"{name} is in the program's documentation")
    lt = _script()
    found = lt.current(path.read_text(encoding="utf-8"))
    assert found is not None, f"{name}: no legend block"
    assert found == lt.block(lang), f"{name}: the legend changed; run scripts/legend_table.py --write"
    from ppg_conservation.vocabulary import UNNAMED, load

    codes = [t.code for t in load().terms() if t.code != UNNAMED]
    for code in codes:
        assert f"| `{code}` |" in found
    assert UNNAMED not in found
