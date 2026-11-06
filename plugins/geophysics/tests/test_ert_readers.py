"""The readers of the ERT readings against files written by pyGIMLi itself (``DataContainerERT.save``
of pyGIMLi 1.6.1, in ``data/``: no pyGIMLi needed to read them), BERT's unit tokens, electrode numbers
with the remote electrode, tables with decimal commas and other encodings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_geophysics import ert

DATA = Path(__file__).resolve().parent / "data"


# ------------------------------------------------------------------ files written by pyGIMLi
def test_a_2d_profile_saved_by_pygimli_keeps_its_topography():
    # pyGIMLi saves a 2-D profile as "# x y z" with y = the elevation and z = 0, and every token of the
    # container, the unset ones (r, u, i, ip) as zeros
    p = ert.read_any(DATA / "gimli_topo.dat")
    assert p.electrodes == 12 and p.count == 45 and not p.world
    assert np.allclose(p.file_u, np.arange(12))
    assert np.allclose(p.file_z, 100 + 0.08 * np.arange(12))
    assert p.has("rhoa") and p.has("err") and p.has("k")
    for absent in ("r", "u", "i", "ip"):
        assert not p.has(absent) and absent not in p.values
    assert np.allclose(p.value("err"), 0.03) and ert.array_kind(p) == "dipole-dipole"
    assert np.isfinite(p.apparent()).all() and (p.apparent() > 0).all()


def test_world_coordinates_saved_by_pygimli():
    p = ert.read_any(DATA / "gimli_world.dat")
    assert p.world and p.file_x[0] == 500000.0 and p.file_y[-1] == 4500003.5
    assert np.allclose(p.file_z, 120 + 0.1 * np.arange(8))
    assert np.allclose(np.diff(p.file_u), np.hypot(1.5, 0.5))


def test_the_remote_electrode_of_pygimli_is_at_infinity():
    p = ert.read_any(DATA / "gimli_pd.dat")
    assert (p.b == -1).all() and ert.array_kind(p) == "pole-dipole" and p.electrodes == 12


def test_world_coordinates_without_heights_are_not_heights(tmp_path):
    # "x y z" with z = 0 is a 2-D profile, unless y cannot be an elevation (a northing)
    f = tmp_path / "w.dat"
    f.write_text("3\n# x y z\n500000 4500000 0\n500001 4500000 0\n500002 4500000 0\n1\n# a b m n rhoa\n1 2 3 0 10\n0\n")
    p = ert.read_any(f)
    assert p.world and p.file_z is None and np.allclose(p.file_u, [0, 1, 2])


def test_unit_tokens_of_bert():
    p = ert.read_any(DATA / "bert_units.ohm")
    assert np.allclose(p.file_z, 10 + 0.1 * np.arange(6))
    assert np.allclose(p.value("u"), [0.0125, 0.010, 0.008, 0.005])
    assert np.allclose(p.value("i"), [0.1, 0.1, 0.05, 0.1])
    assert np.allclose(p.value("rhoa"), [31.4, 25.1, 40.2, 62.8])
    assert np.allclose(p.value("err"), [0.025, 0.03, 0.015, 0.04])


def test_the_readings_of_pygimli_through_its_worker_keep_the_world_and_the_valid_flag():
    arrays = {"sensors_xyz": np.array([[0, 100, 0], [1, 100.5, 0], [2, 101, 0], [3, 101.5, 0]], float),
              "sensors": np.array([[0, 100], [1, 100.5], [2, 101], [3, 101.5]], float),
              "a": np.array([0, 0]), "b": np.array([1, 1]), "m": np.array([2, 2]), "n": np.array([3, 3]),
              "rhoa": np.array([10.0, 11.0]), "valid": np.array([1, 0])}
    p = ert.from_arrays("x", arrays)
    assert p.count == 1 and not p.world and np.allclose(p.file_z, [100, 100.5, 101, 101.5])
    arrays["sensors_xyz"] = np.array([[500000, 4500000, 9], [500001, 4500001, 9.5], [500002, 4500002, 10],
                                      [500003, 4500003, 10.5]], float)
    p = ert.from_arrays("x", arrays)
    assert p.world and np.allclose(p.file_z, [9, 9.5, 10, 10.5]) and np.allclose(p.file_u[1], np.sqrt(2))


# ------------------------------------------------------------------ tables
def test_numbered_electrodes_with_the_remote_one_as_minus_one_or_zero(tmp_path):
    for remote in ("-1", "0"):
        f = tmp_path / f"pd{remote}.csv"
        f.write_text("a,b,m,n,r\n" + "\n".join(f"{k},{remote},{k + 1},{k + 2},1.5" for k in range(1, 9)) + "\n")
        p = ert.read_any(f, spacing=2.0)
        assert p.numbered and p.electrodes == 10 and (p.b == -1).all()
        assert np.allclose(p.file_u, 2.0 * np.arange(10))
        k = ert.flat_k(p.file_u, p.a, p.b, p.m, p.n)
        assert np.allclose(k, 2 * np.pi / (1 / 2.0 - 1 / 4.0))  # the electrodes 2 m apart, not 1


def test_positions_in_whole_metres_are_not_numbers(tmp_path):
    # Syscal positions every 2 m from 2 m: positions, not numbers (no electrodes made up in between)
    f = tmp_path / "pos.csv"
    rows = [f"{2 * k},{2 * k + 2},{2 * k + 4},{2 * k + 6},50" for k in range(1, 9)]
    f.write_text("a,b,m,n,rhoa\n" + "\n".join(rows) + "\n")
    p = ert.read_any(f)
    assert not p.numbered and p.electrodes == 11 and np.allclose(p.file_u, 2.0 * np.arange(1, 12))
    # with Spa.1-4 columns (Syscal: positions in metres) even consecutive whole numbers are positions
    g = tmp_path / "syscal.txt"
    g.write_text("Spa.1\tSpa.2\tSpa.3\tSpa.4\tRho\n" + "\n".join(f"{k}\t{k + 1}\t{k + 2}\t{k + 3}\t40" for k in range(1, 6))
                 + "\n")
    q = ert.read_any(g, spacing=1.0)
    assert not q.numbered and np.allclose(q.file_u, np.arange(1, 9))
    assert not any("non conta" in n for n in q.notes)  # the spacing is the one of the positions: no warning


def test_space_separated_table_with_decimal_commas(tmp_path):
    f = tmp_path / "virgole.txt"
    f.write_text("a b m n rhoa err\n1 2 3 4 12,5 1,5\n2 3 4 5 13,25 2\n3 4 5 6 14 0,5\n")
    p = ert.read_any(f, spacing=1.0)
    assert np.allclose(p.value("rhoa"), [12.5, 13.25, 14.0]) and np.allclose(p.value("err"), [0.015, 0.02, 0.005])


def test_utf16_exports(tmp_path):
    f = tmp_path / "u16.csv"
    f.write_bytes("a;b;m;n;rhoa\n1;2;3;4;12,5\n2;3;4;5;13\n".encode("utf-16"))
    p = ert.read_any(f, spacing=1.0)
    assert p.count == 2 and np.allclose(p.value("rhoa"), [12.5, 13.0])


@pytest.mark.parametrize("text,expected", [
    ("500010,5 4500050,2; 500020 4500050", [[500010.5, 4500050.2], [500020.0, 4500050.0]]),
    ("500010.5, 4500050.2\n500020.5,4500050.5", [[500010.5, 4500050.2], [500020.5, 4500050.5]]),
    ("10 20;30 40", [[10.0, 20.0], [30.0, 40.0]]),
])
def test_typed_points(text, expected):
    from ppg_geophysics.ert_plugin import parse_points

    assert parse_points(text) == pytest.approx(np.array(expected))


def test_typed_points_that_cannot_be_read_are_refused():
    from ppg_geophysics.ert_plugin import parse_points

    for bad in ("500010,5,4500050,2; 1 2", "1 2 3 4; 5 6", "a b; 1 2"):
        with pytest.raises(ValueError):
            parse_points(bad)


def test_a_file_of_pygimli_goes_through_to_the_inversion(tmp_path, monkeypatch):
    import sys

    from ppg_geophysics import ert_ops
    from test_ert import FAKE, _project

    monkeypatch.setattr(ert_ops, "WORKER", FAKE)
    monkeypatch.setattr(ert_ops, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.delenv("PPG_ERT_FAKE", raising=False)
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, DATA / "gimli_topo.dat", name="G")
    assert not e["world"] and e["values"] == ["rhoa", "err", "k"]
    arrays, valid = ert_ops.inversion_input(ch, e["id"])
    assert "r" not in arrays and valid.all()
    e = ert.place(ch, e["id"], [[500010.0, 4500050.0], [500021.0, 4500050.0]], heights="file", along="plan")
    assert e["placement"]["z_range"] == pytest.approx([100.0, 100.88])
    e = ert_ops.invert_profile(ch, e["id"])
    assert e["inversion"]["fit"]["readings"] == 45


def test_an_unknown_unit_only_matters_on_a_column_the_plugin_reads(tmp_path):
    f = tmp_path / "t.ohm"
    f.write_text("4\n# x z\n0 0\n1 0\n2 0\n3 0\n1\n# a b m n rhoa t/s\n1 2 3 4 10 0.5\n0\n")
    p = ert.read_any(f)  # the time of the reading is not read: its unit does not matter
    assert np.allclose(p.value("rhoa"), [10.0])
    g = tmp_path / "u.ohm"
    g.write_text("4\n# x z\n0 0\n1 0\n2 0\n3 0\n1\n# a b m n u/furlong i rhoa\n1 2 3 4 1 1 10\n0\n")
    with pytest.raises(ValueError, match="u/furlong"):
        ert.read_any(g)


def test_two_columns_of_map_coordinates_are_not_distance_and_elevation(tmp_path):
    f = tmp_path / "xy.dat"
    f.write_text("4\n# x y\n500000 4500000\n500001 4500000.5\n500002 4500001\n500003 4500001.5\n"
                 "1\n# a b m n rhoa\n1 2 3 4 10\n0\n")
    p = ert.read_any(f)
    assert p.world and p.file_z is None and p.file_y[0] == 4500000.0
    assert np.allclose(np.diff(p.file_u), np.hypot(1.0, 0.5))
