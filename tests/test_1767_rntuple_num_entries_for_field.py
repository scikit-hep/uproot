# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import uproot

ak = pytest.importorskip("awkward")


def _data(start, stop):
    return ak.Array(
        {
            "x": numpy.arange(start, stop, dtype=numpy.int64),
            "jagged": [list(range(i % 5)) for i in range(start, stop)],
            "struct": [{"a": i, "b": float(i)} for i in range(start, stop)],
        }
    )


@pytest.fixture
def ntuple(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        # two clusters, so that entry ranges can start and stop in either one
        writable = file.mkrntuple("ntuple", _data(0, 600))
        writable.extend(_data(600, 1000))
    with uproot.open(path) as file:
        yield file["ntuple"]


@pytest.mark.parametrize("name", ["x", "jagged", "struct", "struct.a"])
@pytest.mark.parametrize(
    "entries",
    [
        {},
        {"entry_start": 100, "entry_stop": 200},
        {"entry_start": 500, "entry_stop": 700},
    ],
)
def test_field_matches_ntuple(ntuple, name, entries):
    # a field needs as many entries as the RNTuple with only that field selected
    for memory_size in ("1 kB", 100, "1 MB"):
        assert ntuple[name].num_entries_for(
            memory_size, **entries
        ) == ntuple.num_entries_for(memory_size, filter_name=name, **entries)


@pytest.mark.parametrize(
    "entries",
    [
        {},
        {"entry_start": 100, "entry_stop": 200},
        {"entry_start": 0, "entry_stop": 600},
        {"entry_start": 600, "entry_stop": 1000},
        {"entry_start": 500, "entry_stop": 700},
    ],
)
@pytest.mark.parametrize("name", ["x", "struct.a"])
def test_field_value(ntuple, name, entries):
    # 1 kB holds 125 int64 values in any entry range
    assert ntuple[name].num_entries_for("1 kB", **entries) == 125
    assert ntuple.num_entries_for("1 kB", filter_name=name, **entries) == 125
