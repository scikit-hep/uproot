# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.behaviors.RNTuple import _rntuple_to_numpy

ak = pytest.importorskip("awkward")


def _optional_array_of_optionals():
    """``[[1, None], [3, 4], None]``, of type ``option[2 * ?int64]``."""
    inner = ak.mask(
        ak.Array([1, 2, 3, 4, 5, 6]), [True, False, True, True, False, False]
    )
    regular = ak.contents.RegularArray(inner.layout, 2)
    return ak.mask(ak.Array(regular), [True, True, False])


@pytest.fixture
def jagged_record_path(tmp_path):
    """A record that NumPy can't represent directly, since it has a jagged field."""
    path = tmp_path / "record.root"
    jagged = ak.Array([[1], [1, 2], []])
    record = ak.zip({"j": jagged, "o": _optional_array_of_optionals()}, depth_limit=1)
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"rec": record})
    return path


def test_record_with_optional_arrays_to_numpy(jagged_record_path):
    with uproot.open(jagged_record_path) as file:
        out = file["ntuple"]["rec"].array(library="np")

    assert out.dtype.names == ("j", "o")
    assert out.dtype["j"] == np.dtype(object)
    assert out.dtype["o"] == np.dtype(object)
    assert [row.tolist() for row in out["j"]] == [[1], [1, 2], []]
    # entirely missing, partly missing, and complete
    assert out["o"][0].tolist() == [1, None]
    assert out["o"][1].tolist() == [3, 4]
    assert out["o"][2] is None


def test_record_with_optional_arrays_to_pandas(jagged_record_path):
    pandas = pytest.importorskip("pandas")
    with uproot.open(jagged_record_path) as file:
        out = file["ntuple"].arrays(library="pd")
    assert isinstance(out, pandas.DataFrame)
    assert out.columns.tolist() == ["rec"]
    assert out["rec"].iloc[0]["j"].tolist() == [1]
    assert out["rec"].iloc[2]["o"] is None


def test_unconvertible_dataframe(tmp_path, monkeypatch):
    pandas = pytest.importorskip("pandas")
    path = tmp_path / "file.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"x": np.array([1, 2])})

    def fail(*args, **kwargs):
        raise TypeError("no DataFrame today")

    monkeypatch.setattr(pandas, "DataFrame", fail)
    with uproot.open(path) as file:
        with pytest.raises(ValueError, match="RNTuple fields cannot be converted"):
            file["ntuple"].arrays(library="pd")


def test_virtual_optional(tmp_path):
    path = tmp_path / "file.root"
    optional = ak.Array([1, None, 3, None])
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"o": optional})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        log = []
        virtual = ntuple.arrays(virtual=True, access_log=log)
        assert log == []
        assert virtual.o.to_list() == [1, None, 3, None]
        assert sorted(access.field_name for access in log) == ["_0", "o"]


def test_virtual_cardinality():
    path = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(path) as file:
        virtual = file["Events"].arrays(
            filter_name="nMuon", entry_stop=10, virtual=True
        )
        assert virtual.nMuon.to_list() == [2, 2, 1, 4, 4, 3, 2, 2, 2, 2]


def test_virtual_arrays_need_a_recent_awkward(tmp_path, monkeypatch):
    path = tmp_path / "file.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"x": np.array([1, 2])})
    monkeypatch.setattr(ak, "__version__", "2.8.10")
    with uproot.open(path) as file:
        with pytest.raises(ImportError, match="Awkward version 2.8.11 or later"):
            file["ntuple"].arrays(virtual=True)


@pytest.mark.parametrize(
    "data", [ak.contents.NumpyArray(np.array([1, 2, 3])), [1, 2, 3]]
)
def test_to_numpy_of_a_layout_or_something_else(data):
    out = _rntuple_to_numpy(data)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2, 3]


def test_to_numpy_of_a_list_of_chars():
    chars = ak.contents.NumpyArray(
        np.frombuffer(b"abc", dtype=np.uint8), parameters={"__array__": "char"}
    )
    lists = ak.contents.ListOffsetArray(ak.index.Index64(np.array([0, 2, 3])), chars)
    out = _rntuple_to_numpy(ak.Array(lists))
    assert out.dtype == np.dtype(object)
    assert out.tolist() == ["ab", "c"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (ak.contents.NumpyArray(np.array([1, 2])), [2, 1]),
        (ak.contents.NumpyArray(np.array([[1, 2], [3, 4]])), [[3, 4], [1, 2]]),
    ],
)
def test_to_numpy_of_a_record_with_an_indexed_field(content, expected):
    # The type of an indexed field says nothing about its shape, but it converts to a
    # plain array that has to fit in a field of objects
    indexed = ak.contents.IndexedArray(
        ak.index.Index64(np.array([1, 0])),
        content.to_RegularArray() if content.purelist_depth > 1 else content,
    )
    record = ak.contents.RecordArray(
        [ak.Array([[1], []]).layout, indexed], ["jagged", "indexed"]
    )
    out = _rntuple_to_numpy(ak.Array(record))
    assert out.dtype["indexed"] == np.dtype(object)
    assert [np.asarray(x).tolist() for x in out["indexed"]] == expected
