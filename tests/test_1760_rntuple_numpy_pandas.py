# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import math
import os

import awkward as ak
import numpy as np
import pytest

import uproot


def _to_comparable(val):
    if hasattr(val, "to_numpy"):
        val = val.to_numpy()
    if isinstance(val, np.void):
        return {k: _to_comparable(val[k]) for k in val.dtype.names}
    if isinstance(val, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(val)
        if mask.all():
            return None
        filled = val.astype(object)
        if val.ndim > 1:
            out = []
            for i in range(len(val)):
                if mask[i].all():
                    out.append(None)
                else:
                    row = filled[i]
                    row[mask[i]] = None
                    out.append([_to_comparable(x) for x in row])
            return out
        filled[mask] = None
        return [_to_comparable(x) for x in filled]
    if isinstance(val, np.ndarray):
        if val.dtype.names is not None:
            return [{k: _to_comparable(row[k]) for k in val.dtype.names} for row in val]
        if val.dtype == object:
            return [_to_comparable(x) for x in val]
        return [_to_comparable(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [_to_comparable(x) for x in val]
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


@pytest.fixture(scope="module")
def rntuple_file(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rntuple_1760")
    filepath = os.path.join(tmp_path, "all_types.root")

    reg_num = ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    reg_list = ak.to_regular(ak.Array([[[1], [2, 3]], [[4], [5]], [[6, 7], [8]]]))
    reg_str = ak.to_regular(ak.Array([["a", "b"], ["c", "d"], ["e", "f"]]))
    reg_opt = ak.Array(
        ak.contents.RegularArray(ak.Array([1, None, None, 4, 5, 6]).layout, 2)
    )
    opt_reg = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([0, 1, 0], dtype=np.int8)),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.array([1, 2, 0, 0, 5, 6], dtype=np.int64)), 2
            ),
            valid_when=False,
        )
    )

    struct_simple = ak.Array(
        [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}, {"a": 5, "b": 6.0}]
    )
    st_opt = ak.Array([1, None, 3])
    st_jag = ak.Array([[1, 2], [3], [4, 5]])
    st_reg = ak.to_regular(ak.Array([[10, 20], [30, 40], [50, 60]]))
    struct_complex = ak.Array(
        ak.contents.RecordArray(
            [st_opt.layout, st_jag.layout, st_reg.layout], ["opt", "jag", "reg"]
        )
    )

    data = ak.Array(
        {
            "jagged_num": [[1.0, 2.0], [3.0], [4.0, 5.0]],
            "jagged_str": [["a", "bc"], ["def"], ["g", "hi"]],
            "str": ["first", "second", "third"],
            "reg_num": reg_num,
            "reg_list": reg_list,
            "reg_str": reg_str,
            "reg_opt": reg_opt,
            "opt_num": [1.0, None, 3.0],
            "opt_list": [[1.0, 2.0], None, [3.0]],
            "opt_str": ["alpha", None, "gamma"],
            "opt_reg": opt_reg,
            "struct_simple": struct_simple,
            "struct_complex": struct_complex,
        }
    )

    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    return filepath


ALL_FIELDS = [
    "jagged_num",
    "jagged_str",
    "str",
    "reg_num",
    "reg_list",
    "reg_str",
    "reg_opt",
    "opt_num",
    "opt_list",
    "opt_str",
    "opt_reg",
    "struct_simple",
    "struct_complex",
]


@pytest.mark.parametrize("field", ALL_FIELDS)
def test_rntuple_numpy_pandas_dask(rntuple_file, field):
    pytest.importorskip("pandas")
    pytest.importorskip("dask.array")

    ntuple = uproot.open(rntuple_file)["ntuple"]
    expected = ak.to_list(ntuple[field].array())

    arr_np = ntuple[field].array(library="np")
    assert _to_comparable(arr_np) == expected

    arr_pd = ntuple[field].array(library="pd")
    assert _to_comparable(arr_pd) == expected

    dask_dict = uproot.dask(rntuple_file + ":ntuple", library="np")
    dask_arr = dask_dict[field]
    comp = dask_arr.compute()
    assert dask_arr.shape == comp.shape
    assert _to_comparable(comp) == expected


def test_rntuple_arrays_np_and_pd(rntuple_file):
    pytest.importorskip("pandas")
    ntuple = uproot.open(rntuple_file)["ntuple"]

    arrs = ntuple.arrays(library="np")
    assert set(arrs.keys()) == set(ALL_FIELDS)

    df = ntuple.arrays(library="pd")
    assert set(df.columns) == set(ALL_FIELDS)
    assert len(df) == 3

    for field in ALL_FIELDS:
        expected = ak.to_list(ntuple[field].array())
        assert _to_comparable(arrs[field]) == expected, field
        assert _to_comparable(df[field]) == expected, field


def test_rntuple_iterate_numpy_and_pandas(rntuple_file):
    pytest.importorskip("pandas")
    ntuple = uproot.open(rntuple_file)["ntuple"]

    chunks = list(ntuple.iterate(step_size=2, library="np"))
    assert [chunk["jagged_num"].dtype for chunk in chunks] == [np.dtype(object)] * 2
    assert [[x.tolist() for x in c["jagged_num"]] for c in chunks] == [
        [[1.0, 2.0], [3.0]],
        [[4.0, 5.0]],
    ]

    dfs = list(ntuple.iterate(step_size=2, library="pd"))
    assert [df.index.tolist() for df in dfs] == [[0, 1], [2]]
    assert [df["str"].tolist() for df in dfs] == [["first", "second"], ["third"]]


def test_rntuple_empty_slice(rntuple_file):
    ntuple = uproot.open(rntuple_file)["ntuple"]
    empty_arrays = ntuple.arrays(entry_start=0, entry_stop=0, library="np")
    assert empty_arrays["jagged_num"].shape == (0,)
    assert empty_arrays["reg_num"].shape == (0, 2)
    assert empty_arrays["opt_reg"].shape == (0, 2)


def test_rntuple_numpy_cannot_convert_union(tmp_path):
    filepath = os.path.join(tmp_path, "union.root")
    data = ak.Array({"u": [1, "two"]})
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple.arrays(library="np")

    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple["u"].array(library="np")

    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple["u"].array(library="pd")
