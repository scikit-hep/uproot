# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

NESTED = "test_nested_structs_rntuple_v1-0-0-0.root"
STL = "test_stl_containers_rntuple_v1-0-0-0.root"
VLV = "test_int_vfloat_tlv_vtlv_rntuple_v1-0-0-0.root"

SUB_STRUCT = "{i: int32, sub_sub_struct: {i: int32, v: var * int32}}"
MY_STRUCT = f"{{i: int32, sub_struct: {SUB_STRUCT}}}"


def _ntuple(filename):
    return uproot.open(skhep_testdata.data_path(filename))["ntuple"]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({"filter_name": "my_struct"}, f"{{my_struct: {MY_STRUCT}}}"),
        ({"expressions": ["my_struct"]}, f"{{my_struct: {MY_STRUCT}}}"),
        # selecting a field and some of its subfields still gives the whole field
        (
            {"filter_field": lambda f: f.path in ("my_struct", "my_struct.i")},
            f"{{my_struct: {MY_STRUCT}}}",
        ),
        ({"filter_name": "my_struct*"}, f"{{my_struct: {MY_STRUCT}}}"),
        (
            {"filter_name": "my_struct.sub_struct"},
            f"{{my_struct: {{sub_struct: {SUB_STRUCT}}}}}",
        ),
        # selecting only leaves keeps only the paths down to them
        (
            {"filter_name": "my_struct.sub_struct.i"},
            "{my_struct: {sub_struct: {i: int32}}}",
        ),
        (
            {
                "filter_field": lambda f: f.path
                in ("my_struct.i", "my_struct.sub_struct.sub_sub_struct.v")
            },
            "{my_struct: {i: int32, sub_struct: {sub_sub_struct: {v: var * int32}}}}",
        ),
    ],
)
def test_selection(selection, expected):
    ntuple = _ntuple(NESTED)
    assert str(ntuple.arrays(**selection).type) == f"10 * {expected}"


def test_selected_field_matches_its_array():
    ntuple = _ntuple(NESTED)
    assert ak.array_equal(
        ntuple.arrays(filter_name="my_struct").my_struct, ntuple["my_struct"].array()
    )
    # the same applies when reading from a field
    assert ak.array_equal(
        ntuple["my_struct"].arrays(filter_name="sub_struct"),
        ntuple.arrays(filter_name="my_struct.sub_struct"),
    )
    # keys only list what the filters select
    assert ntuple.keys(filter_name="my_struct") == ["my_struct"]
    for filename in [STL, VLV]:
        ntuple = _ntuple(filename)
        for name in ntuple.keys(recursive=False):
            arrays = ntuple.arrays(filter_name=name)
            assert arrays.fields == [name]
            assert ak.array_equal(arrays[name], ntuple[name].array()), name


def test_iterate():
    ntuple = _ntuple(NESTED)
    iterated = ak.concatenate(
        list(ntuple.iterate(filter_name="my_struct", step_size=3))
    )
    assert ak.array_equal(iterated.my_struct, ntuple["my_struct"].array())


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_matches_arrays(open_files):
    pytest.importorskip("dask_awkward")
    for filename, name in [
        (NESTED, "my_struct"),
        (NESTED, "my_struct.sub_struct.i"),
        # "pt" is also the name of fields in other collections
        (STL, "lorentz_vector.pt"),
        (VLV, "four_v_LVs.pt"),
    ]:
        path = skhep_testdata.data_path(filename)
        expected = uproot.open(path)["ntuple"].arrays(filter_name=name)
        lazy = uproot.dask({path: "ntuple"}, open_files=open_files, filter_name=name)
        assert lazy.fields == expected.fields, name
        assert ak.array_equal(lazy.compute(), expected), name


def test_dask_reads_common_subfields(tmp_path):
    pytest.importorskip("dask_awkward")
    files = []
    for name, other in [("a.root", "y"), ("b.root", "z")]:
        files.append({str(tmp_path / name): "ntuple"})
        with uproot.recreate(tmp_path / name) as f:
            f.mkrntuple(
                "ntuple", {"record": ak.zip({"x": [1, 2], other: [3, 4]}), "n": [0, 1]}
            )
    # only the subfields that every file has are read
    for selection, expected in [
        ({}, "{n: int64, record: {x: int64}}"),
        ({"filter_name": "record"}, "{record: {x: int64}}"),
    ]:
        lazy = uproot.dask(files, **selection)
        assert str(lazy.compute().type) == f"4 * {expected}"


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_selects_fields_by_full_path(tmp_path, open_files):
    pytest.importorskip("dask_awkward")
    # the top-level "x" has the same name as "rec.x"
    files = []
    for name, other, value in [("a.root", "b", 100), ("b.root", "c", 200)]:
        files.append({str(tmp_path / name): "ntuple"})
        with uproot.recreate(tmp_path / name) as f:
            rec_x = ak.zip({"a": [10, 10], other: [value, value]})
            f.mkrntuple("ntuple", {"x": [1, 2], "rec": ak.zip({"x": rec_x})})

    lazy = uproot.dask(
        files[0], open_files=open_files, filter_branch=lambda f: f.path == "x"
    )
    assert str(lazy.compute().type) == "2 * {x: int64}"

    # the subfields of "rec.x" that only some files have are not read as the common ones
    if open_files:
        out = uproot.dask(files).compute()
        assert str(out.type) == "4 * {rec: {x: {a: int64}}, x: int64}"
        assert out.rec.x.a.tolist() == [10] * 4
