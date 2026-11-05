# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

SAMPLE_FILE = "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
MULTI_FIELD_FILE = "test_int_float_rntuple_v1-0-0-0.root"


INTEGERS = list(range(9, -1, -1))
FLOATS = [9.9, 8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0.0]
FIRST_NMUON = [2, 2, 1, 4, 4, 3, 2, 2, 2, 2]


@pytest.mark.parametrize("options", [{}, {"how": dict}])
def test_arrays_numpy_returns_dict(options):
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    with uproot.open(filename) as f:
        np_arrays = f["ntuple"].arrays(library="np", **options)
    assert type(np_arrays) is dict
    assert list(np_arrays) == ["one_integers", "two_floats"]
    assert np_arrays["one_integers"].dtype == np.dtype("int32")
    assert np_arrays["one_integers"].tolist() == INTEGERS
    assert np_arrays["two_floats"].dtype == np.dtype("float32")
    assert np_arrays["two_floats"].tolist() == pytest.approx(FLOATS)


@pytest.mark.parametrize("how", [tuple, list])
def test_arrays_numpy_how_sequence(how):
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    with uproot.open(filename) as f:
        result = f["ntuple"].arrays(library="np", how=how)
    assert type(result) is how
    assert len(result) == 2
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert result[0].tolist() == INTEGERS
    assert result[1].tolist() == pytest.approx(FLOATS)


def test_arrays_pandas_returns_dataframe():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    with uproot.open(filename) as f:
        result = f["ntuple"].arrays(library="pd")
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["one_integers", "two_floats"]
    assert result["one_integers"].tolist() == INTEGERS
    assert result["two_floats"].tolist() == pytest.approx(FLOATS)


def test_arrays_pandas_how_dict():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    with uproot.open(filename) as f:
        result = f["ntuple"].arrays(library="pd", how=dict)
    assert type(result) is dict
    assert list(result) == ["one_integers", "two_floats"]
    assert all(isinstance(v, pd.Series) for v in result.values())
    assert result["one_integers"].tolist() == INTEGERS
    assert result["two_floats"].tolist() == pytest.approx(FLOATS)


def test_single_field_array_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        result = f["Events"]["nMuon"].array(library="np")
    assert isinstance(result, np.ndarray)
    assert len(result) == 1000
    assert result[:10].tolist() == FIRST_NMUON


def test_single_field_array_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        result = f["Events"]["nMuon"].array(library="pd")
    assert isinstance(result, pd.Series)
    assert len(result) == 1000
    assert result.iloc[:10].tolist() == FIRST_NMUON


def test_iterate_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        chunks = list(f["Events"].iterate(step_size=100, library="np"))
    assert [type(chunk) for chunk in chunks] == [dict] * 10
    assert [len(chunk["nMuon"]) for chunk in chunks] == [100] * 10
    assert chunks[0]["nMuon"][:10].tolist() == FIRST_NMUON
    assert int(sum(chunk["nMuon"].sum() for chunk in chunks)) == 2372


def test_iterate_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        chunks = list(f["Events"].iterate(step_size=100, library="pd"))
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
    assert [len(chunk) for chunk in chunks] == [100] * 10
    assert chunks[0]["nMuon"].iloc[:10].tolist() == FIRST_NMUON
    assert int(sum(chunk["nMuon"].sum() for chunk in chunks)) == 2372


def test_concatenate_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    result = uproot.concatenate({filename: "Events"}, library="np")
    assert isinstance(result, dict)
    assert len(result["nMuon"]) == 1000
    assert result["nMuon"][:10].tolist() == FIRST_NMUON


def test_concatenate_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    result = uproot.concatenate({filename: "Events"}, library="pd")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1000
    assert result["nMuon"].iloc[:10].tolist() == FIRST_NMUON


def test_nested_structs_numpy():
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        result = f["ntuple"].arrays(library="np")
    assert list(result) == ["my_struct"]
    my_struct = result["my_struct"]
    assert isinstance(my_struct, np.ndarray)
    assert my_struct.shape == (10,)
    assert my_struct.dtype.names == ("i", "sub_struct")
    assert my_struct["i"][0] == 0
    assert my_struct["sub_struct"]["i"][0] == 1
    assert my_struct["sub_struct"]["sub_sub_struct"]["i"][0] == 2
    assert my_struct["sub_struct"]["sub_sub_struct"]["v"][0].tolist() == [0, 1]


def test_jagged_arrays_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        ak_arrays = obj.arrays(filter_name=["Muon_pt"])
        np_arrays = obj.arrays(library="np", filter_name=["Muon_pt"])

    assert isinstance(np_arrays, dict)
    # the name also matches the subfield that the top-level field is a projection of
    assert list(np_arrays) == ["_collection0", "Muon_pt"]
    assert isinstance(np_arrays["Muon_pt"], np.ndarray)
    assert np_arrays["Muon_pt"].dtype == object
    assert len(np_arrays["Muon_pt"]) == 1000
    assert np_arrays["Muon_pt"][0].tolist() == [10.763696670532227, 15.736522674560547]

    for i in range(10):
        assert ak.array_equal(ak_arrays["Muon_pt"][i], np_arrays["Muon_pt"][i])
