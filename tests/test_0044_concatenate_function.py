# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

VERSIONS = [
    "5.23.02",
    "5.24.00",
    "5.25.02",
    "5.26.00",
    "5.27.02",
    "5.28.00",
    "5.29.02",
    "5.30.00",
    "6.08.04",
    "6.10.05",
    "6.14.00",
    "6.16.00",
    "6.18.00",
    "6.20.04",
]


@pytest.fixture(scope="module")
def files():
    # The glob only matches files that are already in the cache, so download all
    # of them first instead of relying on other tests having done it
    for version in VERSIONS:
        path = skhep_testdata.data_path(f"uproot-sample-{version}-uncompressed.root")
    return path.replace(VERSIONS[-1], "*")


def test_concatenate_numpy(files):
    arrays = uproot.concatenate({files: "sample"}, ["i8", "f8"], library="np")
    assert len(arrays["i8"]) == 420
    assert len(arrays["f8"]) == 420
    assert arrays["i8"][:30].tolist() == list(range(-15, 15))
    assert arrays["i8"].tolist() == arrays["i8"][:30].tolist() * 14
    assert arrays["f8"][:3].tolist() == [-14.9, -13.9, -12.9]


def test_concatenate_awkward(files):
    awkward = pytest.importorskip("awkward")
    arrays = uproot.concatenate({files: "sample"}, ["i8", "f8"], library="ak")
    assert isinstance(arrays, awkward.Array)
    assert set(awkward.fields(arrays)) == {"i8", "f8"}
    assert len(arrays) == 420


def test_concatenate_pandas(files):
    pandas = pytest.importorskip("pandas")
    arrays = uproot.concatenate({files: "sample"}, ["i8", "f8"], library="pd")
    assert isinstance(arrays, pandas.DataFrame)
    assert set(arrays.columns.tolist()) == {"i8", "f8"}
    assert len(arrays) == 420
