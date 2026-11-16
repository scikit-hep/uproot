# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import awkward as ak
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")


def test_column_projection_sanity_check():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    columns = ["px1", "px2", "py1", "py2"]
    projected = dak_array[columns]
    (necessary,) = dask_awkward.necessary_columns(projected).values()
    assert necessary == frozenset(columns)
    assert ak.almost_equal(
        projected.compute(scheduler="synchronous"), ak_array[columns]
    )
