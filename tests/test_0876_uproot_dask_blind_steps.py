# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata
import awkward as ak

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize("library", ["np", "ak"])
@pytest.mark.parametrize(
    ("open_files", "steps", "num_partitions"),
    [
        (open_files, {"steps_per_file": n}, n)
        for open_files in (False, True)
        # uproot-Zmumu.root has one TBasket: all but 1 are blind steps
        for n in (1, 2, 15)
    ]
    + [(False, {}, 1), (True, {}, 1), (True, {"step_size": "100MB"}, 1)],
)
def test_uproot_dask_steps(library, open_files, steps, num_partitions):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    columns = ["Type", "Run", "px1", "py2"]
    with uproot.open(test_path) as tree:
        arrays = tree.arrays(columns, library=library)

    dask_arrays = uproot.dask(
        test_path, filter_name=columns, library=library, open_files=open_files, **steps
    )

    if library == "np":
        assert list(dask_arrays.keys()) == list(
            arrays.keys()
        ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"
        assert dask_arrays["px1"].numblocks == (num_partitions,)

        computed = dask.compute(dask_arrays, scheduler="synchronous")[0]
        for key in arrays.keys():
            assert numpy.array_equal(computed[key], arrays[key]), key

    else:
        assert dask_arrays.npartitions == num_partitions
        assert ak.array_equal(dask_arrays.compute(scheduler="synchronous"), arrays)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"step_size": "100MB", "open_files": False},
            "'step_size' should not be set when 'open_files' is False",
        ),
        (
            {"step_size": "100MB", "steps_per_file": 5, "open_files": False},
            "only 'step_size' or 'steps_per_file' should be set, not both",
        ),
        (
            {"step_size": "100MB", "steps_per_file": 5, "open_files": True},
            "only 'step_size' or 'steps_per_file' should be set, not both",
        ),
    ],
)
@pytest.mark.parametrize("library", ["np", "ak"])
def test_uproot_dask_incompatible_steps(library, kwargs, message):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    with pytest.raises(TypeError, match=message):
        uproot.dask(test_path, library=library, **kwargs)
