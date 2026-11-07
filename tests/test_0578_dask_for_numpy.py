# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_single_dask_array():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    np_arrays = ttree.arrays(library="np")
    dask_arrays = uproot.dask(test_path, library="np")

    assert list(dask_arrays.keys()) == list(
        np_arrays.keys()
    ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

    for key in np_arrays.keys():
        comp = dask_arrays[key].compute() == np_arrays[key]
        assert comp.all(), f"Incorrect array at key {key}"


def test_dask_concatenation():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = skhep_testdata.data_path("uproot-Zmumu-zlib.root") + ":events"
    test_path4 = skhep_testdata.data_path("uproot-Zmumu-lzma.root") + ":events"

    np_arrays = uproot.concatenate(
        [test_path1, test_path2, test_path3, test_path4], library="np"
    )
    dask_arrays = uproot.dask(
        [test_path1, test_path2, test_path3, test_path4], library="np"
    )

    assert list(dask_arrays.keys()) == list(
        np_arrays.keys()
    ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

    for key in np_arrays.keys():
        comp = dask_arrays[key].compute() == np_arrays[key]
        assert comp.all(), f"Incorrect array at key {key}"


def test_multidim_array():
    test_path = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )
    ttree = uproot.open(test_path)

    np_arrays = ttree.arrays(library="np")
    dask_arrays = uproot.dask(test_path, library="np")

    assert list(dask_arrays.keys()) == list(
        np_arrays.keys()
    ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

    assert dask_arrays["ab"].shape == (30, 3)
    computed = dask.compute(dask_arrays, scheduler="synchronous")[0]
    for key, expected in np_arrays.items():
        assert computed[key].shape == expected.shape, key
        assert [numpy.asarray(row).tolist() for row in computed[key]] == [
            numpy.asarray(row).tolist() for row in expected
        ], key


@pytest.mark.parametrize("num_files", [1, 2])
@pytest.mark.parametrize(
    ("step_size", "chunks_per_file"),
    [(30, (30,) * 76 + (24,)), ("500B", (3,) * 768)],
)
def test_chunking(num_files, step_size, chunks_per_file):
    files = [
        skhep_testdata.data_path(name) + ":events"
        for name in ("uproot-Zmumu.root", "uproot-Zmumu-uncompressed.root")
    ][:num_files]
    assert uproot.dask(files, step_size=step_size, library="np")["px1"].chunks == (
        chunks_per_file * num_files,
    )
