# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")

from dask_awkward.lib.testutils import assert_eq


def test_single_dask_awkward_array():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert_eq(dak_array, ak_array)


def test_dask_concatenation():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = skhep_testdata.data_path("uproot-Zmumu-zlib.root") + ":events"
    test_path4 = skhep_testdata.data_path("uproot-Zmumu-lzma.root") + ":events"

    ak_array = uproot.concatenate([test_path1, test_path2, test_path3, test_path4])
    dak_array = uproot.dask(
        [test_path1, test_path2, test_path3, test_path4], library="ak"
    )

    assert_eq(dak_array, ak_array)


def test_multidim_array():
    test_path = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert len(ak_array) == len(dak_array)
    assert_eq(dak_array, ak_array)


@pytest.mark.parametrize("num_files", [1, 2])
@pytest.mark.parametrize(
    ("step_size", "partitions_per_file"), [(10, 231), ("500B", 768)]
)
def test_chunking(num_files, step_size, partitions_per_file):
    files = [
        skhep_testdata.data_path(name) + ":events"
        for name in ("uproot-Zmumu.root", "uproot-Zmumu-uncompressed.root")
    ][:num_files]
    lazy = uproot.dask(files, step_size=step_size, library="ak")
    assert lazy.npartitions == partitions_per_file * num_files
    assert lazy.divisions[-1] == 2304 * num_files


def test_delay_open():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = skhep_testdata.data_path("uproot-Zmumu-zlib.root") + ":events"
    test_path4 = skhep_testdata.data_path("uproot-Zmumu-lzma.root") + ":events"

    ak_array = uproot.concatenate([test_path1, test_path2, test_path3, test_path4])
    dak_array = uproot.dask(
        [test_path1, test_path2, test_path3, test_path4], open_files=False, library="ak"
    )

    assert_eq(dak_array, ak_array)
