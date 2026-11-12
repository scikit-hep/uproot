# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dask_awkward = pytest.importorskip("dask_awkward")


def test_single_delay_open():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(filename1)

    assert uproot.dask(filename1, open_files=False, library="np")["px1"].chunks == (
        (numpy.nan,),
    )
    arr = uproot.dask(filename1, open_files=False, library="np")[
        "px1"
    ].compute() == ttree["px1"].array(library="np")
    assert arr.all()


def test_multiple_delay_open():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    true_val = uproot.concatenate([filename1, filename2], "px1", library="np")["px1"]

    daskarr = uproot.dask([filename1, filename2], open_files=False, library="np")["px1"]
    assert daskarr.chunks == ((numpy.nan, numpy.nan),)
    assert numpy.array_equal(daskarr.compute(), true_val)


@pytest.mark.parametrize("open_files", [False, True])
@pytest.mark.parametrize("library", ["np", "ak"])
def test_supplied_steps(open_files, library):
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root")
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")
    true_val = uproot.concatenate(
        [filename1 + ":events", filename2 + ":events"], "px1", library=library
    )["px1"]

    files = [filename1, filename2]
    daskarr = uproot.dask(files, open_files=open_files, library=library)["px1"]

    if library == "ak":
        if open_files:
            assert daskarr.divisions == (0, 2304, 4608)
        else:
            assert daskarr.divisions == (None, None, None)
    else:
        if open_files:
            assert daskarr.chunks == ((2304, 2304),)
        else:
            assert daskarr.chunks == ((numpy.nan, numpy.nan),)

    assert daskarr.compute().tolist() == true_val.tolist()

    steps1 = [0, 1000, 2304]
    steps2 = [[0, 1200], [1200, 2304]]
    files = {
        filename1: {"object_path": "events", "steps": steps1},
        filename2: {"object_path": "events", "steps": steps2},
    }
    daskarr = uproot.dask(files, open_files=open_files, library=library)["px1"]

    if library == "ak":
        if open_files:
            assert daskarr.divisions == (0, 1000, 2304, 3504, 4608)
        else:
            assert daskarr.divisions == (0, 1000, 2304, 3504, 4608)
    else:
        if open_files:
            assert daskarr.chunks == ((1000, 1304, 1200, 1104),)
        else:
            assert daskarr.chunks == ((1000, 1304, 1200, 1104),)

    assert daskarr.compute().tolist() == true_val.tolist()
