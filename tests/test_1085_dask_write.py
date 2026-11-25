import pytest
import awkward as ak
import skhep_testdata

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dd = pytest.importorskip("dask.distributed")
dask_awkward = pytest.importorskip("dask_awkward")

from distributed import Client
import uproot


def _assert_parts(directory, prefix, arr, partitions=2):
    """The parts are ``arr`` split evenly, in order, with nothing else in ``directory``."""
    expected_paths = {f"{prefix}-part{i}.root" for i in range(partitions)}
    assert {path.name for path in directory.iterdir()} == expected_paths

    stop = 0
    for part in range(partitions):
        start, stop = stop, stop + len(arr) // partitions + (
            part < len(arr) % partitions
        )
        with uproot.open(directory / f"{prefix}-part{part}.root") as file:
            written = file["tree"].arrays(arr.fields)
        assert ak.array_equal(
            written, arr[start:stop], dtype_exact=False, check_parameters=False
        )


def test_simple(tmp_path):
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=2)
    uproot.dask_write(dask_arr, tmp_path, prefix="data")
    _assert_parts(tmp_path, "data", arr)


def test_HZZ(tmp_path):
    """
    Write data from HZZ with 2 partitions.
    """
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as f:
        arr = f["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), 2)
    uproot.dask_write(dask_arr, tmp_path, prefix="data")
    _assert_parts(tmp_path, "data", arr)


def test_graph(tmp_path):
    """
    Test compute parameter (want to return highlevelgraph)
    """
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=2)
    graph = uproot.dask_write(
        dask_arr,
        str(tmp_path),
        prefix="compute",
        compute=False,
    )
    assert list(tmp_path.iterdir()) == []
    graph.compute()
    _assert_parts(tmp_path, "compute", arr)


@pytest.mark.distributed
def test_compute(tmp_path):
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as f:
        arr = f["events"].arrays(["Jet_Px", "MET_px"], entry_stop=101)
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), 2)
    # no worker process: its startup and shutdown time out on a busy machine
    with Client(
        n_workers=1, threads_per_worker=1, processes=False, dashboard_address=":0"
    ):
        graph = uproot.dask_write(
            dask_arr, str(tmp_path), prefix="distribute", compute=False
        )
        dask.compute(graph)
    _assert_parts(tmp_path, "distribute", arr)
