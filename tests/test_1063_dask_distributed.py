import pytest
import skhep_testdata

import uproot
import uproot.source.file
import uproot.source.fsspec

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")
dask_distributed = pytest.importorskip("dask.distributed")


@pytest.mark.distributed
@pytest.mark.parametrize(
    "handler",
    [None, uproot.source.file.MemmapSource, uproot.source.fsspec.FSSpecSource],
)
def test_issue_1063(handler):
    # any file does, and the TTree is pickled: this one has few TBranches
    file_path = skhep_testdata.data_path("uproot-HZZ.root")

    expected = uproot.concatenate({file_path: "events"}, "Muon_Px").Muon_Px

    # a worker process only adds its startup and shutdown, which time out on a busy
    # machine: the graph is pickled, with the handler's Source, without one too
    with dask_distributed.Client(
        n_workers=1, threads_per_worker=1, processes=False, dashboard_address=":0"
    ):
        events = uproot.dask(
            {file_path: "events"}, filter_name="Muon_Px", handler=handler
        )
        (computed,) = dask.compute(events.Muon_Px)

    assert computed.to_list() == expected.to_list()
