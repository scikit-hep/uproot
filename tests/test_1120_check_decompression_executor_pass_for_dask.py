# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("dask")
pytest.importorskip("dask_awkward")


def test_decompression_executor_for_dask():

    class TestExecutor(uproot.source.futures.TrivialExecutor):
        def __init__(self):
            self.submit_count = 0

        def submit(self, task, /, *args, **kwargs):
            self.submit_count += 1
            super().submit(task, *args, **kwargs)

    testexecutor = TestExecutor()

    a = uproot.dask(
        {skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"): "sample"},
        decompression_executor=testexecutor,
    )

    assert a["i4"].compute(scheduler="synchronous").tolist() == list(range(-15, 15))

    # once per TBasket of the only TBranch read
    assert testexecutor.submit_count == 5
