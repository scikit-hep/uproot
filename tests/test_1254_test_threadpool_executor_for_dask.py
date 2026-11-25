import os

import pytest
import skhep_testdata

import uproot

pytest.importorskip("dask")
pytest.importorskip("dask_awkward")


@pytest.mark.parametrize(
    ("max_workers", "expected_workers"), [(None, os.cpu_count()), (1, 1)]
)
def test_decompression_threadpool_executor_for_dask(max_workers, expected_workers):

    class TestThreadPoolExecutor(uproot.source.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None):
            super().__init__(max_workers=max_workers)
            self.submit_count = 0

        def submit(self, task, /, *args, **kwargs):
            self.submit_count += 1
            super().submit(task, *args, **kwargs)

    executor = TestThreadPoolExecutor(max_workers=max_workers)

    a = uproot.dask(
        {skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"): "sample"},
        decompression_executor=executor,
    )

    assert a["i4"].compute(scheduler="synchronous").tolist() == list(range(-15, 15))

    assert executor.max_workers == expected_workers

    # once per TBasket of the only TBranch read
    assert executor.submit_count == 5

    executor.shutdown()
