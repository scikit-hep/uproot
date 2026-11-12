# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")


def test_pandas_performance_many_branches(tmp_path):
    frames = list(
        uproot.iterate(
            skhep_testdata.data_path("uproot-issue-1070.root") + ":tree",
            step_size=100,
            library="pandas",
        )
    )

    assert len(frames) == 10
    for i, frame in enumerate(frames):
        assert frame.shape == (100, 101)
        assert frame.columns[:3].tolist() == ["index", "column_0", "column_1"]
        assert frame.index.tolist() == list(range(100 * i, 100 * (i + 1)))
        assert frame["index"].tolist() == frame.index.tolist()
