# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest

import uproot


def test(http_server):
    pytest.importorskip("aiohttp")
    with uproot.open({f"{http_server}/uproot-HZZ.root": "events"}) as t:
        met_px = t["MET_px"].array(library="np")
        assert len(met_px) == 2421
        assert met_px[:2].tolist() == [5.912771224975586, 24.76520347595215]
        assert met_px[-1] == 19.713748931884766

        met_py = t["MET_py"].array(library="np")
        assert len(met_py) == 2421
        assert met_py[:2].tolist() == [2.5636332035064697, -16.349109649658203]
        assert met_py[-1] == -3.5954182147979736
