# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest

import uproot

import skhep_testdata


@pytest.mark.network
def test_issue176():
    pytest.importorskip("aiohttp")

    with uproot.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree/Y1"].array(library="np")
        assert len(data) == 100000
        assert data[[0, -1]].tolist() == [0.440404057503, 0.405843913555]


@pytest.mark.network
def test_issue176_again():
    pytest.importorskip("aiohttp")

    with uproot.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree"].arrays(["Y1", "Y2"], library="np")
        assert len(data["Y1"]) == 100000
        assert len(data["Y2"]) == 100000
        assert data["Y1"][[0, -1]].tolist() == [0.440404057503, 0.405843913555]
        assert data["Y2"][[0, -1]].tolist() == [0.526256978512, 0.5391775369640001]


def test_issue121():
    with uproot.open(skhep_testdata.data_path("uproot-issue121.root")) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40
        assert data[:2].tolist() == [25.948482513427734, 40.51326370239258]
        assert data[-1] == 42.52241897583008
