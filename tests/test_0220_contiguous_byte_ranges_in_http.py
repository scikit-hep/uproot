# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest

import uproot


@pytest.mark.network
def test():
    pytest.importorskip("aiohttp")

    with uproot.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/RD_distribution.root:tree"
    ) as f:
        whole_branch = f["vchi2_b"].array(library="np")
        assert whole_branch[0] == 5.234916687011719
        assert whole_branch[-1] == 12.466843605041504

        whole_branch = f["mu_pt_sum"].array(library="np")
        assert whole_branch[0] == 26.4675350189209
        assert whole_branch[-1] == 39.84319305419922
