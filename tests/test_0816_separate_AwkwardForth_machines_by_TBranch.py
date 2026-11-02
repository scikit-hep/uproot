# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


@pytest.mark.parametrize("do_hack", [False, True])
def test(do_hack):
    vvvars = ["SumPtTrkPt500", "NumTrkPt1000", "TrackWidthPt1000"]
    with uproot.open(skhep_testdata.data_path("uproot-issue-816.root")) as f:
        t = f["AnalysisMiniTree"]
        vv_branches = [f"offline_akt4_pf_NOSYS_{var}" for var in vvvars]
        if do_hack:
            assert (
                str(t[f"offline_akt4_pf_NOSYS_SumPtTrkPt500"].array().type)
                == "100 * var * var * float32"
            )
            assert (
                str(t[f"offline_akt4_pf_NOSYS_NumTrkPt1000"].array().type)
                == "100 * var * var * int32"
            )
            assert (
                str(t[f"offline_akt4_pf_NOSYS_TrackWidthPt1000"].array().type)
                == "100 * var * var * float32"
            )

        jetinfo = t.arrays(vv_branches)

        assert (
            str(jetinfo.type)
            == "100 * {offline_akt4_pf_NOSYS_SumPtTrkPt500: var * var * float32, offline_akt4_pf_NOSYS_NumTrkPt1000: var * var * int32, offline_akt4_pf_NOSYS_TrackWidthPt1000: var * var * float32}"
        )

        # each TBranch must have kept its own type after reading them together
        assert (
            str(t[f"offline_akt4_pf_NOSYS_SumPtTrkPt500"].array().type)
            == "100 * var * var * float32"
        )
        assert (
            str(t[f"offline_akt4_pf_NOSYS_NumTrkPt1000"].array().type)
            == "100 * var * var * int32"
        )
        assert (
            str(t[f"offline_akt4_pf_NOSYS_TrackWidthPt1000"].array().type)
            == "100 * var * var * float32"
        )

        # three different branches, not one branch read three times
        assert jetinfo[vv_branches[0]][0, 0, 0] == 0.0
        assert jetinfo[vv_branches[1]][0, 0, 0] == 0
        assert jetinfo[vv_branches[2]][0, 0, 0] == -1.0
