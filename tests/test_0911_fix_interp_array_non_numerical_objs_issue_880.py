# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_fix_interpretation_for_arrays_of_nonnumerical_objects_issue_880_p2():
    with uproot.open(skhep_testdata.data_path("uproot-issue-880.root")) as file:
        branch = file["Z/Event/Cluster[6]"]
        array = branch.array(library="ak")
        interp = uproot.interpretation.identify.interpretation_of(
            branch, {}, False
        )  # AsObjects(AsArray(False, False, Model_zCluster, (6,)))

        assert len(array) == 116
        assert len(array[0][0]) == 6  # all 6 cluster can now be accessed
        assert str(interp) == "AsObjects(AsArray(False, False, Model_zCluster, (6,)))"

        first = array[0][0][0]
        assert (first.Status, first.Bin, first.Charge, first.Eff) == (1, 3, 1, 1.0)
        assert first.xTrue.tolist() == pytest.approx(
            {"fX": 0.21726272157226714, "fY": -0.42488166710952635, "fZ": 240.2}
        )
        assert first.xRec.tolist() == pytest.approx(
            {"fX": 0.04978510872240323, "fY": -0.7507430064136305, "fZ": 240.2}
        )
        # the last one of this entry is not filled
        assert array[0][0][5].Used == -1
        assert array[0][0][5].xTrue.tolist() == {"fX": 0.0, "fY": 0.0, "fZ": 0.0}
