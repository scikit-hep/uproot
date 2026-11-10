# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot
from skhep_testdata import data_path


def test_fix_906_members_non_numerical_branches():
    filename = data_path("uproot-issue-798.root")  # PHYSLITE example file
    f = uproot.open(filename)
    tree = f["CollectionTree"]

    assert (
        str(tree["EventInfo"].interpretation)
        == "AsStridedObjects(Model_xAOD_3a3a_EventInfo_5f_v1_v1)"
    )
    # all of the content of EventInfo is in the Aux. branches
    assert tree["EventInfo"].interpretation.members == []
    assert tree["EventInfo"].num_entries == 40
