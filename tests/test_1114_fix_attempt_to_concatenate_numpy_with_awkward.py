# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_partially_fix_issue_951():

    with uproot.open(
        skhep_testdata.data_path("uproot-issue-951.root") + ":CollectionTree"
    ) as tree:
        num_read = 0
        for key, branch in tree.iteritems(filter_typename="*ElementLink*"):
            # two branches have different issues, which are not fixed yet
            if key == "METAssoc_AnalysisMETAux./METAssoc_AnalysisMETAux.jetLink":
                with pytest.raises(ValueError, match="has the wrong number of bytes"):
                    branch.array()
            elif key == "EventInfoAuxDyn.hardScatterVertexLink":
                with pytest.raises(
                    uproot.interpretation.identify.UnknownInterpretation,
                    match="none of the rules matched",
                ):
                    branch.array()
            else:
                branch.interpretation._forth = True
                array = branch.array()
                assert len(array) == branch.num_entries
                assert [x.rsplit(".")[-1] for x in array.fields] == [
                    "m_persKey",
                    "m_persIndex",
                ]
                num_read += 1

        assert num_read == 80
