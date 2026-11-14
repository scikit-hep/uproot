# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import uproot
import skhep_testdata


def test_fix_awkward_form_breadcrumbs():
    file = uproot.open(skhep_testdata.data_path("uproot-issue-880.root"))
    tree = file["Z"]
    assert tree.num_entries == 116

    # TVector3 appears in several members of zCluster: a repeated type, not recursion
    branch = tree["Event/Cluster[6]"]
    form = branch.interpretation.awkward_form(file.file)
    cluster = form.content.content
    assert cluster.parameters == {"__record__": "zCluster"}
    for name in ["xTrue", "xRec", "xTrue1", "xTrue2", "xPre1", "xPre2"]:
        vector = cluster.content(name)
        assert vector.parameters == {"__record__": "TVector3"}
        assert vector.fields == ["fX", "fY", "fZ"]
