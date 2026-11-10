# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle

import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue232.root"))["fTreeV0"] as t:
        branch = t["V0Hyper.fNsigmaHe3Pos"]
        unpickled = pickle.loads(pickle.dumps(branch))

        assert unpickled is not branch
        assert unpickled.name == "V0Hyper.fNsigmaHe3Pos"
        assert unpickled.num_entries == 11269
        assert unpickled.interpretation == branch.interpretation
        # the pickled TBranch reads for itself, with locks of its own
        original, copy = branch.array(library="np"), unpickled.array(library="np")
        assert [x.tolist() for x in copy] == [x.tolist() for x in original]
