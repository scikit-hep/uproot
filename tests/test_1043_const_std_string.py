# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import skhep_testdata
import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-1043.root")) as file:
        tree = file["FooBar"]
        assert tree["Foo"].branches[0].typename == "std::string"
        assert tree["Foo"].branches[0].array().tolist() == [""]
