# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata

import uproot


def test_UprootModelAdapter():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        blah = f["Events"]
        print(blah.all_members)
        print(blah.bases)
        print(blah.closed)
        print(blah.concrete)
        print(blah.cursor)
        print(blah.file)
        print(blah.instance_version)
        print(blah.is_memberwise)
        print(blah.members)
        print(blah.num_bytes)
        print(blah.parent)
