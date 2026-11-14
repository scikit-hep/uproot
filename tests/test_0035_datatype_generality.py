# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_TH2_in_ttree():
    with uproot.open(skhep_testdata.data_path("uproot-issue-tbranch-of-th2.root"))[
        "g4SimHits/tree"
    ] as tree:
        assert (
            tree["histogram"].array(library="np")[0].member("fXaxis").member("fName")
            == "xaxis"
        )


def test_iofeatures_offsets():
    with uproot.open(skhep_testdata.data_path("uproot-small-dy-withoffsets.root"))[
        "tree/Muon_pt"
    ] as withoffsets:
        muonpt1 = withoffsets.array(library="np", entry_start=10, entry_stop=20)
        assert [x.tolist() for x in muonpt1] == [
            [20.60145378112793],
            [50.36957550048828, 41.21387481689453, 3.1869382858276367],
            [51.685970306396484, 35.227813720703125],
            [],
            [],
            [],
            [],
            [23.073759078979492],
            [32.921417236328125, 8.922308921813965, 4.368383407592773],
            [51.9132194519043, 31.930095672607422],
        ]

    with uproot.open(skhep_testdata.data_path("uproot-small-dy-nooffsets.root"))[
        "tree/Muon_pt"
    ] as nooffsets:
        muonpt2 = nooffsets.array(library="np", entry_start=10, entry_stop=20)
        assert [x.tolist() for x in muonpt2] == [
            [20.60145378112793],
            [50.36957550048828, 41.21387481689453, 3.1869382858276367],
            [51.685970306396484, 35.227813720703125],
            [],
            [],
            [],
            [],
            [23.073759078979492],
            [32.921417236328125, 8.922308921813965, 4.368383407592773],
            [51.9132194519043, 31.930095672607422],
        ]
