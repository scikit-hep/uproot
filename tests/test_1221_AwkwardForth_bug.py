# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-1221.root")) as file:
        record = file["TrkAna"]["trkana"]["demtsh"].array()[-1, -1, -1]
        assert record.tolist() == {
            "wdot": -0.6311486959457397,
            "dhit": True,
            "poca": {
                "fCoordinates": {
                    "fX": -526.5504760742188,
                    "fY": -195.0541534423828,
                    "fZ": 1338.90771484375,
                }
            },
            "dactive": True,
        }
