# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import json

import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("string-example.root")) as file:
        assert json.loads(file["FileSummaryRecord"]) == {
            "LumiCounter.eventsByRun": {
                "counts": {},
                "empty": True,
                "type": "LumiEventCounter",
            },
            "guid": "5FE9437E-D958-11EE-AB88-3CECEF1070AC",
        }
