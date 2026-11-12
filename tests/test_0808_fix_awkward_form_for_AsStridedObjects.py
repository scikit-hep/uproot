# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


def test():
    path = skhep_testdata.data_path("uproot-HZZ-objects.root")
    with uproot.open(path)["events"] as tree:
        form = tree["jetp4"].interpretation.awkward_form(None)
        assert form.to_dict() == {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "RecordArray",
                "fields": ["fP", "fE"],
                "contents": [
                    {
                        "class": "RecordArray",
                        "fields": ["fX", "fY", "fZ"],
                        "contents": [
                            {
                                "class": "NumpyArray",
                                "primitive": "float64",
                                "inner_shape": [],
                                "parameters": {},
                                "form_key": None,
                            },
                            {
                                "class": "NumpyArray",
                                "primitive": "float64",
                                "inner_shape": [],
                                "parameters": {},
                                "form_key": None,
                            },
                            {
                                "class": "NumpyArray",
                                "primitive": "float64",
                                "inner_shape": [],
                                "parameters": {},
                                "form_key": None,
                            },
                        ],
                        "parameters": {"__record__": "TVector3"},
                        "form_key": None,
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "float64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": None,
                    },
                ],
                "parameters": {"__record__": "TLorentzVector"},
                "form_key": None,
            },
            "parameters": {},
            "form_key": None,
        }
