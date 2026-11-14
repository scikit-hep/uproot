# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata
import copy
from uproot._awkwardforth import Node


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


def test_derive_form_nested_list_mutation():
    nested_form_details = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "form_key": "node-leaf",
                "itemsize": 8,
                "format": "d",
                "shape": [],
                "inner_shape": [],
                "endian": ">",
                "parameters": {},
            },
        },
    }

    leaf_node = Node("leaf", form_details=nested_form_details["content"]["content"])
    inner_node = Node(
        "inner", form_details=nested_form_details["content"], children=[leaf_node]
    )
    outer_node = Node("outer", form_details=nested_form_details, children=[inner_node])

    original_details = copy.deepcopy(outer_node.form_details)

    # Call derive_form() twice
    form1 = outer_node.derive_form()
    form2 = outer_node.derive_form()

    # Assertions
    assert form1 == form2
    assert outer_node.form_details == original_details
