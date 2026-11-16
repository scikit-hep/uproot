# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import copy

from uproot._awkwardforth import Node


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

    # Calling derive_form() twice must give the same result...
    form1 = outer_node.derive_form()
    form2 = outer_node.derive_form()

    assert form1 == form2
    # ...and must not modify the node's own form_details
    assert outer_node.form_details == original_details
