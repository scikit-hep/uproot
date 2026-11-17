# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import copy

from uproot._awkwardforth import Node


def test_derive_form_fixed_size_array_of_lists():
    # AsArray with an inner shape (e.g. std::vector<float> fV[3] in a split
    # TClonesArray) puts ListOffsetArray -> RegularArray in a single node, whose
    # child is another list node.
    leaf = Node(
        "node3",
        form_details={
            "class": "NumpyArray",
            "primitive": "float32",
            "form_key": "node3",
        },
    )
    vector = Node(
        "node2",
        form_details={
            "class": "ListOffsetArray",
            "offsets": "i64",
            "parameters": {},
            "form_key": "node2",
        },
        children=[leaf],
    )
    array = Node(
        "node1",
        form_details={
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {"class": "RegularArray", "size": 3},
            "parameters": {},
            "form_key": "node1",
        },
        children=[vector],
    )

    original = copy.deepcopy(array.form_details)

    # derive_form is called once per entry until the form has no NULLs
    forms = [array.derive_form() for _ in range(3)]

    assert array.form_details == original
    assert forms[0] == forms[1] == forms[2]

    regular = forms[0]["content"]
    assert regular["class"] == "RegularArray"
    assert regular["content"]["form_key"] == "node2"
    assert regular["content"]["content"] == leaf.form_details
