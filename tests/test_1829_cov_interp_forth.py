# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of the Forth node tree in uproot._awkwardforth, independently of any
deserialization: how a tree of nodes derives its Awkward form.
"""

from __future__ import annotations

import pytest

from uproot._awkwardforth import ForthGenerator, Node, SpecialPathItem, add_to_path

numbers = {"class": "NumpyArray", "primitive": "int32", "form_key": "node1"}


def number_field(name):
    return Node(f"node-{name}", form_details=dict(numbers), field_name=name)


def test_special_path_item():
    item = SpecialPathItem("vector")

    assert repr(item) == "SpecialPathItem('vector')"
    assert item == SpecialPathItem("vector")
    assert item != SpecialPathItem("map")
    assert item != "vector"
    assert hash(item) == hash(SpecialPathItem("vector"))


def test_add_to_path():
    context = {"path": ("a",)}

    assert add_to_path(None, context, "b") is context
    assert add_to_path(object(), context, "b") == {"path": ("a", "b")}
    assert context == {"path": ("a",)}


def test_node_properties():
    node = Node("name")

    node.name = "other"
    node.field_name = "field"
    node.form_details = {"class": "NumpyArray"}

    assert node.name == "other"
    assert node.field_name == "field"
    assert node.form_details == {"class": "NumpyArray"}
    assert node.pre_code == node.post_code == node.init_code == node.header_code == []
    assert node.children == []


def test_generator_sets_active_nodes():
    generator = ForthGenerator(None)
    top = generator.model
    child = Node("child")

    generator.add_node(child)
    assert top.children == [child]
    assert generator.active_node is top

    # already there
    generator.add_node(Node("child"))
    assert top.children == [child]

    generator.push_active_node(child)
    assert generator.active_node is child
    generator.add_node(Node("grandchild"))
    assert [x.name for x in child.children] == ["grandchild"]

    generator.pop_active_node()
    assert generator.active_node is top

    generator.set_active_node(child)
    generator.reset_active_node()
    assert generator.active_node is top

    assert generator.final_code == []
    assert generator.final_header == []
    assert generator.final_init == []


@pytest.mark.parametrize(
    "node, expected",
    [
        (Node("x:prebuilt", form_details={"a": 1}), {"a": 1}),
        (Node("x", form_details=dict(numbers)), numbers),
        (
            Node(
                "x",
                form_details={
                    "class": "ListOffsetArray",
                    "content": dict(numbers),
                    "parameters": {"__array__": "string"},
                },
            ),
            {
                "class": "ListOffsetArray",
                "content": numbers,
                "parameters": {"__array__": "string"},
            },
        ),
        # not known yet
        (
            Node("x", form_details={"class": "ListOffsetArray"}),
            {"class": "ListOffsetArray", "content": "NULL"},
        ),
        (
            Node("x", form_details={"class": "RegularArray", "content": dict(numbers)}),
            {"class": "RegularArray", "content": numbers},
        ),
        (
            Node(
                "x",
                form_details={"class": "ListOffsetArray"},
                children=[Node("y", form_details=dict(numbers))],
            ),
            {"class": "ListOffsetArray", "content": numbers},
        ),
        (
            Node(
                "x",
                form_details={
                    "class": "ListOffsetArray",
                    "content": {"class": "RegularArray", "size": 2},
                },
                children=[Node("y", form_details=dict(numbers))],
            ),
            {
                "class": "ListOffsetArray",
                "content": {"class": "RegularArray", "size": 2, "content": numbers},
            },
        ),
        (
            Node(
                "x",
                form_details={
                    "class": "ListOffsetArray",
                    "content": {
                        "class": "RecordArray",
                        "parameters": {"__array__": "sorted_map"},
                    },
                },
                children=[
                    Node("key", form_details=dict(numbers)),
                    Node("value", form_details=dict(numbers)),
                ],
            ),
            {
                "class": "ListOffsetArray",
                "content": {
                    "class": "RecordArray",
                    "parameters": {"__array__": "sorted_map"},
                    "fields": None,
                    "contents": [numbers, numbers],
                },
            },
        ),
    ],
    ids=[
        "prebuilt",
        "numpy",
        "string",
        "unknown-list",
        "regular",
        "list",
        "list-of-regular",
        "map",
    ],
)
def test_derive_form(node, expected):
    assert node.derive_form() == expected


@pytest.mark.parametrize(
    "name",
    ["TOP", "dispatch-by-version X", "wrong-instance-version X", "start-of-model X"],
)
def test_derive_form_of_pass_through_nodes(name):
    node = Node(name, children=[Node("y", form_details=dict(numbers))])
    assert node.derive_form() == numbers


def base_class(wrapper, *fields):
    model = Node(
        "start-of-model Base",
        children=[Node("read-members Base", children=list(fields))],
    )
    if wrapper:
        model = Node("dispatch-by-version Base", children=[model])
    return Node("base-class Base", children=[model])


@pytest.mark.parametrize("wrapper", [False, True], ids=["plain", "versioned"])
def test_derive_form_of_records_with_base_classes(wrapper):
    node = Node(
        "read-members Thing",
        form_details={"parameters": {"__record__": "Thing"}},
        children=[
            base_class(wrapper, number_field("a"), number_field("b")),
            number_field("c"),
        ],
    )

    assert node.derive_form() == {
        "parameters": {"__record__": "Thing"},
        "class": "RecordArray",
        "fields": ["a", "b", "c"],
        "contents": [numbers, numbers, numbers],
    }


def test_derive_form_of_records_with_nested_field_names():
    # the name of the field is on the innermost node of the list
    field = Node(
        "outer-list",
        form_details={"class": "ListOffsetArray"},
        children=[
            Node(
                "inner-list",
                form_details={"class": "RegularArray", "size": 2},
                children=[number_field("deep")],
            )
        ],
    )
    node = Node("read-members Thing", children=[field])

    assert node.derive_form() == {
        "class": "RecordArray",
        "fields": ["deep"],
        "contents": [
            {
                "class": "ListOffsetArray",
                "content": {"class": "RegularArray", "size": 2, "content": numbers},
            }
        ],
    }
