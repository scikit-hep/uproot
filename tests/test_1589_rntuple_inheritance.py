# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
import uproot
import skhep_testdata

import awkward as ak


def test_inherited_fields():

    # The inheritance structure of the test RNTuple is as follows:
    # GrandChild -> Child -> BaseA
    # MultiParent -> BaseA, BaseB
    # MultiGrandParent -> Child(-> BaseA), MultiParent(-> BaseA, BaseB)

    BaseA_fields = frozenset(["base_a1", "base_a2", "base_a3"])
    BaseB_fields = frozenset(["base_b"])
    Child_fields = frozenset(["child_1", "child_2", *BaseA_fields])
    GrandChild_fields = frozenset(["grandchild_1", "grandchild_2", *Child_fields])
    MultiParent_fields = frozenset(
        ["multi_parent_1", "multi_parent_2", *BaseA_fields, *BaseB_fields]
    )

    # For MultiGrandParent, BaseA fields appear twice,
    # so we need to add prefixes to distinguish them.
    MultiGrandParent_fields = frozenset(
        [
            "multi_grand_parent1",
            "multi_grand_parent2",
            "multi_parent_1",
            "multi_parent_2",
            "child_1",
            "child_2",
            *BaseB_fields,
        ]
        + [f"Child::{field}" for field in BaseA_fields]
        + [f"MultiParent::{field}" for field in BaseA_fields]
    )

    filepath = skhep_testdata.data_path("test_class_inheritance_rntuple_v1-0-0-1.root")
    obj = uproot.open(filepath)["rntpl"]

    assert frozenset(obj["child"].keys()) == Child_fields
    assert frozenset(obj["grandchild"].keys()) == GrandChild_fields
    assert frozenset(obj["multi_parent"].keys()) == MultiParent_fields
    assert frozenset(obj["multi_grandparent"].keys()) == MultiGrandParent_fields

    arrays = obj.arrays()

    assert frozenset(arrays["child"].fields) == Child_fields
    assert frozenset(arrays["grandchild"].fields) == GrandChild_fields
    assert frozenset(arrays["multi_parent"].fields) == MultiParent_fields
    assert frozenset(arrays["multi_grandparent"].fields) == MultiGrandParent_fields

    assert ak.array_equal(
        arrays.multi_grandparent["Child::base_a1"],
        [i for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent["Child::base_a2"],
        [i * 0.1 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent["Child::base_a3"],
        [[i * j for j in range(3)] for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent["MultiParent::base_a1"],
        [i for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent["MultiParent::base_a2"],
        [i * 0.1 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent["MultiParent::base_a3"],
        [[i * j for j in range(3)] for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.base_b,
        [i * 10.0 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.child_1, [i * 2 for i in range(10)], dtype_exact=False
    )
    assert ak.array_equal(
        arrays.multi_grandparent.child_2,
        [i * 20.0 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.multi_parent_1,
        [i * 4 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.multi_parent_2,
        [i * 40.0 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.multi_grand_parent1,
        [i * 5 for i in range(10)],
        dtype_exact=False,
    )
    assert ak.array_equal(
        arrays.multi_grandparent.multi_grand_parent2,
        [i * 50.0 for i in range(10)],
        dtype_exact=False,
    )
