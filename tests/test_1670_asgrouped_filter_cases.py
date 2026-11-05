# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Tests for the four semantically distinct cases that arise when a filter_name
matches an AsGrouped branch (parent), its children, or both.

The four cases and their expected behaviour:

  Case 1 — only parent matched:
      Return all sub-branches grouped as a RecordArray under the parent's name.

  Case 2 — only a subset of children matched (no parent):
      Return the matched children individually as flat top-level fields.

  Case 3 — parent AND all children matched:
      Return all children individually (flat); the parent is dropped because
      every leaf is already captured, so grouping would be redundant.

  Case 4 — parent AND some (but not all) children matched:
      Return ALL sub-branches grouped as a RecordArray under the parent's name;
      the individually-matched children are absorbed into the group and are not
      also returned as flat fields.

All four cases are tested for eager (tree.arrays), virtual (tree.arrays
virtual=True), and dask (uproot.dask) modes, and the three modes are verified
to return identical arrays.
"""

import pytest
import skhep_testdata
import awkward

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


GENPARTICLES_CORE = [
    "genParticles.core.pdgId",
    "genParticles.core.charge",
    "genParticles.core.status",
    "genParticles.core.vertex.x",
    "genParticles.core.vertex.y",
    "genParticles.core.vertex.z",
    "genParticles.core.p4.mass",
    "genParticles.core.p4.px",
    "genParticles.core.p4.py",
    "genParticles.core.p4.pz",
    "genParticles.core.bits",
]


def fcc_path():
    return skhep_testdata.data_path("uproot-FCCDelphesOutput.root")


def fcc_tree(f):
    return f["events"]


# ---------------------------------------------------------------------------
# Case 1: only parent matched → all sub-branches grouped
# ---------------------------------------------------------------------------


def test_case1_only_parent_eager():
    """filter_name matching only the AsGrouped parent returns all children grouped."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(filter_name="genParticles")

    assert arr.fields == ["genParticles"]
    sub = arr["genParticles"]
    assert sub.fields == GENPARTICLES_CORE
    assert sub["genParticles.core.pdgId"][0][:3].tolist() == [11, -11, 11]


def test_case1_only_parent_virtual():
    """Virtual mode matches eager mode for case 1."""
    path = fcc_path()
    with uproot.open(path) as f:
        tree = fcc_tree(f)
        arr = tree.arrays(filter_name="genParticles")
        virtual_arr = tree.arrays(virtual=True, filter_name="genParticles")
        assert awkward.array_equal(arr, awkward.materialize(virtual_arr))


def test_case1_only_parent_dask():
    """Dask mode matches eager mode for case 1."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(filter_name="genParticles")

    dask_arr = uproot.dask(path + ":events", filter_name="genParticles")
    assert dask_arr.fields == ["genParticles"]
    assert awkward.array_equal(arr, dask_arr.compute())


# ---------------------------------------------------------------------------
# Case 2: only a subset of children matched → returned ungrouped (flat)
# ---------------------------------------------------------------------------


def test_case2_only_children_eager():
    """filter_name matching only a child branch returns it as a flat field."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(filter_name="genParticles.core.pdgId")

    assert arr.fields == ["genParticles.core.pdgId"]
    assert arr["genParticles.core.pdgId"][0][:3].tolist() == [11, -11, 11]


def test_case2_only_children_virtual():
    """Virtual mode matches eager mode for case 2."""
    path = fcc_path()
    with uproot.open(path) as f:
        tree = fcc_tree(f)
        arr = tree.arrays(filter_name="genParticles.core.pdgId")
        virtual_arr = tree.arrays(virtual=True, filter_name="genParticles.core.pdgId")
        assert awkward.array_equal(arr, awkward.materialize(virtual_arr))


def test_case2_only_children_dask():
    """Dask mode matches eager mode for case 2."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(filter_name="genParticles.core.pdgId")

    dask_arr = uproot.dask(path + ":events", filter_name="genParticles.core.pdgId")
    assert dask_arr.fields == ["genParticles.core.pdgId"]
    assert awkward.array_equal(arr, dask_arr.compute())


# ---------------------------------------------------------------------------
# Case 3: parent AND all children matched → children returned ungrouped (flat)
# ---------------------------------------------------------------------------


def test_case3_parent_and_all_children_eager():
    """filter_name matching parent AND all children drops the parent and returns children flat."""
    path = fcc_path()
    with uproot.open(path) as f:
        # "genParticles*" matches both the parent and all 11 sub-branches
        arr = fcc_tree(f).arrays(filter_name="genParticles*")

    # parent should NOT appear; all 11 core sub-branches should be flat
    assert "genParticles" not in arr.fields
    core_fields = [f for f in arr.fields if f.startswith("genParticles.core")]
    assert core_fields == GENPARTICLES_CORE
    assert arr["genParticles.core.pdgId"][0][:3].tolist() == [11, -11, 11]


def test_case3_parent_and_all_children_virtual():
    """Virtual mode matches eager mode for case 3."""
    path = fcc_path()
    with uproot.open(path) as f:
        tree = fcc_tree(f)
        arr = tree.arrays(filter_name="genParticles*")
        virtual_arr = tree.arrays(virtual=True, filter_name="genParticles*")
        # compare only the genParticles.core fields (filter also picks up #0/#1 counters)
        core = [f for f in arr.fields if f.startswith("genParticles.core")]
        mat = awkward.materialize(virtual_arr)
        assert awkward.array_equal(arr[core], mat[core])


def test_case3_parent_and_all_children_dask():
    """Dask mode matches eager mode for case 3."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(filter_name="genParticles*")

    dask_arr = uproot.dask(path + ":events", filter_name="genParticles*")
    assert "genParticles" not in dask_arr.fields
    core = [f for f in arr.fields if f.startswith("genParticles.core")]
    computed = dask_arr.compute()
    assert awkward.array_equal(arr[core], computed[core])


# ---------------------------------------------------------------------------
# Case 4: parent AND some children matched → ALL children returned grouped
# ---------------------------------------------------------------------------


def test_case4_parent_and_some_children_eager():
    """filter_name matching parent AND some children returns ALL children grouped."""
    path = fcc_path()
    with uproot.open(path) as f:
        # parent + exactly 1 of 11 children
        arr = fcc_tree(f).arrays(
            filter_name=["genParticles", "genParticles.core.pdgId"]
        )

    # result must be a single grouped field, not the individual child
    assert arr.fields == ["genParticles"]
    sub = arr["genParticles"]
    # ALL 11 sub-branches should be present (not just the one that was in the filter)
    assert sub.fields == GENPARTICLES_CORE
    assert sub["genParticles.core.pdgId"][0][:3].tolist() == [11, -11, 11]


def test_case4_parent_and_some_children_virtual():
    """Virtual mode matches eager mode for case 4."""
    path = fcc_path()
    with uproot.open(path) as f:
        tree = fcc_tree(f)
        arr = tree.arrays(filter_name=["genParticles", "genParticles.core.pdgId"])
        virtual_arr = tree.arrays(
            virtual=True, filter_name=["genParticles", "genParticles.core.pdgId"]
        )
        assert awkward.array_equal(arr, awkward.materialize(virtual_arr))


def test_case4_parent_and_some_children_dask():
    """Dask mode matches eager mode for case 4, and does not crash."""
    path = fcc_path()
    with uproot.open(path) as f:
        arr = fcc_tree(f).arrays(
            filter_name=["genParticles", "genParticles.core.pdgId"]
        )

    dask_arr = uproot.dask(
        path + ":events", filter_name=["genParticles", "genParticles.core.pdgId"]
    )
    assert dask_arr.fields == ["genParticles"]
    computed = dask_arr.compute()
    assert awkward.array_equal(arr, computed)


# ---------------------------------------------------------------------------
# Consistency: case 1 and case 4 should return the same result because both
# include the full grouped record (case 4 merely happened to also match some
# individual children, but they are absorbed into the group).
# ---------------------------------------------------------------------------


def test_case1_and_case4_return_same_grouped_record():
    """Case 1 (parent only) and case 4 (parent + some children) yield identical arrays."""
    path = fcc_path()
    with uproot.open(path) as f:
        tree = fcc_tree(f)
        arr_case1 = tree.arrays(filter_name="genParticles")
        arr_case4 = tree.arrays(filter_name=["genParticles", "genParticles.core.pdgId"])

    assert arr_case1.fields == arr_case4.fields == ["genParticles"]
    assert awkward.array_equal(arr_case1, arr_case4)
