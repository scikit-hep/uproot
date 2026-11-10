# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import numpy
import awkward

import uproot

# ==============================================================================
# SECTION 1: TTree Slicing Tests (Legacy Format)
# ==============================================================================


@pytest.fixture(scope="module")
def tree():
    """Fixture to open the standard HZZ TTree test file once for this module."""
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as file:
        yield file["events"]


TTREE_BRANCH_ARCHETYPES = [
    pytest.param("NJet", id="TTree-Flat-Int"),
    pytest.param("Muon_Px", id="TTree-Jagged-Float"),
]


@pytest.mark.parametrize("branch_name", TTREE_BRANCH_ARCHETYPES)
@pytest.mark.parametrize("beyond", [0, 100], ids=["exact-end", "far-beyond"])
def test_ttree_read_beyond_the_end(tree, branch_name, beyond):
    """Verify that reading at or beyond the end of a TTree returns an empty array."""
    total = tree.num_entries
    data = tree[branch_name].array(
        entry_start=total + beyond, entry_stop=total + beyond + 100
    )
    assert data.tolist() == []


@pytest.mark.parametrize("branch_name", TTREE_BRANCH_ARCHETYPES)
def test_ttree_clamped_out_of_bounds(tree, branch_name):
    """Verify that a TTree slice overlapping the end is correctly clamped."""
    total = tree.num_entries
    data = tree[branch_name].array(entry_start=total - 5, entry_stop=total + 100)
    assert data.tolist() == tree[branch_name].array().tolist()[-5:]


@pytest.mark.parametrize("branch_name", TTREE_BRANCH_ARCHETYPES)
def test_ttree_massive_negative_index(tree, branch_name):
    """Verify a large negative start index on a TTree returns the whole array."""
    total = tree.num_entries
    data = tree[branch_name].array(entry_start=-1_000_000)
    assert len(data) == total
    assert data.tolist() == tree[branch_name].array().tolist()


@pytest.mark.parametrize("branch_name", TTREE_BRANCH_ARCHETYPES)
def test_ttree_iterate_out_of_bounds(tree, branch_name):
    """Verify that iterating a TTree over an invalid range yields zero batches."""
    total = tree.num_entries
    iterator = tree.iterate(
        [branch_name], step_size=100, entry_start=total + 100, entry_stop=total + 200
    )
    assert len(list(iterator)) == 0


# ==============================================================================
# SECTION 2: RNTuple Slicing Tests (Modern Format)
# ==============================================================================


@pytest.fixture(scope="module")
def rntuple():
    """Fixture to open a standard RNTuple test file from skhep_testdata."""
    filename = "test_1jag_int_float_rntuple_v1-0-0-0.root"
    with uproot.open(skhep_testdata.data_path(filename)) as file:
        yield file["ntuple"]


RNTUPLE_BRANCH_ARCHETYPES = [
    pytest.param("one_v_integers", id="RNTuple-Jagged-Int"),
    pytest.param("two_v_floats", id="RNTuple-Jagged-Float"),
]


@pytest.mark.parametrize("branch_name", RNTUPLE_BRANCH_ARCHETYPES)
@pytest.mark.parametrize("beyond", [0, 100], ids=["exact-end", "far-beyond"])
def test_rntuple_read_beyond_the_end(rntuple, branch_name, beyond):
    """Verify that reading at or beyond the end of an RNTuple returns an empty array."""
    total = rntuple.num_entries
    branch = rntuple[branch_name]
    data = branch.array(entry_start=total + beyond, entry_stop=total + beyond + 100)
    assert data.tolist() == []


@pytest.mark.parametrize("branch_name", RNTUPLE_BRANCH_ARCHETYPES)
def test_rntuple_clamped_out_of_bounds(rntuple, branch_name):
    """Verify that a slice overlapping the end of an RNTuple is correctly clamped."""
    total = rntuple.num_entries
    branch = rntuple[branch_name]
    data = branch.array(entry_start=total - 5, entry_stop=total + 100)
    assert data.tolist() == branch.array().tolist()[-5:]


@pytest.mark.parametrize("branch_name", RNTUPLE_BRANCH_ARCHETYPES)
def test_rntuple_massive_negative_index(rntuple, branch_name):
    """Verify a large negative start index on an RNTuple returns the whole array."""
    total = rntuple.num_entries
    branch = rntuple[branch_name]
    data = branch.array(entry_start=-1_000_000)
    assert len(data) == total
    assert data.tolist() == branch.array().tolist()


@pytest.mark.parametrize("branch_name", RNTUPLE_BRANCH_ARCHETYPES)
def test_rntuple_iterate_out_of_bounds(rntuple, branch_name):
    """
    Verify that iterating over an invalid range in an RNTuple yields zero batches.
    This acts as a regression test for the fix in behaviors/RNTuple.py.
    """
    total = rntuple.num_entries

    iterator = rntuple.iterate(
        [branch_name],
        step_size=50,
        entry_start=total + 100,
        entry_stop=total + 200,
        library="ak",
    )

    batches = list(iterator)
    assert len(batches) == 0
