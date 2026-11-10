# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import awkward

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_read_AsGrouped_with_dask_or_virtual_array():
    """
    Reading a TTree that contains an AsGrouped branch without any filter must
    return consistent results across eager, virtual, and dask modes.
    """
    path = skhep_testdata.data_path("uproot-issue-1502.root")
    with uproot.open(path) as f:
        arr = f["tree"].arrays()
        virtual_arr = f["tree"].arrays(virtual=True)

        assert awkward.array_equal(arr, awkward.materialize(virtual_arr))

    assert arr.value.tolist() == list(range(10))

    dask_arr = uproot.dask(path).compute()
    assert awkward.array_equal(arr, dask_arr)


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize(
    ("full_paths", "value"), [(False, "value"), (True, "branch/value")]
)
def test_read_AsGrouped_with_dask_numpy(open_files, full_paths, value):
    """
    As in an eager read of all of a TTree, a group of TBranches that are also
    selected is left out: it has no NumPy dtype.
    """
    path = skhep_testdata.data_path("uproot-issue-1502.root")
    lazy = uproot.dask(path, library="np", open_files=open_files, full_paths=full_paths)
    assert len(lazy) == 3
    assert value in lazy
    computed = dask.compute(lazy, scheduler="synchronous")[0]
    assert computed[value].tolist() == list(range(10))


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize("names", ["branch", ["branch", "value"]])
def test_AsGrouped_alone_cannot_be_a_dask_array(open_files, names):
    path = skhep_testdata.data_path("uproot-issue-1502.root")
    with pytest.raises(
        NotImplementedError, match="cannot read the grouped TBranch 'branch'"
    ):
        uproot.dask(path, filter_name=names, library="np", open_files=open_files)
