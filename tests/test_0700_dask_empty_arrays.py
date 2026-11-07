# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_numpy_empty_arrays(open_files):
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"

    dask_arrays = uproot.dask(test_path, library="np", open_files=open_files)

    assert list(dask_arrays) == ["int_branch", "float_branch"]
    computed = dask.compute(dask_arrays, scheduler="synchronous")[0]
    for key, dtype in [("int_branch", "int32"), ("float_branch", "float32")]:
        assert dask_arrays[key].dtype == dtype
        assert computed[key].dtype == dtype
        assert computed[key].shape == (0,)


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_awkward_empty_arrays(open_files):
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"

    dak_array = uproot.dask(test_path, library="ak", open_files=open_files)

    assert dak_array.npartitions == 1
    computed = dak_array.compute(scheduler="synchronous")
    assert str(computed.type) == "0 * {int_branch: int32, float_branch: float32}"


@pytest.mark.parametrize("library", ["np", "ak"])
def test_no_common_tree_branches(library):
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"

    with pytest.raises(ValueError, match="have no TBranches in common"):
        uproot.dask([test_path1, test_path2], library=library)
