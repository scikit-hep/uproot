# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import awkward as ak
import pytest
import skhep_testdata

import uproot

pytest.importorskip("dask")
pytest.importorskip("dask_awkward")


# uproot.dask only tokenizes the open TTrees of open_files=True, which is too slow with the
# 1833 TBranches of the first file, and it's the same code as for the last one
@pytest.mark.parametrize(
    ("file", "tree", "branch", "open_files"),
    [
        ("uproot-metadata-performance.root", "Events", "FatJet_btagDDBvLV2", False),
        ("uproot-issue513.root", "Delphes", "Particle", False),
        ("uproot-issue443.root", "muonDataTree", "hitEnd", False),
        ("uproot-issue443.root", "muonDataTree", "hitEnd", True),
    ],
)
def test_dask_duplicated_keys(file, tree, branch, open_files):
    path = skhep_testdata.data_path(file)
    with uproot.open(path) as f:
        assert f[tree].keys(filter_name=branch) == [branch, branch]
        expected = f[tree][branch].array()

    lazy = uproot.dask(f"{path}:{tree}", filter_name=branch, open_files=open_files)
    assert lazy.fields == [branch]
    materialized = lazy[branch].compute(scheduler="synchronous")
    assert ak.array_equal(materialized, expected)


@pytest.mark.parametrize("library", ["ak", "np"])
def test_dask_duplicated_keys_all_branches(library):
    path = skhep_testdata.data_path("uproot-issue443.root") + ":muonDataTree"
    lazy = uproot.dask(path, library=library)
    assert lazy["hitEnd"].compute(scheduler="synchronous").tolist() == []
