# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")

MULTIPLE_TTREES = "TTree or RNTuple object paths must be specified in the 'files'"


@pytest.fixture(scope="module")
def files():
    # The glob only matches files that are already in the cache.
    for version in ("6.16.00", "6.18.00"):
        path = skhep_testdata.data_path(f"uproot-sample-{version}-uncompressed.root")
    return path.replace("6.18.00", "6.1[68].00")


def test_open():
    directory = uproot.open(skhep_testdata.data_path("uproot-issue63.root"))
    assert isinstance(directory, uproot.reading.ReadOnlyDirectory)
    assert directory.keys() == [
        "WtLoop_nominal;1",
        "WtLoop_Fake_nominal;1",
        "WtLoop_meta;1",
    ]

    tree = uproot.open(
        {skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal"}
    )
    assert isinstance(tree, uproot.behaviors.TTree.TTree)
    assert tree.name == "WtLoop_nominal"
    assert tree.num_entries == 9812

    with pytest.raises(ValueError, match="'path' must be a string"):
        uproot.open([skhep_testdata.data_path("uproot-issue63.root")])


@pytest.mark.parametrize("wrap", [str, lambda x: {x: "sample"}, lambda x: [x]])
def test_concatenate_globs(files, wrap):
    assert uproot.concatenate(wrap(files), "Ai8").Ai8[:3].tolist() == [
        [],
        [-15],
        [-15, -13],
    ]
    assert len(uproot.concatenate(wrap(files), "Ai8")) == 60


def test_concatenate_globs_in_dict_in_list(files):
    assert len(uproot.concatenate([{files: "sample"}], "Ai8")) == 60


def test_concatenate():
    with pytest.raises(ValueError, match=MULTIPLE_TTREES):
        uproot.concatenate(skhep_testdata.data_path("uproot-issue63.root"))

    assert (
        len(
            uproot.concatenate(
                {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
                allow_missing=True,
            )
        )
        == 0
    )


def test_iterate():
    with pytest.raises(ValueError, match=MULTIPLE_TTREES):
        list(uproot.iterate(skhep_testdata.data_path("uproot-issue63.root")))

    assert (
        len(
            list(
                uproot.iterate(
                    {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
                    allow_missing=True,
                )
            )
        )
        == 0
    )


@pytest.mark.parametrize(
    "wrap",
    [str, lambda x: {x: "sample"}, lambda x: [x], lambda x: [{x: "sample"}]],
)
def test_iterate_globs(files, wrap):
    batches = list(uproot.iterate(wrap(files), "Ai8"))
    assert [len(arrays) for arrays in batches] == [30, 30]
    assert [arrays.Ai8[1].tolist() for arrays in batches] == [[-15], [-15]]


pathlib = pytest.importorskip("pathlib")


def test_open_colon():
    tree = uproot.open(
        skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal"
    )
    assert isinstance(tree, uproot.behaviors.TTree.TTree)
    assert tree.name == "WtLoop_nominal"
    assert tree.num_entries == 9812

    with pytest.raises(FileNotFoundError, match="issue63.root:WtLoop_nominal"):
        uproot.open(
            pathlib.Path(
                skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal"
            )
        )

    with pytest.raises(FileNotFoundError, match="issue63.root:WtLoop_nominal"):
        uproot.open(
            {skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal": None}
        )
