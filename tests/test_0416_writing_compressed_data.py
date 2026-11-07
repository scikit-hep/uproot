# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


def use_zlib_library(monkeypatch, library):
    if library != "zlib":
        pytest.importorskip(library)
    monkeypatch.setattr(uproot.ZLIB, "library", library)


@pytest.fixture(params=["zlib", "isal", "deflate"])
def zlib_library(request, monkeypatch):
    use_zlib_library(monkeypatch, request.param)


@pytest.fixture(params=["zlib", "isal", "deflate", "LZMA", "LZ4", "ZSTD"])
def algorithm(request, monkeypatch):
    if request.param == "LZMA":
        pytest.importorskip("lzma")
    elif request.param in ("LZ4", "ZSTD"):
        pytest.importorskip("cramjam")
    else:
        use_zlib_library(monkeypatch, request.param)
        return uproot.ZLIB
    return getattr(uproot, request.param)


@pytest.mark.parametrize(
    "library, module, filename",
    [
        ("zlib", "zlib", "uproot-Zmumu-zlib.root"),
        ("isal", "isal", "uproot-Zmumu-zlib.root"),
        ("deflate", "deflate", "uproot-Zmumu-zlib.root"),
        (None, "lzma", "uproot-Zmumu-lzma.root"),
        (None, "cramjam", "uproot-Zmumu-lz4.root"),
        (None, "cramjam", "uproot-Zmumu-zstd.root"),
    ],
)
def test_read_compressed_file(monkeypatch, library, module, filename):
    pytest.importorskip(module)
    if library is not None:
        use_zlib_library(monkeypatch, library)

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path(filename))["events"] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def assert_compressed_with(branch, algorithm):
    # TBranch.fCompress is always written as 0: only the blocks tell what was used
    assert branch.compressed_bytes < branch.uncompressed_bytes
    ((name, compressed, uncompressed),) = branch.basket(0).block_compression_info
    assert name == algorithm.__name__
    assert compressed < uncompressed


def test_histogram(tmp_path, algorithm):
    newfile = os.path.join(tmp_path, "newfile.root")

    # compressed blocks have at most 2**24 - 1 bytes, so 2**21 float64 are two blocks;
    # splitting doesn't depend on the algorithm, so that size is for the fastest ones
    SIZE = 2**21 if algorithm in (uproot.LZ4, uproot.ZSTD) else 2**15
    entries = np.random.default_rng(416).integers(0, 10, SIZE)
    edges = np.linspace(0, 1, SIZE + 1)

    with uproot.recreate(newfile, compression=algorithm(1)) as fout:
        fout["out"] = entries, edges

    with uproot.open(newfile) as fin:
        content, read_edges = fin["out"].to_numpy()
        assert np.array_equal(content, entries)
        assert np.array_equal(read_edges, edges)
    assert os.path.getsize(newfile) < 8 * SIZE

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == entries[-1]
    assert h3.Integral() == entries.sum()
    f3.Close()


def test_flattree(tmp_path, algorithm):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=algorithm(5)) as fout:
        fout.mktree("tree", {"branch1": branch1, "branch2": branch2})
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2
        assert_compressed_with(fin["tree/branch1"], algorithm)

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist() * 2
    assert [x.branch2 for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_jaggedtree(tmp_path, algorithm):
    ak = pytest.importorskip("awkward")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = ak.Array([[1, 2, 3], [], [4, 5]] * 10)
    branch2 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 10)

    with uproot.recreate(newfile, compression=algorithm(5)) as fout:
        fout.mktree("tree", {"branch1": branch1, "branch2": branch2})
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array().tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array().tolist() == branch2.tolist() * 2
        assert_compressed_with(fin["tree/branch1"], algorithm)

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [list(x.branch1) for x in t3] == branch1.tolist() * 2
    assert [list(x.branch2) for x in t3] == branch2.tolist() * 2
    f3.Close()


def no_change(fout):
    pass


def file_to_None(fout):
    fout.compression = None


def each_branch(fout):
    fout["tree"]["branch1"].compression = uproot.ZLIB(5)
    fout["tree"]["branch2"].compression = None


def tree_with_dict(fout):
    fout["tree"].compression = {"branch1": uproot.ZLIB(5), "branch2": None}


def tree_with_dict_then_one_algorithm(fout):
    fout["tree"].compression = {"branch1": uproot.ZLIB(5), "branch2": None}
    fout["tree"].compression = uproot.ZLIB(5)


@pytest.mark.parametrize(
    "options, before_mktree, after_mktree, branch1_compressed, branch2_compressed",
    [
        ({}, no_change, each_branch, True, False),
        ({}, no_change, tree_with_dict, True, False),
        ({}, no_change, tree_with_dict_then_one_algorithm, True, True),
        ({"compression": uproot.ZLIB(5)}, no_change, no_change, True, True),
        ({"compression": uproot.ZLIB(5)}, file_to_None, no_change, False, False),
    ],
    ids=["each-branch", "dict", "dict-then-one", "file", "file-None-after-opening"],
)
def test_multicompression(
    tmp_path,
    zlib_library,
    options,
    before_mktree,
    after_mktree,
    branch1_compressed,
    branch2_compressed,
):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, **options) as fout:
        before_mktree(fout)
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        after_mktree(fout)
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        for name, array, compressed in [
            ("branch1", branch1, branch1_compressed),
            ("branch2", branch2, branch2_compressed),
        ]:
            branch = fin["tree"][name]
            assert branch.array(library="np").tolist() == array.tolist()
            assert branch.uncompressed_bytes == 874
            if compressed:
                assert_compressed_with(branch, uproot.ZLIB)
            else:
                assert branch.compressed_bytes == 874
                assert branch.basket(0).block_compression_info is None

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()
