# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
awkward = pytest.importorskip("awkward")


def test_awkward_as_numpy(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": awkward.types.from_datashape("int32", highlevel=False),
                "b2": awkward.types.from_datashape("2 * float64", highlevel=False),
                "b3": awkward.types.from_datashape("2 * 3 * float64", highlevel=False),
                "b4": awkward.Array([1.1, 2.2, 3.3]).type,
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2"].typename == "double[2]"
        assert fin["tree/b3"].typename == "double[2][3]"
        assert fin["tree/b4"].typename == "double"


def test_awkward_record(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": "int32",
                "b2": awkward.types.from_datashape(
                    '{"x": float64, "y": 3 * float64}', highlevel=False
                ),
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "double[3]"


def test_awkward_record_data(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = awkward.Array([{"x": 1.1, "y": 4}, {"x": 2.2, "y": 5}, {"x": 3.3, "y": 6}])
        fout.mktree("tree", {"b1": b1.dtype, "b2": b2.type})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6]


def test_awkward_record_dict_1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = {"x": np.array([1.1, 2.2, 3.3]), "y": np.array([4, 5, 6], np.int64)}
        fout.mktree(
            "tree", {"b1": b1.dtype, "b2": {"x": b2["x"].dtype, "y": b2["y"].dtype}}
        )
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6]


def test_awkward_record_dict_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = {"x": np.array([1.1, 2.2, 3.3]), "y": np.array([4, 5, 6], np.int64)}
        fout.mktree("tree", {"b1": b1, "b2": b2})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6, 4, 5, 6]


def test_awkward_record_recarray(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = np.array(
            [(1.1, 4), (2.2, 5), (3.3, 6)], [("x", np.float64), ("y", np.int64)]
        )
        fout.mktree("tree", {"b1": b1, "b2": b2})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6, 4, 5, 6]


def test_awkward_record_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")

    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = pandas.DataFrame({"x": [1.1, 2.2, 3.3], "y": [4, 5, 6]})
        fout.mktree("tree", {"b1": b1, "b2": b2})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6, 4, 5, 6]


def test_top_level(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    df1 = awkward.Array({"x": [1, 2, 3], "y": [1.1, 2.2, 3.3]})
    df2 = awkward.Array({"x": [4, 5, 6], "y": [4.4, 5.5, 6.6]})

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree("tree", df1)
        fout["tree"].extend(df2)

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert t1.GetBranch("x").GetName() == "x"
    assert t1.GetBranch("y").GetName() == "y"
    assert [np.asarray(x.x).tolist() for x in t1] == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x.y).tolist() for x in t1] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    with uproot.open(newfile) as fin:
        assert fin["tree/x"].name == "x"
        assert fin["tree/y"].name == "y"
        assert fin["tree/x"].typename.startswith("int")
        assert fin["tree/y"].typename == "double"
        assert fin["tree/x"].array().tolist() == [1, 2, 3, 4, 5, 6]
        assert fin["tree/y"].array().tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    f1.Close()


def test_awkward_jagged_metadata(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": "int64",
                "b2": awkward.types.from_datashape("var * float64", highlevel=False),
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int64_t"
        assert fin["tree/nb2"].typename == "int32_t"
        assert fin["tree/b2"].typename == "double[]"

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    b1 = t1.GetBranch("b1")
    assert b1.GetLeaf("b1").GetName() == "b1"
    assert not b1.GetLeaf("b1").GetLeafCount()

    nb2 = t1.GetBranch("nb2")
    assert nb2.GetLeaf("nb2").GetName() == "nb2"
    assert not nb2.GetLeaf("nb2").GetLeafCount()

    b2 = t1.GetBranch("b2")
    assert b2.GetLeaf("b2").GetName() == "b2"
    assert b2.GetLeaf("b2").GetLeafCount()
    assert b2.GetLeaf("b2").GetLeafCount().GetName() == "nb2"

    f1.Close()


def test_awkward_jagged_record_metadata(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": "int64",
                "b2": awkward.types.from_datashape(
                    'var * {"x": float64, "y": int8}', highlevel=False
                ),
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int64_t"
        assert fin["tree/nb2"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double[]"
        assert fin["tree/b2_y"].typename == "int8_t[]"

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    b1 = t1.GetBranch("b1")
    assert b1.GetLeaf("b1").GetName() == "b1"
    assert not b1.GetLeaf("b1").GetLeafCount()

    nb2 = t1.GetBranch("nb2")
    assert nb2.GetLeaf("nb2").GetName() == "nb2"
    assert not nb2.GetLeaf("nb2").GetLeafCount()

    b2_x = t1.GetBranch("b2_x")
    assert b2_x.GetLeaf("b2_x").GetName() == "b2_x"
    assert b2_x.GetLeaf("b2_x").GetLeafCount()
    assert b2_x.GetLeaf("b2_x").GetLeafCount().GetName() == "nb2"

    b2_y = t1.GetBranch("b2_y")
    assert b2_y.GetLeaf("b2_y").GetName() == "b2_y"
    assert b2_y.GetLeaf("b2_y").GetLeafCount()
    assert b2_y.GetLeaf("b2_y").GetLeafCount().GetName() == "nb2"

    f1.Close()


def test_awkward_jagged_data_1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3, 4, 5], np.int64)
        b2 = awkward.Array(
            [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
        )
        fout.mktree("tree", {"b1": b1.dtype, "b2": b2.type})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb2"].member("fLeaves")[0].member("fMaximum") == 4
        assert fin["tree/b2"].member("fEntryOffsetLen") == 4 * 5
        assert fin["tree/b1"].array().tolist() == [1, 2, 3, 4, 5]
        assert fin["tree/nb2"].array().tolist() == [3, 0, 2, 1, 4]
        assert fin["tree/b2"].array().tolist() == [
            [0.0, 1.1, 2.2],
            [],
            [3.3, 4.4],
            [5.5],
            [6.6, 7.7, 8.8, 9.9],
        ]

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")
    assert [x.b1 for x in t1] == [1, 2, 3, 4, 5]
    assert [x.nb2 for x in t1] == [3, 0, 2, 1, 4]
    assert [list(x.b2) for x in t1] == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    f1.Close()


def test_awkward_jagged_data_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3, 4, 5], np.int64)
        b2 = awkward.Array(
            [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
        )
        fout.mktree("tree", {"b1": b1, "b2": b2})
        fout["tree"].extend({"b1": b1[:3], "b2": b2[:3]})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb2"].member("fLeaves")[0].member("fMaximum") == 4
        assert fin["tree/b2"].member("fEntryOffsetLen") == 4 * 3
        assert fin["tree/b1"].array().tolist() == [1, 2, 3, 4, 5, 1, 2, 3]
        assert fin["tree/nb2"].array().tolist() == [3, 0, 2, 1, 4, 3, 0, 2]
        assert fin["tree/b2"].array().tolist() == [
            [0.0, 1.1, 2.2],
            [],
            [3.3, 4.4],
            [5.5],
            [6.6, 7.7, 8.8, 9.9],
            [0.0, 1.1, 2.2],
            [],
            [3.3, 4.4],
        ]

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")
    assert [x.b1 for x in t1] == [1, 2, 3, 4, 5, 1, 2, 3]
    assert [x.nb2 for x in t1] == [3, 0, 2, 1, 4, 3, 0, 2]
    assert [list(x.b2) for x in t1] == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
    ]
    f1.Close()


def test_awkward_jagged_data_3(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        big = awkward.Array(
            [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]] * 300
        )
        fout.mktree("tree", {"big": big})
        # more than 1000 entries, a special number for fNevBufSize and fEntryOffsetLen

    with uproot.open(newfile) as fin:
        assert fin["tree/nbig"].member("fLeaves")[0].member("fMaximum") == 4
        assert fin["tree/big"].member("fEntryOffsetLen") == 4 * 1500
        assert fin["tree/nbig"].array().tolist() == [3, 0, 2, 1, 4] * 300
        assert (
            fin["tree/big"].array().tolist()
            == [
                [0.0, 1.1, 2.2],
                [],
                [3.3, 4.4],
                [5.5],
                [6.6, 7.7, 8.8, 9.9],
            ]
            * 300
        )

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")
    assert [x.nbig for x in t1] == [3, 0, 2, 1, 4] * 300
    assert [list(x.big) for x in t1] == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ] * 300
    f1.Close()


def test_awkward_jagged_record_1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        array = awkward.Array(
            [
                [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                [],
                [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
            ]
        )
        fout.mktree("tree", {"array": array})
        fout["tree"].extend({"array": array})

    with uproot.open(newfile) as fin:
        assert fin["tree/narray"].array().tolist() == [3, 0, 2] * 2
        assert fin["tree/array_x"].array().tolist() == [[1, 2, 3], [], [4, 5]] * 2
        assert (
            fin["tree/array_y"].array().tolist()
            == [[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 2
        )

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")
    assert [x.narray for x in t1] == [3, 0, 2] * 2
    assert [list(x.array_x) for x in t1] == [[1, 2, 3], [], [4, 5]] * 2
    assert [list(x.array_y) for x in t1] == [[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 2
    f1.Close()


def test_awkward_jagged_record_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            awkward.Array(
                [
                    [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                    [],
                    [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
                ]
            ),
        )
        fout["tree"].extend(
            awkward.Array(
                [
                    [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                    [],
                    [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
                ]
            )
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/n"].array().tolist() == [3, 0, 2] * 2
        assert fin["tree/x"].array().tolist() == [[1, 2, 3], [], [4, 5]] * 2
        assert fin["tree/y"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 2

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")
    assert [x.n for x in t1] == [3, 0, 2] * 2
    assert [list(x.x) for x in t1] == [[1, 2, 3], [], [4, 5]] * 2
    assert [list(x.y) for x in t1] == [[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 2
    f1.Close()
