# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest

import uproot

ak = pytest.importorskip("awkward")


def test_rntuple_variant_multicluster_two_clusters(tmp_path):
    path = tmp_path / "var.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], "a"]), "y": [1, 2]}
        f["n"].extend({"x": ak.Array([[4], "b"]), "y": [3, 4]})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().tolist() == [
            {"x": [1], "y": 1},
            {"x": "a", "y": 2},
            {"x": [4], "y": 3},
            {"x": "b", "y": 4},
        ]
        assert n.arrays(entry_start=2).x.tolist() == [[4], "b"]
        assert n.arrays(entry_start=1, entry_stop=3).x.tolist() == ["a", [4]]


def test_rntuple_variant_multicluster_three_clusters_asymmetric(tmp_path):
    path = tmp_path / "var3.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], [2], "a"])}
        f["n"].extend({"x": ak.Array([[3], "b", "c"])})
        f["n"].extend({"x": ak.Array([[4], [5], "d"])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [
            [1],
            [2],
            "a",
            [3],
            "b",
            "c",
            [4],
            [5],
            "d",
        ]
        assert n.arrays(entry_start=3).x.tolist() == [[3], "b", "c", [4], [5], "d"]
        assert n.arrays(entry_start=6).x.tolist() == [[4], [5], "d"]
        assert n.arrays(entry_start=2, entry_stop=7).x.tolist() == [
            "a",
            [3],
            "b",
            "c",
            [4],
        ]


def test_rntuple_variant_multicluster_nested_vector(tmp_path):
    path = tmp_path / "var_vec.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[[1], "a"], ["b"]])}
        f["n"].extend({"x": ak.Array([[[2], "c"], [[3], [4]]])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [
            [[1], "a"],
            ["b"],
            [[2], "c"],
            [[3], [4]],
        ]
        assert n.arrays(entry_start=1).x.tolist() == [
            ["b"],
            [[2], "c"],
            [[3], [4]],
        ]
        assert n.arrays(entry_start=2).x.tolist() == [
            [[2], "c"],
            [[3], [4]],
        ]


def test_rntuple_variant_multicluster_virtual(tmp_path):
    path = tmp_path / "var_virtual.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[10], "x"]), "y": [100, 200]}
        f["n"].extend({"x": ak.Array([[20], "y"]), "y": [300, 400]})

    with uproot.open(path) as f:
        n = f["n"]
        arr = n.arrays(virtual=True)
        assert arr.tolist() == [
            {"x": [10], "y": 100},
            {"x": "x", "y": 200},
            {"x": [20], "y": 300},
            {"x": "y", "y": 400},
        ]


def test_rntuple_variant_multicluster_scalars(tmp_path):
    path = tmp_path / "var_scalars.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([1, "first"])}
        f["n"].extend({"x": ak.Array([2, "second"])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [1, "first", 2, "second"]
        assert n.arrays(entry_start=1, entry_stop=3).x.tolist() == ["first", 2]


def test_rntuple_variant_multicluster_disjoint_tags(tmp_path):
    import numpy as np

    c1 = ak.Array([1, 2]).layout
    c2 = ak.Array(["a", "b"]).layout
    u1 = ak.Array(
        ak.contents.UnionArray(
            tags=ak.index.Index8(np.array([0, 0], dtype=np.int8)),
            index=ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            contents=[c1, c2],
        )
    )
    u2 = ak.Array(
        ak.contents.UnionArray(
            tags=ak.index.Index8(np.array([1, 1], dtype=np.int8)),
            index=ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            contents=[c1, c2],
        )
    )

    path = tmp_path / "var_disjoint.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": u1}
        f["n"].extend({"x": u2})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [1, 2, "a", "b"]


def test_rntuple_variant_combine_cluster_arrays_unit(tmp_path):
    import numpy as np

    path = tmp_path / "var_unit.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], "a"])}
        f["n"].extend({"x": ak.Array([[2], "b"])})

    with uproot.open(path) as f:
        n = f["n"]
        meta = n.get_field_metadata(0)
        assert meta.dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]

        # Empty cluster in middle: start == stop
        arr_empty = np.zeros(4, dtype=[("index", np.int64), ("tag", np.int32)])
        arr_empty["tag"] = [1, 2, 1, 2]
        n.combine_cluster_arrays(arr_empty, [0, 2, 2], meta)
        assert list(arr_empty["index"]) == [0, 0, 1, 1]

        # Tag 0 sentinel preservation
        arr_tag0 = np.zeros(4, dtype=[("index", np.int64), ("tag", np.int32)])
        arr_tag0["tag"] = [1, 0, 0, 1]
        n.combine_cluster_arrays(arr_tag0, [0, 2], meta)
        assert list(arr_tag0["index"]) == [0, 0, 0, 1]

        # Multicluster running offset accumulator scaling
        n_clusters = 50
        cluster_len = 4
        total_len = n_clusters * cluster_len
        arr_many = np.zeros(total_len, dtype=[("index", np.int64), ("tag", np.int32)])
        arr_many["tag"] = np.tile([1, 2, 1, 2], n_clusters)
        arr_many["index"] = np.tile([0, 0, 1, 1], n_clusters)
        many_starts = [i * cluster_len for i in range(n_clusters)]
        n.combine_cluster_arrays(arr_many, many_starts, meta)
        assert list(arr_many["index"][arr_many["tag"] == 1]) == list(
            range(2 * n_clusters)
        )
        assert list(arr_many["index"][arr_many["tag"] == 2]) == list(
            range(2 * n_clusters)
        )
