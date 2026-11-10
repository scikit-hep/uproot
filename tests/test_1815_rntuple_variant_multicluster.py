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


def test_rntuple_variant_multicluster_none(tmp_path):
    path = tmp_path / "var_none.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], None, "a"])}
        f["n"].extend({"x": ak.Array([None, [2], "b"])})
        f["n"].extend({"x": ak.Array([[3], None, "c"])})

    with uproot.open(path) as f:
        n = f["n"]
        expected = [[1], None, "a", None, [2], "b", [3], None, "c"]
        assert n.arrays().x.tolist() == expected
        assert n.arrays(entry_start=2).x.tolist() == expected[2:]
        assert n.arrays(entry_start=4, entry_stop=8).x.tolist() == expected[4:8]


def _switch_array(tags, index):
    import numpy as np

    array = np.zeros(len(tags), dtype=[("index", np.int64), ("tag", np.int32)])
    array["tag"] = tags
    array["index"] = index
    return array


def test_rntuple_variant_combine_cluster_arrays_unit(tmp_path):
    import numpy as np

    path = tmp_path / "var_unit.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], "a"])}
        f["n"].extend({"x": ak.Array([[2], "b"])})

    with uproot.open(path) as f:
        n = f["n"]
        switch = uproot.const.rntuple_col_type_to_num_dict["switch"]
        (col_idx,) = [cr.idx for cr in n.column_records if cr.type == switch]
        meta = n.get_field_metadata(col_idx)

        # Empty cluster in the middle (start == stop)
        array = _switch_array([1, 2, 1, 2], [0, 0, 0, 0])
        n.combine_cluster_arrays(array, [0, 2, 2], meta)
        assert array["index"].tolist() == [0, 0, 1, 1]

        # Tag 0 (invalid/None) entries are left untouched
        array = _switch_array([1, 0, 0, 1], [0, 0, 0, 0])
        n.combine_cluster_arrays(array, [0, 2], meta)
        assert array["index"].tolist() == [0, 0, 0, 1]

        # Tag 0 absent, tags not starting at 1
        array = _switch_array([2, 3, 3, 2], [0, 0, 0, 0])
        n.combine_cluster_arrays(array, [0, 2], meta)
        assert array["index"].tolist() == [0, 0, 1, 1]

        # Padding before the first cluster is left untouched
        array = _switch_array([0, 0, 1, 1, 1], [0, 0, 0, 1, 0])
        n.combine_cluster_arrays(array, [2, 4], meta)
        assert array["index"].tolist() == [0, 0, 0, 1, 2]

        # A single cluster is unchanged
        array = _switch_array([1, 2, 1], [0, 0, 1])
        n.combine_cluster_arrays(array, [0], meta)
        assert array["index"].tolist() == [0, 0, 1]

        # Many clusters
        n_clusters = 50
        array = _switch_array(
            np.tile([1, 2, 1, 2], n_clusters), np.tile([0, 0, 1, 1], n_clusters)
        )
        n.combine_cluster_arrays(array, list(range(0, 4 * n_clusters, 4)), meta)
        assert array["index"][array["tag"] == 1].tolist() == list(range(2 * n_clusters))
        assert array["index"][array["tag"] == 2].tolist() == list(range(2 * n_clusters))
