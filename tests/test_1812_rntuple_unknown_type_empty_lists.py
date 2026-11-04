# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest

import uproot

ak = pytest.importorskip("awkward")


def test_extend_with_only_empty_lists(tmp_path):
    # a chunk whose lists are all empty has Awkward type "var * unknown"
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = {
            "jagged": ak.Array([[1, 2], [3]]),
            "option": ak.Array([[1.5], None]),
            "record": ak.Array([{"a": [1]}, {"a": []}]),
        }
        file["ntuple"].extend(
            {
                "jagged": ak.Array([[], []]),
                "option": ak.Array([[], None]),
                "record": ak.Array([{"a": []}, {"a": []}]),
            }
        )

    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays()
    assert arrays.jagged.tolist() == [[1, 2], [3], [], []]
    assert arrays.option.tolist() == [[1.5], None, [], None]
    assert arrays.record.a.tolist() == [[1], [], [], []]
    assert str(arrays.type) == (
        "4 * {jagged: var * int64, option: option[var * float64], "
        "record: {a: var * int64}}"
    )


def test_extend_tuple_rntuple_with_only_empty_lists(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = ak.Array([([1, 2], [1.5])])
        file["ntuple"].extend({"0": ak.Array([[]]), "1": ak.Array([[]])})
        file["ntuple"].extend(ak.Array([([], [])]))

    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays()
    assert arrays.tolist() == [([1, 2], [1.5]), ([], []), ([], [])]


def test_extend_with_zero_length_unknown_chunk(tmp_path):
    # like a zero-length chunk with a known type, this adds an empty cluster
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = {"jagged": ak.Array([[1, 2], [3]])}
        file["ntuple"].extend(ak.Array([]))
        file["ntuple"].extend({"jagged": ak.Array([[4]])})

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert [c.num_entries for c in ntuple.cluster_summaries] == [2, 0, 1]
        assert ntuple.arrays().jagged.tolist() == [[1, 2], [3], [4]]


def test_extend_with_incompatible_type_still_fails(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = {"jagged": ak.Array([[1, 2], [3]])}
        with pytest.raises(ValueError, match="not compatible"):
            file["ntuple"].extend({"jagged": ak.Array([[1.5]])})
        # an "unknown" field elsewhere in the chunk must not let 1.5 be cast to 1
        file["ntuple"].extend({"jagged": ak.Array([[4]])})
    path2 = tmp_path / "test2.root"
    with uproot.recreate(path2) as file:
        file["ntuple"] = {"a": ak.Array([[1]]), "b": ak.Array([[2]])}
        with pytest.raises(ValueError, match="not compatible"):
            file["ntuple"].extend({"a": ak.Array([[]]), "b": ak.Array([[1.5]])})


def test_create_with_only_empty_lists_raises(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        with pytest.raises(TypeError, match="mkrntuple"):
            file["ntuple"] = {"jagged": ak.Array([[], []])}

        # the suggested workaround: give the type explicitly, then extend
        file.mkrntuple("explicit", {"jagged": "var * float64"})
        file["explicit"].extend({"jagged": ak.Array([[], []])})
        file["explicit"].extend({"jagged": ak.Array([[2.5]])})

    with uproot.open(path) as file:
        arrays = file["explicit"].arrays()
    assert arrays.jagged.tolist() == [[], [], [2.5]]
    assert str(arrays.type) == "3 * {jagged: var * float64}"
