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


def test_extend_with_incompatible_type_still_fails(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = {"jagged": ak.Array([[1, 2], [3]])}
        with pytest.raises(ValueError, match="not compatible"):
            file["ntuple"].extend({"jagged": ak.Array([[1.5]])})


def test_create_with_only_empty_lists(tmp_path):
    # with no other information, "unknown" is written as float64, like in TTrees
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        file["ntuple"] = {"jagged": ak.Array([[], []])}
        file["ntuple"].extend({"jagged": ak.Array([[2.5]])})

    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays()
    assert arrays.jagged.tolist() == [[], [], [2.5]]
    assert str(arrays.type) == "3 * {jagged: var * float64}"
