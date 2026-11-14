# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import pytest

import uproot
import uproot.writing


def test_write_empty(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    tlist = uproot.writing.identify.to_TList([])

    with uproot.recreate(filename) as f:
        f["test"] = tlist

    with uproot.open(filename) as f2:
        assert f2["test"].classname == "TList"
        assert list(f2["test"]) == []


def test_write_single_key(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    tlist = uproot.writing.identify.to_TList(
        [uproot.writing.identify.to_TObjString("test string")]
    )

    with uproot.recreate(filename) as f:
        f["test"] = tlist

    with uproot.open(filename) as f2:
        assert [str(x) for x in f2["test"]] == ["test string"]


def test_write_nested(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    tlist_child = uproot.writing.identify.to_TList(
        [uproot.writing.identify.to_TObjString(s) for s in "this is a test".split()]
    )

    entries = [
        uproot.writing.identify.to_TObjString("this string goes in the front"),
        tlist_child,
        uproot.writing.identify.to_TObjString("test string"),
    ]

    tlist = uproot.writing.identify.to_TList(entries)

    with uproot.recreate(filename) as f:
        f["test"] = tlist

    with uproot.open(filename) as f2:
        parent_list = f2["test"]
        assert len(parent_list) == 3
        assert isinstance(parent_list[0], uproot.models.TObjString.Model_TObjString)
        assert str(parent_list[0]) == "this string goes in the front"
        assert str(parent_list[2]) == "test string"
        child_list = parent_list[1]
        assert isinstance(child_list, uproot.models.TList.Model_TList)
        assert len(child_list) == 4
        assert " ".join([str(s) for s in child_list]) == "this is a test"
