# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import pytest

import uproot
import uproot.writing

# THashList should be serialized the same as TList


def test_write_compare_tlist(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    entries = [
        uproot.writing.identify.to_TObjString(s)
        for s in "this is a test string".split()
    ]

    tlist = uproot.writing.identify.to_TList(entries)
    thashlist = uproot.writing.identify.to_THashList(entries)

    with uproot.recreate(filename) as f1:
        f1["tlist"] = tlist
        f1["thashlist"] = thashlist

    with uproot.open(filename) as f2:
        tlist_out = []
        uproot.serialization._serialize_object_any(tlist_out, f2["tlist"], "test")

        thashlist_out = []
        uproot.serialization._serialize_object_any(
            thashlist_out, f2["thashlist"], "test"
        )

        assert (
            tlist_out[0] != thashlist_out[0]
        )  # this section contains the class name which is different
        assert (
            tlist_out[1:] == thashlist_out[1:]
        )  # the remaining bytes should be the same
