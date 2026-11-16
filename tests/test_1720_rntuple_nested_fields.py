# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import uproot

ak = pytest.importorskip("awkward")


def _nested_rntuple(tmp_path):
    arr = ak.Array(
        [
            {
                "jets": [{"pt": float(j), "eta": 0.1 * j} for j in range(k % 3)],
                "rec": {"x": k, "y": 2 * k, "lst": [k] * (k % 2)},
                "z": 3 * k,
            }
            for k in range(6)
        ]
    )
    path = os.path.join(tmp_path, "nested.root")
    with uproot.recreate(path) as f:
        f["nt"] = arr
    return path + ":nt", arr


def test_a_dotted_key_under_a_collection_selects_the_leaf(tmp_path):
    where, arr = _nested_rntuple(tmp_path)
    nt = uproot.open(where)
    assert nt.keys(filter_name="jets.pt") == ["jets.pt"]
    assert nt.keys(filter_name="rec.lst") == ["rec.lst"]
    assert str(nt.arrays(["jets.pt"]).type) == "6 * {jets: var * {pt: float64}}"
    assert ak.array_equal(nt.arrays(["jets.pt"]).jets.pt, arr.jets.pt)
    assert (
        str(nt.to_akform(filter_name=["jets.pt"])[0].type)
        == "{jets: var * {pt: float64}}"
    )


def test_dask_reads_nested_rntuple_fields(tmp_path):
    dak = pytest.importorskip("dask_awkward")
    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.dask(where)
    assert ak.array_equal(ev.rec.x.compute(), arr.rec.x)
    assert ak.array_equal(ev.jets.pt.compute(), arr.jets.pt)
    assert ak.array_equal(ev.rec.lst.compute(), arr.rec.lst)
    assert ak.array_equal(ev.compute(), arr)
    assert set(next(iter(dak.necessary_columns(ev.rec.x).values()))) == {"rec.x"}
    assert ak.array_equal(
        uproot.dask(where, steps_per_file=3).rec.y.compute(), arr.rec.y
    )
    only_x = uproot.dask(where, filter_name="rec.x").compute()
    assert only_x.fields == ["rec"] and only_x.rec.fields == ["x"]
    assert ak.array_equal(only_x.rec.x, arr.rec.x)


def test_leaf_paths_are_spelled_like_rntuple_keys():
    from uproot._dask import _rntuple_leaf_paths

    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["jets", "v", "n"],
            "contents": [
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "fields": ["pt", "tag"],
                        "contents": [
                            {"class": "NumpyArray", "primitive": "float64"},
                            {
                                "class": "UnionArray",
                                "tags": "i8",
                                "index": "i64",
                                "contents": [
                                    {"class": "NumpyArray", "primitive": "int64"},
                                    {
                                        "class": "RecordArray",
                                        "fields": ["a"],
                                        "contents": [
                                            {"class": "NumpyArray", "primitive": "bool"}
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
                {
                    "class": "UnmaskedArray",
                    "content": {"class": "NumpyArray", "primitive": "int32"},
                },
                {"class": "NumpyArray", "primitive": "int64"},
            ],
        }
    )
    assert _rntuple_leaf_paths(form.content("jets"), "jets") == [
        "jets.pt",
        "jets.tag",
        "jets.tag.a",
    ]
    assert _rntuple_leaf_paths(form.content("v"), "v") == ["v"]
    assert _rntuple_leaf_paths(form.content("n"), "n") == ["n"]
