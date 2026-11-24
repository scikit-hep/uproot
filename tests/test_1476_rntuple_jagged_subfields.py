# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import pytest

import uproot

ak = pytest.importorskip("awkward")

data = ak.Array(
    {
        "field1": [[(0, 1), (2, 3)], [], [(4, 5)]],
        "field2": [[{"x": 0, "y": 1}, {"x": 2, "y": 3}], [], [{"x": 4, "y": 5}]],
        "field3": [
            {"x": [1, 2, 3], "y": [(1, 2), (2, 3), (3, 4)]},
            {"x": [], "y": [(7, 8)]},
            {"x": [9, 10], "y": [(11, 12), (13, 14)]},
        ],
        "field4": [(1, [(1, 2), (3, 4)]), (2, []), (3, [(5, 6)])],
        "field5": [
            {
                "x": [
                    {"up": [(0, 2), (1, 2)], "down": [[1, 2, 3], []]},
                    {"up": [], "down": [[4]]},
                ],
                "y": [
                    (
                        {"left": [[0, "hi", 2.3]], "right": 6},
                        {"left": [[], [""]], "right": 8.0},
                    )
                ],
            },
            {
                "x": [
                    {"up": [(10, 2), (12, 2)], "down": []},
                    {"up": [(1, 2)], "down": [[], [4, 2, 3]]},
                ],
                "y": [
                    (
                        {"left": [[23, 4.1, "hello"]], "right": 14},
                        {"left": [], "right": 16.0},
                    )
                ],
            },
            {
                "x": [
                    {"up": [], "down": [[4, 5], [1, 2, 3], []]},
                    {"up": [(0, 2), (1, 2)], "down": []},
                ],
                "y": [
                    (
                        {"left": [[]], "right": 14},
                        {"left": [[2, 3], ["bye", ""]], "right": 16.0},
                    )
                ],
            },
        ],
        "field6": [(1, 2, 3, 4), (2, 5, 6, 7), ("hello", 2.3, 8, "")],
    }
)


def test_jagged_subfields(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)

    obj = uproot.open(filepath)["ntuple"]

    assert ak.array_equal(obj["field1"].array(), data["field1"])
    assert ak.array_equal(obj["field1.0"].array(), data["field1"]["0"])
    assert ak.array_equal(obj["field1.1"].array(), data["field1"]["1"])

    assert ak.array_equal(obj["field2"].array(), data["field2"])
    assert ak.array_equal(obj["field2.x"].array(), data["field2"]["x"])
    assert ak.array_equal(obj["field2.y"].array(), data["field2"]["y"])

    assert ak.array_equal(obj["field3"].array(), data["field3"])
    assert ak.array_equal(obj["field3.x"].array(), data["field3"]["x"])
    assert ak.array_equal(obj["field3.y"].array(), data["field3"]["y"])
    assert ak.array_equal(obj["field3.y.0"].array(), data["field3"]["y"]["0"])
    assert ak.array_equal(obj["field3.y.1"].array(), data["field3"]["y"]["1"])

    assert ak.array_equal(obj["field4"].array(), data["field4"])
    assert ak.array_equal(obj["field4.0"].array(), data["field4"]["0"])
    assert ak.array_equal(obj["field4.1"].array(), data["field4"]["1"])
    assert ak.array_equal(obj["field4.1.0"].array(), data["field4"]["1"]["0"])
    assert ak.array_equal(obj["field4.1.1"].array(), data["field4"]["1"]["1"])

    assert obj["field5"].array().tolist() == data["field5"].tolist()
    assert obj["field5.x"].array().tolist() == data["field5"]["x"].tolist()
    assert obj["field5.x.up"].array().tolist() == data["field5"]["x"]["up"].tolist()
    assert obj["field5.x.down"].array().tolist() == data["field5"]["x"]["down"].tolist()
    assert obj["field5.y"].array().tolist() == data["field5"]["y"].tolist()
    assert obj["field5.y.0"].array().tolist() == data["field5"]["y"]["0"].tolist()
    assert (
        obj["field5.y.0.left"].array().tolist()
        == data["field5"]["y"]["0"]["left"].tolist()
    )
    assert (
        obj["field5.y.0.right"].array().tolist()
        == data["field5"]["y"]["0"]["right"].tolist()
    )
    assert obj["field5.y.1"].array().tolist() == data["field5"]["y"]["1"].tolist()
    assert (
        obj["field5.y.1.left"].array().tolist()
        == data["field5"]["y"]["1"]["left"].tolist()
    )
    assert (
        obj["field5.y.1.right"].array().tolist()
        == data["field5"]["y"]["1"]["right"].tolist()
    )

    assert obj["field6"].array().tolist() == data["field6"].tolist()
    assert obj["field6.0"].array().tolist() == data["field6"]["0"].tolist()
    assert obj["field6.1"].array().tolist() == data["field6"]["1"].tolist()
    assert obj["field6.2"].array().tolist() == data["field6"]["2"].tolist()
    assert obj["field6.3"].array().tolist() == data["field6"]["3"].tolist()
