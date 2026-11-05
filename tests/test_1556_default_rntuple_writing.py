# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest

import uproot

ak = pytest.importorskip("awkward")
pd = pytest.importorskip("pandas")
import numpy as np


def test_write_type_spec(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple(
            "ntuple", {"a": "int32", "b": "float64", "c": np.int64}
        )
        rntuple.extend(
            {
                "a": np.array([1, 2, 3], dtype=np.int32),
                "b": np.array([1.1, 2.2, 3.3], dtype=np.float64),
                "c": np.array([1, 2, 3], dtype=np.int64),
            }
        )

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3]
        assert rntuple["c"].array().tolist() == [1, 2, 3]


def test_write_complex_type_spec(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple(
            "ntuple",
            {
                "a": "var * int64",
                "b": "var * union[int64, bool]",
                "c": "var * ?int64",
                "d": "{one: int64, two: float64}",
            },
        )
        rntuple.extend(
            {
                "a": ak.Array([[1, 2, 3], [2, 3]]),
                "b": ak.Array([[1, True, 3], [2, False]]),
                "c": ak.Array([[1, None, 3], [2, None]]),
                "d": ak.Array([{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}]),
            }
        )

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [[1, 2, 3], [2, 3]]
        assert rntuple["b"].array().tolist() == [[1, True, 3], [2, False]]
        assert rntuple["c"].array().tolist() == [[1, None, 3], [2, None]]
        assert rntuple["d"].array().tolist() == [
            {"one": 1, "two": 1.1},
            {"one": 2, "two": 2.2},
        ]


def test_write_form(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = ak.Array({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", data.layout.form)
        rntuple.extend(data)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple.arrays().tolist() == data.tolist()

    sliced_data = data[data.a > 1]

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", sliced_data.layout.form)
        rntuple.extend(sliced_data)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple.arrays().tolist() == sliced_data.tolist()


@pytest.mark.parametrize(
    "extension",
    [
        {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]},
        {"b": [1.1, 2.2, 3.3], "a": [1, 2, 3]},
    ],
    ids=["same_order", "mixed_order"],
)
def test_write_dict(tmp_path, extension):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        rntuple.extend(extension)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]


def test_write_awkward(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = ak.Array({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", data)
        rntuple.extend(data)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple.arrays()[:3].tolist() == data.tolist()
        assert rntuple.arrays()[3:].tolist() == data.tolist()

    sliced_data = data[data.a > 1]

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", sliced_data)
        rntuple.extend(sliced_data)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple.arrays()[:2].tolist() == sliced_data.tolist()
        assert rntuple.arrays()[2:].tolist() == sliced_data.tolist()


def test_write_pandas(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 32, 18, 47],
        "City": ["New York", "Los Angeles", "Chicago", "Houston"],
    }
    df = pd.DataFrame(data)

    with uproot.recreate(filepath) as file:
        r = file.mkrntuple("ntuple", df)
        r.extend(df)

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["Name"].array().tolist() == [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Alice",
            "Bob",
            "Charlie",
            "David",
        ]
        assert rntuple["Age"].array().tolist() == [25, 32, 18, 47, 25, 32, 18, 47]
        assert rntuple["City"].array().tolist() == [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
        ]


def test_write_with_setitem(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    ak_data = ak.Array({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})

    pandas_data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 32, 18, 47],
        "City": ["New York", "Los Angeles", "Chicago", "Houston"],
    }
    pandas_df = pd.DataFrame(pandas_data)

    with uproot.recreate(filepath) as file:
        file["ntuple1"] = {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}
        file["ntuple2"] = ak_data.layout.form
        file["ntuple3"] = ak_data
        file["ntuple4"] = pandas_df

    with uproot.open(filepath) as file:
        assert file["ntuple1"].arrays()["a"].tolist() == [1, 2, 3]
        assert file["ntuple2"].arrays()["a"].tolist() == []
        assert file["ntuple3"].arrays()["a"].tolist() == [1, 2, 3]
        assert file["ntuple4"].arrays()["Age"].tolist() == [25, 32, 18, 47]


def test_invalid_inputs(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = ak.Array({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
    with uproot.recreate(filepath) as file:
        with pytest.raises(ValueError):
            file["ntuple1"] = {"a": "int32", "b": "float64", "c": np.int64}
        with pytest.raises(TypeError, match="array must be a RecordArray"):
            file["ntuple2"] = data.layout.contents[0]
        with pytest.raises(TypeError, match="form must be a RecordForm"):
            file["ntuple3"] = data.layout.contents[0].form
