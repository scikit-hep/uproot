# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot

awkward = pytest.importorskip("awkward")


def test_awkward_strings():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["string"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["tstring"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_vector_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_int32"].array(library="ak")) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_string"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_string_forth():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        temp_branch = tree["vector_string"]
        temp_branch.interpretation._forth = True
        assert awkward.to_list(temp_branch.array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_array_tref_array_forth():
    awk_data = None

    with uproot.open(skhep_testdata.data_path("uproot-delphes-pr442.root"))[
        "Delphes"
    ] as tree:
        temp_branch = tree["GenJet04/GenJet04.Constituents"]
        temp_branch.interpretation._forth = True
        awk_data = temp_branch.array(library="ak")

    assert awk_data[0][0]["refs"][-1] == 2579
    assert awk_data[4][1]["refs"][-1] == 3391
    assert awk_data[6][2]["refs"][-1] == 676


def test_awkward_array_tvector2_array_forth():
    awk_data = None

    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events/MET"
    ] as tree:
        interp = uproot.interpretation.identify.interpretation_of(tree, {}, False)
        interp._forth = True
        awk_data = tree.array(interp, library="ak")
    assert awk_data[0]["fX"] == pytest.approx(5.912771224975586)
    assert awk_data[4]["fY"] == pytest.approx(-1.3100523948669434)
    assert awk_data[1200]["fX"] == pytest.approx(1.9457910060882568)


def test_awkward_vector_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_tstring"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_vector_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_int32"].array(library="ak")) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_vector_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_string"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_vector_vector_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_tstring"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_set_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["set_int32"].array(library="ak")) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_vector_set_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_set_int32"].array(library="ak")) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_vector_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_set_string"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "three", "two"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
                ["five", "four", "one", "three", "two"],
            ],
        ]


def test_awkward_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["set_string"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]


def test_awkward_map_int32_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_map_int32_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_vector_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_map_int32_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_int32_vector_string"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_map_int32_set_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_set_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_map_int32_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_set_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "three", "two"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
                ["five", "four", "one", "three", "two"],
            ],
        ]


def test_awkward_map_string_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [1],
            [1, 2],
            [1, 3, 2],
            [4, 1, 3, 2],
            [5, 4, 1, 3, 2],
        ]


def test_awkward_map_string_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_string_vector_int16"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
        ]


def test_awkward_map_string_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_string_vector_string"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two", "three"], ["one", "two"]],
            [
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
            [
                ["one", "two", "three", "four", "five"],
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
        ]


def test_awkward_map_string_set_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_set_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
        ]


def test_awkward_map_string_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_set_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "three", "two"], ["one", "two"]],
            [
                ["four", "one", "three", "two"],
                ["one"],
                ["one", "three", "two"],
                ["one", "two"],
            ],
            [
                ["five", "four", "one", "three", "two"],
                ["four", "one", "three", "two"],
                ["one"],
                ["one", "three", "two"],
                ["one", "two"],
            ],
        ]


def test_awkward_map_int32_vector_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_int32_vector_vector_int16"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[[1]]],
            [[[1]], [[1], [1, 2]]],
            [[[1]], [[1], [1, 2]], [[1], [1, 2], [1, 2, 3]]],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            ],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            ],
        ]


def test_awkward_map_string_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]


def test_awkward_map_string_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_tstring"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]
