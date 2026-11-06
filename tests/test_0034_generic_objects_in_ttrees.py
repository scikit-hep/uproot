# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.numerical import AsDtype


def test_histograms_outside_of_ttrees():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        contents = numpy.asarray(f["hpx"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 2417.0)

        contents = numpy.asarray(f["hpxpy"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 497.0)

        contents = numpy.asarray(f["hprof"].bases[-1].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 3054.7299575805664)

        numpy.asarray(f["ntuple"])


def test_gohep_nosplit_file():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        result = branch.array(library="np", entry_start=5, entry_stop=6)[0]
        assert result.member("Beg") == "beg-005"
        assert result.member("I16") == 5
        assert result.member("I32") == 5
        assert result.member("I64") == 5
        assert result.member("U16") == 5
        assert result.member("U32") == 5
        assert result.member("U64") == 5
        assert result.member("F32") == 5.0
        assert result.member("F64") == 5.0
        assert result.member("Str") == "evt-005"
        assert result.member("P3").member("Px") == 4
        assert result.member("P3").member("Py") == 5.0
        assert result.member("P3").member("Pz") == 4
        assert result.member("ArrayI16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("StdStr") == "std-005"
        assert result.member("SliceI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecF32").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecF64").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecStr").tolist() == [
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
        ]
        assert result.member("End") == "end-005"


def test_TLorentzVectors_show():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        tree.show()


def test_TVector2():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="np", entry_stop=1)[0]
        assert (result.member("fX"), result.member("fY")) == (
            5.912771224975586,
            2.5636332035064697,
        )


def test_vector_TLorentzVector():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="np", entry_stop=1)[0]
        assert len(result) == 2
        assert (
            result[0].member("fE"),
            result[0].member("fP").member("fX"),
            result[0].member("fP").member("fY"),
            result[0].member("fP").member("fZ"),
        ) == (
            54.77949905395508,
            -52.89945602416992,
            -11.654671669006348,
            -8.16079330444336,
        )


def test_strided():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        assert tree.file.class_named("TLorentzVector", "max").strided_interpretation(
            tree.file
        ).from_dtype == numpy.dtype(
            [
                ("@instance_version", numpy.dtype(">u2")),
                ("@num_bytes", numpy.dtype(">u4")),
                ("@fUniqueID", numpy.dtype(">u4")),
                ("@fBits", numpy.dtype(">u4")),
                ("@pidf", numpy.dtype(">u2")),
                ("fP/@instance_version", numpy.dtype(">u2")),
                ("fP/@num_bytes", numpy.dtype(">u4")),
                ("fP/@fUniqueID", numpy.dtype(">u4")),
                ("fP/@fBits", numpy.dtype(">u4")),
                ("fP/@pidf", numpy.dtype(">u2")),
                ("fP/fX", ">f8"),
                ("fP/fY", ">f8"),
                ("fP/fZ", ">f8"),
                ("fE", ">f8"),
            ]
        )


def test_read_strided_TVector2():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        interp = tree.file.class_named("TVector2", "max").strided_interpretation(
            tree.file
        )

        assert tree["MET"].interpretation == interp

        result = tree["MET"].array(entry_stop=1, library="np")[0]
        assert (result.member("fX"), result.member("fY")) == (
            5.912771224975586,
            2.5636332035064697,
        )


def test_read_strided_TLorentzVector():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        interp = tree.file.class_named("TLorentzVector", "max").strided_interpretation(
            tree.file
        )
        interp = uproot.interpretation.jagged.AsJagged(interp, header_bytes=10)

        assert tree["muonp4"].interpretation == interp

        result = tree["muonp4"].array(library="np", entry_stop=1)[0]
        assert len(result) == 2
        assert (
            result[0].member("fE"),
            result[0].member("fP").member("fX"),
            result[0].member("fP").member("fY"),
            result[0].member("fP").member("fZ"),
        ) == (
            54.77949905395508,
            -52.89945602416992,
            -11.654671669006348,
            -8.16079330444336,
        )


def test_strided_awkward():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="ak")

        assert str(awkward.type(result)) == "2421 * TVector2[fX: float64, fY: float64]"

        assert awkward.to_list(result["fX"][:10]) == [
            5.912771224975586,
            24.76520347595215,
            -25.78508758544922,
            8.619895935058594,
            5.393138885498047,
            -3.7594752311706543,
            23.962148666381836,
            -57.533348083496094,
            42.416194915771484,
            -1.9144694805145264,
        ]
        assert awkward.to_list(result["fY"][:10]) == [
            2.5636332035064697,
            -16.349109649658203,
            16.237131118774414,
            -22.78654670715332,
            -1.3100523948669434,
            -19.417020797729492,
            -9.049156188964844,
            -20.48767852783203,
            -94.35086059570312,
            -23.96303367614746,
        ]


def test_jagged_strided_awkward():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="ak")

        assert (
            str(awkward.type(result))
            == "2421 * var * TLorentzVector[fP: TVector3[fX: float64, "
            "fY: float64, fZ: float64], fE: float64]"
        )

        assert result[0, 0, "fE"] == 54.77949905395508
        assert result[0, 0, "fP", "fX"] == -52.89945602416992
        assert result[0, 0, "fP", "fY"] == -11.654671669006348
        assert result[0, 0, "fP", "fZ"] == -8.16079330444336


def test_jagged_awkward_1():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/evt/SliceU64"
    ] as branch:
        assert branch.interpretation == AsJagged(AsDtype(">u8"), header_bytes=1)
        result = branch.array(library="ak", entry_stop=6)
        assert awkward.to_list(result) == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ]


def test_jagged_awkward_2():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/evt/StlVecF64"
    ] as branch:
        assert branch.interpretation == AsJagged(AsDtype(">f8"), header_bytes=10)
        result = branch.array(library="ak", entry_stop=6)
        assert awkward.to_list(result) == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ]


def test_general_awkward_form():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        context = {
            "index_format": "i64",
            "header": False,
            "tobject_header": True,
            "breadcrumbs": (),
        }

        assert json.loads(
            branch.interpretation.awkward_form(branch.file, context).to_json()
        ) == json.loads(
            """{
    "class": "RecordArray",
    "fields": [
        "Beg",
        "I16",
        "I32",
        "I64",
        "U16",
        "U32",
        "U64",
        "F32",
        "F64",
        "Str",
        "P3",
        "ArrayI16",
        "ArrayI32",
        "ArrayI64",
        "ArrayU16",
        "ArrayU32",
        "ArrayU64",
        "ArrayF32",
        "ArrayF64",
        "N",
        "SliceI16",
        "SliceI32",
        "SliceI64",
        "SliceU16",
        "SliceU32",
        "SliceU64",
        "SliceF32",
        "SliceF64",
        "StdStr",
        "StlVecI16",
        "StlVecI32",
        "StlVecI64",
        "StlVecU16",
        "StlVecU32",
        "StlVecU64",
        "StlVecF32",
        "StlVecF64",
        "StlVecStr",
        "End"
    ],
    "contents": [
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint8",
                "inner_shape": [],
                "parameters": {
                    "__array__": "char"
                },
                "form_key": null
            },
            "parameters": {
                "__array__": "string"
            },
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "int16",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "int64",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "uint16",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "uint32",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "uint64",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "float32",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "float64",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint8",
                "inner_shape": [],
                "parameters": {
                    "__array__": "char"
                },
                "form_key": null
            },
            "parameters": {
                "__array__": "string"
            },
            "form_key": null
        },
        {
            "class": "RecordArray",
            "fields": [
                "Px",
                "Py",
                "Pz"
            ],
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int32",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": null
                },
                {
                    "class": "NumpyArray",
                    "primitive": "float64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": null
                },
                {
                    "class": "NumpyArray",
                    "primitive": "int32",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": null
                }
            ],
            "parameters": {
                "__record__": "P3"
            },
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "int16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "int32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "uint16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "uint32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "uint64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "float32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "RegularArray",
            "size": 10,
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "NumpyArray",
            "primitive": "uint32",
            "inner_shape": [],
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint8",
                "inner_shape": [],
                "parameters": {
                    "__array__": "char"
                },
                "form_key": null
            },
            "parameters": {
                "__array__": "string"
            },
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint16",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float32",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "inner_shape": [],
                "parameters": {},
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "uint8",
                    "inner_shape": [],
                    "parameters": {
                        "__array__": "char"
                    },
                    "form_key": null
                },
                "parameters": {
                    "__array__": "string"
                },
                "form_key": null
            },
            "parameters": {},
            "form_key": null
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "uint8",
                "inner_shape": [],
                "parameters": {
                    "__array__": "char"
                },
                "form_key": null
            },
            "parameters": {
                "__array__": "string"
            },
            "form_key": null
        }
    ],
    "parameters": {
        "__record__": "Event"
    },
    "form_key": null
}"""
        )


def test_jagged_awkward_3():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/evt/StlVecStr"
    ] as branch:
        assert awkward.to_list(branch.array(library="ak")[:6, :3]) == [
            [],
            ["vec-001"],
            ["vec-002", "vec-002"],
            ["vec-003", "vec-003", "vec-003"],
            ["vec-004", "vec-004", "vec-004"],
            ["vec-005", "vec-005", "vec-005"],
        ]


def test_awkward_string():
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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


def test_awkward_vector_tstring():
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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


def test_awkward_map_int32_int16():
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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


def test_awkward_map_string_int16():
    awkward = pytest.importorskip("awkward")
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


def test_awkward_map_string_vector_string():
    awkward = pytest.importorskip("awkward")
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


def test_awkward_map_int32_vector_vector_int16():
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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
    awkward = pytest.importorskip("awkward")
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


def test_awkward_map_int_struct():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-issue468.root"))[
        "Geant4Data/Geant4Data./Geant4Data.particles"
    ] as branch:
        assert (
            repr(branch.interpretation) == "AsObjects(AsMap(True, dtype('>i4'), "
            "Model_BDSOutputROOTGeant4Data_3a3a_ParticleInfo))"
        )
        result = branch.array(library="ak")
        assert (
            str(awkward.type(result))
            == '1 * var * tuple[[int32, struct[{name: string, charge: int32, mass: float64}, parameters={"__record__": "BDSOutputROOTGeant4Data::ParticleInfo"}]], parameters={"__array__": "sorted_map"}]'
        )
        assert awkward.to_list(result[0, "0"]) == [
            -1000020040,
            -1000020030,
            -1000010030,
            -1000010020,
            -2212,
            -2112,
            -321,
            -211,
            -16,
            -15,
            -14,
            -13,
            -12,
            -11,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            22,
            211,
            321,
            2112,
            2212,
            1000010020,
            1000010030,
            1000020030,
            1000020040,
        ]
        assert awkward.to_list(result[0, "1", "name"]) == [
            "anti_alpha",
            "anti_He3",
            "anti_triton",
            "anti_deuteron",
            "anti_proton",
            "anti_neutron",
            "kaon-",
            "pi-",
            "anti_nu_tau",
            "tau+",
            "anti_nu_mu",
            "mu+",
            "anti_nu_e",
            "e+",
            "geantino",
            "e-",
            "nu_e",
            "mu-",
            "nu_mu",
            "tau-",
            "nu_tau",
            "gamma",
            "pi+",
            "kaon+",
            "neutron",
            "proton",
            "deuteron",
            "triton",
            "He3",
            "alpha",
        ]
        assert awkward.to_list(result[0, "1", "charge"]) == [
            -2,
            -2,
            -1,
            -1,
            -1,
            0,
            -1,
            -1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            -1,
            0,
            -1,
            0,
            -1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            2,
            2,
        ]
        assert awkward.to_list(result[0, "1", "mass"]) == [
            3.727379,
            2.808391,
            2.808921,
            1.875613,
            0.938272013,
            0.93956536,
            0.493677,
            0.1395701,
            0.0,
            1.77686,
            0.0,
            0.1056583715,
            0.0,
            0.00051099891,
            0.0,
            0.00051099891,
            0.0,
            0.1056583715,
            0.0,
            1.77686,
            0.0,
            0.0,
            0.1395701,
            0.493677,
            0.93956536,
            0.938272013,
            1.875613,
            2.808921,
            2.808391,
            3.727379,
        ]


def test_awkward_nosplit_file():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        result = branch.array(library="ak", entry_stop=5)
        assert awkward.to_list(result["Beg"]) == [
            "beg-000",
            "beg-001",
            "beg-002",
            "beg-003",
            "beg-004",
        ]
        assert awkward.to_list(result["I16"]) == [0, 1, 2, 3, 4]
        assert awkward.to_list(result["F32"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert awkward.to_list(result["Str"]) == [
            "evt-000",
            "evt-001",
            "evt-002",
            "evt-003",
            "evt-004",
        ]
        assert awkward.to_list(result["P3", "Px"]) == [-1, 0, 1, 2, 3]
        assert awkward.to_list(result["P3", "Py"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert awkward.to_list(result["P3", "Pz"]) == [-1, 0, 1, 2, 3]
        assert awkward.to_list(result["ArrayI32"]) == [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ]
        assert awkward.to_list(result["StdStr"]) == [
            "std-000",
            "std-001",
            "std-002",
            "std-003",
            "std-004",
        ]
        assert awkward.to_list(result["SliceI64"]) == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
        ]
        assert awkward.to_list(result["StlVecStr"]) == [
            [],
            ["vec-001"],
            ["vec-002", "vec-002"],
            ["vec-003", "vec-003", "vec-003"],
            ["vec-004", "vec-004", "vec-004", "vec-004"],
        ]
        assert awkward.to_list(result["End"]) == [
            "end-000",
            "end-001",
            "end-002",
            "end-003",
            "end-004",
        ]


def test_pandas_TVector2():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="pd")

        assert result.ak["fX"][0] == 5.912771224975586
        assert result.ak["fY"][0] == 2.5636332035064697


def test_pandas_vector_TLorentzVector():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="pd")

        assert result.ak["fP", "fX"][0] == [
            -52.89945602416992,
            37.7377815246582,
        ]
        assert result.ak["fP", "fY"][0] == [
            -11.654671669006348,
            0.6934735774993896,
        ]
        assert result.ak["fP", "fZ"][0] == [
            -8.16079330444336,
            -11.307581901550293,
        ]
        assert result.ak["fE"][0] == [
            54.77949905395508,
            39.401695251464844,
        ]


def test_map_string_TVector3():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-issue494.root"))[
        "Model/Model.scoringMeshTranslation"
    ] as branch:
        result = branch.array(library="ak")
        assert awkward.to_list(result["0"]) == [
            [
                "global_mesh",
                "mesh_foil1",
                "mesh_foil10",
                "mesh_foil11",
                "mesh_foil2",
                "mesh_foil3",
                "mesh_foil4",
                "mesh_foil5",
                "mesh_foil6",
                "mesh_foil7",
                "mesh_foil8",
                "mesh_foil9",
                "mesh_t1",
                "mesh_t2",
                "mesh_t3",
                "mesh_t4",
                "mesh_t5",
                "mesh_t6",
                "mesh_t7",
            ]
        ]
        assert awkward.to_list(result["1"]) == [
            [
                {"fX": 0.0, "fY": 0.0, "fZ": 0.074},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.048509515},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.09400956000000006},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.09500956300000007},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.049509518},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.050509521},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.079009536},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.08000953900000002},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.08100954200000002},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.08200954500000003},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.09200955400000005},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.09300955700000006},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.038000009},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.054000023999999994},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.06850002999999999},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.08550004800000004},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.09850006600000008},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.11300007200000009},
                {"fX": 0.0, "fY": 0.04, "fZ": 0.13600007800000008},
            ]
        ]


def test_gohep_output_file():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-issue413.root"))[
        "mytree"
    ] as tree:
        assert awkward.to_list(tree["I32"].array()) == [0, 1, 2, 3, 4]
        assert awkward.to_list(tree["F64"].array()) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert list(tree["Str"].array()) == [
            "evt-0",
            "evt-1",
            "evt-2",
            "evt-3",
            "evt-4",
        ]
        assert awkward.to_list(tree["ArrF64"].array()) == [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
        ]
        assert awkward.to_list(tree["N"].array()) == [0, 1, 2, 3, 4]
        assert awkward.to_list(tree["SliF64"].array()) == [
            [],
            [1.0],
            [2.0, 3.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
