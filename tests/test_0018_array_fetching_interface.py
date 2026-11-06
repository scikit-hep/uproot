# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_leaf_interpretation():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert repr(sample["n"].interpretation) == """AsDtype('>i4')"""
        assert repr(sample["b"].interpretation) == """AsDtype('bool')"""
        assert repr(sample["ab"].interpretation) == """AsDtype("('?', (3,))")"""
        assert repr(sample["Ab"].interpretation) == """AsJagged(AsDtype('bool'))"""
        assert repr(sample["i1"].interpretation) == """AsDtype('int8')"""
        assert repr(sample["ai1"].interpretation) == """AsDtype("('i1', (3,))")"""
        assert repr(sample["Ai1"].interpretation) == """AsJagged(AsDtype('int8'))"""
        assert repr(sample["u1"].interpretation) == """AsDtype('uint8')"""
        assert repr(sample["au1"].interpretation) == """AsDtype("('u1', (3,))")"""
        assert repr(sample["Au1"].interpretation) == """AsJagged(AsDtype('uint8'))"""
        assert repr(sample["i2"].interpretation) == """AsDtype('>i2')"""
        assert repr(sample["ai2"].interpretation) == """AsDtype("('>i2', (3,))")"""
        assert repr(sample["Ai2"].interpretation) == """AsJagged(AsDtype('>i2'))"""
        assert repr(sample["u2"].interpretation) == """AsDtype('>u2')"""
        assert repr(sample["au2"].interpretation) == """AsDtype("('>u2', (3,))")"""
        assert repr(sample["Au2"].interpretation) == """AsJagged(AsDtype('>u2'))"""
        assert repr(sample["i4"].interpretation) == """AsDtype('>i4')"""
        assert repr(sample["ai4"].interpretation) == """AsDtype("('>i4', (3,))")"""
        assert repr(sample["Ai4"].interpretation) == """AsJagged(AsDtype('>i4'))"""
        assert repr(sample["u4"].interpretation) == """AsDtype('>u4')"""
        assert repr(sample["au4"].interpretation) == """AsDtype("('>u4', (3,))")"""
        assert repr(sample["Au4"].interpretation) == """AsJagged(AsDtype('>u4'))"""
        assert repr(sample["i8"].interpretation) == """AsDtype('>i8')"""
        assert repr(sample["ai8"].interpretation) == """AsDtype("('>i8', (3,))")"""
        assert repr(sample["Ai8"].interpretation) == """AsJagged(AsDtype('>i8'))"""
        assert repr(sample["u8"].interpretation) == """AsDtype('>u8')"""
        assert repr(sample["au8"].interpretation) == """AsDtype("('>u8', (3,))")"""
        assert repr(sample["Au8"].interpretation) == """AsJagged(AsDtype('>u8'))"""
        assert repr(sample["f4"].interpretation) == """AsDtype('>f4')"""
        assert repr(sample["af4"].interpretation) == """AsDtype("('>f4', (3,))")"""
        assert repr(sample["Af4"].interpretation) == """AsJagged(AsDtype('>f4'))"""
        assert repr(sample["f8"].interpretation) == """AsDtype('>f8')"""
        assert repr(sample["af8"].interpretation) == """AsDtype("('>f8', (3,))")"""
        assert repr(sample["Af8"].interpretation) == """AsJagged(AsDtype('>f8'))"""

        assert sample["n"].typename == "int32_t"
        assert sample["b"].typename == "bool"
        assert sample["ab"].typename == "bool[3]"
        assert sample["Ab"].typename == "bool[]"
        assert sample["i1"].typename == "int8_t"
        assert sample["ai1"].typename == "int8_t[3]"
        assert sample["Ai1"].typename == "int8_t[]"
        assert sample["u1"].typename == "uint8_t"
        assert sample["au1"].typename == "uint8_t[3]"
        assert sample["Au1"].typename == "uint8_t[]"
        assert sample["i2"].typename == "int16_t"
        assert sample["ai2"].typename == "int16_t[3]"
        assert sample["Ai2"].typename == "int16_t[]"
        assert sample["u2"].typename == "uint16_t"
        assert sample["au2"].typename == "uint16_t[3]"
        assert sample["Au2"].typename == "uint16_t[]"
        assert sample["i4"].typename == "int32_t"
        assert sample["ai4"].typename == "int32_t[3]"
        assert sample["Ai4"].typename == "int32_t[]"
        assert sample["u4"].typename == "uint32_t"
        assert sample["au4"].typename == "uint32_t[3]"
        assert sample["Au4"].typename == "uint32_t[]"
        assert sample["i8"].typename == "int64_t"
        assert sample["ai8"].typename == "int64_t[3]"
        assert sample["Ai8"].typename == "int64_t[]"
        assert sample["u8"].typename == "uint64_t"
        assert sample["au8"].typename == "uint64_t[3]"
        assert sample["Au8"].typename == "uint64_t[]"
        assert sample["f4"].typename == "float"
        assert sample["af4"].typename == "float[3]"
        assert sample["Af4"].typename == "float[]"
        assert sample["f8"].typename == "double"
        assert sample["af8"].typename == "double[3]"
        assert sample["Af8"].typename == "double[]"


def test_compute():
    awkward = pytest.importorskip("awkward")

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        object_cache=100,
        array_cache="100 MB",
    )["sample"] as sample:
        result = sample.arrays(["stuff", "i4"], aliases={"stuff": "abs(i4) + Ai8"})
        assert result.tolist() == [
            {"stuff": [], "i4": -15},
            {"stuff": [-1], "i4": -14},
            {"stuff": [-2, 0], "i4": -13},
            {"stuff": [-3, -1, 1], "i4": -12},
            {"stuff": [-4, -2, 0, 2], "i4": -11},
            {"stuff": [], "i4": -10},
            {"stuff": [-1], "i4": -9},
            {"stuff": [-2, 0], "i4": -8},
            {"stuff": [-3, -1, 1], "i4": -7},
            {"stuff": [-4, -2, 0, 2], "i4": -6},
            {"stuff": [], "i4": -5},
            {"stuff": [-1], "i4": -4},
            {"stuff": [-2, 0], "i4": -3},
            {"stuff": [-3, -1, 1], "i4": -2},
            {"stuff": [-4, -2, 0, 2], "i4": -1},
            {"stuff": [], "i4": 0},
            {"stuff": [1], "i4": 1},
            {"stuff": [2, 4], "i4": 2},
            {"stuff": [3, 5, 7], "i4": 3},
            {"stuff": [4, 6, 8, 10], "i4": 4},
            {"stuff": [], "i4": 5},
            {"stuff": [11], "i4": 6},
            {"stuff": [12, 14], "i4": 7},
            {"stuff": [13, 15, 17], "i4": 8},
            {"stuff": [14, 16, 18, 20], "i4": 9},
            {"stuff": [], "i4": 10},
            {"stuff": [21], "i4": 11},
            {"stuff": [22, 24], "i4": 12},
            {"stuff": [23, 25, 27], "i4": 13},
            {"stuff": [24, 26, 28, 30], "i4": 14},
        ]

        assert set(sample.file.array_cache) == {
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:i4:AsDtype(Bi4(),Li4()):0-30:ak",
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:Ai8:AsJagged(AsDtype(Bi8(),Li8()),0):0-30:ak",
        }


def test_arrays():
    awkward = pytest.importorskip("awkward")

    interp_i4 = uproot.interpretation.numerical.AsDtype(">i4")
    interp_f4 = uproot.interpretation.numerical.AsDtype(">f4")
    interp_Ai4 = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">i4")
    )
    interp_Af8 = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">f8")
    )

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        object_cache=100,
        array_cache="100 MB",
    )["sample"] as sample:
        result = sample.arrays(["I4", "F4"], aliases={"I4": "i4", "F4": "f4"})
        assert result.tolist() == [
            {"I4": -15, "F4": -14.899999618530273},
            {"I4": -14, "F4": -13.899999618530273},
            {"I4": -13, "F4": -12.899999618530273},
            {"I4": -12, "F4": -11.899999618530273},
            {"I4": -11, "F4": -10.899999618530273},
            {"I4": -10, "F4": -9.899999618530273},
            {"I4": -9, "F4": -8.899999618530273},
            {"I4": -8, "F4": -7.900000095367432},
            {"I4": -7, "F4": -6.900000095367432},
            {"I4": -6, "F4": -5.900000095367432},
            {"I4": -5, "F4": -4.900000095367432},
            {"I4": -4, "F4": -3.9000000953674316},
            {"I4": -3, "F4": -2.9000000953674316},
            {"I4": -2, "F4": -1.899999976158142},
            {"I4": -1, "F4": -0.8999999761581421},
            {"I4": 0, "F4": 0.10000000149011612},
            {"I4": 1, "F4": 1.100000023841858},
            {"I4": 2, "F4": 2.0999999046325684},
            {"I4": 3, "F4": 3.0999999046325684},
            {"I4": 4, "F4": 4.099999904632568},
            {"I4": 5, "F4": 5.099999904632568},
            {"I4": 6, "F4": 6.099999904632568},
            {"I4": 7, "F4": 7.099999904632568},
            {"I4": 8, "F4": 8.100000381469727},
            {"I4": 9, "F4": 9.100000381469727},
            {"I4": 10, "F4": 10.100000381469727},
            {"I4": 11, "F4": 11.100000381469727},
            {"I4": 12, "F4": 12.100000381469727},
            {"I4": 13, "F4": 13.100000381469727},
            {"I4": 14, "F4": 14.100000381469727},
        ]

        assert set(sample.file.array_cache) == {
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:i4:AsDtype(Bi4(),Li4()):0-30:ak",
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:f4:AsDtype(Bf4(),Lf4()):0-30:ak",
        }

        result = sample.arrays({"i4": interp_i4, "f4": interp_f4})
        assert result.tolist() == [
            {"i4": -15, "f4": -14.899999618530273},
            {"i4": -14, "f4": -13.899999618530273},
            {"i4": -13, "f4": -12.899999618530273},
            {"i4": -12, "f4": -11.899999618530273},
            {"i4": -11, "f4": -10.899999618530273},
            {"i4": -10, "f4": -9.899999618530273},
            {"i4": -9, "f4": -8.899999618530273},
            {"i4": -8, "f4": -7.900000095367432},
            {"i4": -7, "f4": -6.900000095367432},
            {"i4": -6, "f4": -5.900000095367432},
            {"i4": -5, "f4": -4.900000095367432},
            {"i4": -4, "f4": -3.9000000953674316},
            {"i4": -3, "f4": -2.9000000953674316},
            {"i4": -2, "f4": -1.899999976158142},
            {"i4": -1, "f4": -0.8999999761581421},
            {"i4": 0, "f4": 0.10000000149011612},
            {"i4": 1, "f4": 1.100000023841858},
            {"i4": 2, "f4": 2.0999999046325684},
            {"i4": 3, "f4": 3.0999999046325684},
            {"i4": 4, "f4": 4.099999904632568},
            {"i4": 5, "f4": 5.099999904632568},
            {"i4": 6, "f4": 6.099999904632568},
            {"i4": 7, "f4": 7.099999904632568},
            {"i4": 8, "f4": 8.100000381469727},
            {"i4": 9, "f4": 9.100000381469727},
            {"i4": 10, "f4": 10.100000381469727},
            {"i4": 11, "f4": 11.100000381469727},
            {"i4": 12, "f4": 12.100000381469727},
            {"i4": 13, "f4": 13.100000381469727},
            {"i4": 14, "f4": 14.100000381469727},
        ]

        result = sample.arrays(
            {"i4": interp_i4, "f4": interp_f4, "Ai4": interp_Ai4, "Af8": interp_Af8},
        )
        assert result.tolist() == [
            {"i4": -15, "f4": -14.899999618530273, "Ai4": [], "Af8": []},
            {"i4": -14, "f4": -13.899999618530273, "Ai4": [-15], "Af8": [-15.0]},
            {
                "i4": -13,
                "f4": -12.899999618530273,
                "Ai4": [-15, -13],
                "Af8": [-15.0, -13.9],
            },
            {
                "i4": -12,
                "f4": -11.899999618530273,
                "Ai4": [-15, -13, -11],
                "Af8": [-15.0, -13.9, -12.8],
            },
            {
                "i4": -11,
                "f4": -10.899999618530273,
                "Ai4": [-15, -13, -11, -9],
                "Af8": [-15.0, -13.9, -12.8, -11.7],
            },
            {"i4": -10, "f4": -9.899999618530273, "Ai4": [], "Af8": []},
            {"i4": -9, "f4": -8.899999618530273, "Ai4": [-10], "Af8": [-10.0]},
            {
                "i4": -8,
                "f4": -7.900000095367432,
                "Ai4": [-10, -8],
                "Af8": [-10.0, -8.9],
            },
            {
                "i4": -7,
                "f4": -6.900000095367432,
                "Ai4": [-10, -8, -6],
                "Af8": [-10.0, -8.9, -7.8],
            },
            {
                "i4": -6,
                "f4": -5.900000095367432,
                "Ai4": [-10, -8, -6, -4],
                "Af8": [-10.0, -8.9, -7.8, -6.7],
            },
            {"i4": -5, "f4": -4.900000095367432, "Ai4": [], "Af8": []},
            {"i4": -4, "f4": -3.9000000953674316, "Ai4": [-5], "Af8": [-5.0]},
            {"i4": -3, "f4": -2.9000000953674316, "Ai4": [-5, -3], "Af8": [-5.0, -3.9]},
            {
                "i4": -2,
                "f4": -1.899999976158142,
                "Ai4": [-5, -3, -1],
                "Af8": [-5.0, -3.9, -2.8],
            },
            {
                "i4": -1,
                "f4": -0.8999999761581421,
                "Ai4": [-5, -3, -1, 1],
                "Af8": [-5.0, -3.9, -2.8, -1.7],
            },
            {"i4": 0, "f4": 0.10000000149011612, "Ai4": [], "Af8": []},
            {"i4": 1, "f4": 1.100000023841858, "Ai4": [0], "Af8": [0.0]},
            {"i4": 2, "f4": 2.0999999046325684, "Ai4": [0, 2], "Af8": [0.0, 1.1]},
            {
                "i4": 3,
                "f4": 3.0999999046325684,
                "Ai4": [0, 2, 4],
                "Af8": [0.0, 1.1, 2.2],
            },
            {
                "i4": 4,
                "f4": 4.099999904632568,
                "Ai4": [0, 2, 4, 6],
                "Af8": [0.0, 1.1, 2.2, 3.3],
            },
            {"i4": 5, "f4": 5.099999904632568, "Ai4": [], "Af8": []},
            {"i4": 6, "f4": 6.099999904632568, "Ai4": [5], "Af8": [5.0]},
            {"i4": 7, "f4": 7.099999904632568, "Ai4": [5, 7], "Af8": [5.0, 6.1]},
            {
                "i4": 8,
                "f4": 8.100000381469727,
                "Ai4": [5, 7, 9],
                "Af8": [5.0, 6.1, 7.2],
            },
            {
                "i4": 9,
                "f4": 9.100000381469727,
                "Ai4": [5, 7, 9, 11],
                "Af8": [5.0, 6.1, 7.2, 8.3],
            },
            {"i4": 10, "f4": 10.100000381469727, "Ai4": [], "Af8": []},
            {"i4": 11, "f4": 11.100000381469727, "Ai4": [10], "Af8": [10.0]},
            {"i4": 12, "f4": 12.100000381469727, "Ai4": [10, 12], "Af8": [10.0, 11.1]},
            {
                "i4": 13,
                "f4": 13.100000381469727,
                "Ai4": [10, 12, 14],
                "Af8": [10.0, 11.1, 12.2],
            },
            {
                "i4": 14,
                "f4": 14.100000381469727,
                "Ai4": [10, 12, 14, 16],
                "Af8": [10.0, 11.1, 12.2, 13.3],
            },
        ]

        result = sample.arrays(
            {"i4": interp_i4, "f4": interp_f4, "Ai4": interp_Ai4, "Af8": interp_Af8},
            how="zip",
        )
        assert result.tolist() == [
            {"i4": -15, "f4": -14.899999618530273, "jagged0": []},
            {
                "i4": -14,
                "f4": -13.899999618530273,
                "jagged0": [{"Ai4": -15, "Af8": -15.0}],
            },
            {
                "i4": -13,
                "f4": -12.899999618530273,
                "jagged0": [{"Ai4": -15, "Af8": -15.0}, {"Ai4": -13, "Af8": -13.9}],
            },
            {
                "i4": -12,
                "f4": -11.899999618530273,
                "jagged0": [
                    {"Ai4": -15, "Af8": -15.0},
                    {"Ai4": -13, "Af8": -13.9},
                    {"Ai4": -11, "Af8": -12.8},
                ],
            },
            {
                "i4": -11,
                "f4": -10.899999618530273,
                "jagged0": [
                    {"Ai4": -15, "Af8": -15.0},
                    {"Ai4": -13, "Af8": -13.9},
                    {"Ai4": -11, "Af8": -12.8},
                    {"Ai4": -9, "Af8": -11.7},
                ],
            },
            {"i4": -10, "f4": -9.899999618530273, "jagged0": []},
            {
                "i4": -9,
                "f4": -8.899999618530273,
                "jagged0": [{"Ai4": -10, "Af8": -10.0}],
            },
            {
                "i4": -8,
                "f4": -7.900000095367432,
                "jagged0": [{"Ai4": -10, "Af8": -10.0}, {"Ai4": -8, "Af8": -8.9}],
            },
            {
                "i4": -7,
                "f4": -6.900000095367432,
                "jagged0": [
                    {"Ai4": -10, "Af8": -10.0},
                    {"Ai4": -8, "Af8": -8.9},
                    {"Ai4": -6, "Af8": -7.8},
                ],
            },
            {
                "i4": -6,
                "f4": -5.900000095367432,
                "jagged0": [
                    {"Ai4": -10, "Af8": -10.0},
                    {"Ai4": -8, "Af8": -8.9},
                    {"Ai4": -6, "Af8": -7.8},
                    {"Ai4": -4, "Af8": -6.7},
                ],
            },
            {"i4": -5, "f4": -4.900000095367432, "jagged0": []},
            {
                "i4": -4,
                "f4": -3.9000000953674316,
                "jagged0": [{"Ai4": -5, "Af8": -5.0}],
            },
            {
                "i4": -3,
                "f4": -2.9000000953674316,
                "jagged0": [{"Ai4": -5, "Af8": -5.0}, {"Ai4": -3, "Af8": -3.9}],
            },
            {
                "i4": -2,
                "f4": -1.899999976158142,
                "jagged0": [
                    {"Ai4": -5, "Af8": -5.0},
                    {"Ai4": -3, "Af8": -3.9},
                    {"Ai4": -1, "Af8": -2.8},
                ],
            },
            {
                "i4": -1,
                "f4": -0.8999999761581421,
                "jagged0": [
                    {"Ai4": -5, "Af8": -5.0},
                    {"Ai4": -3, "Af8": -3.9},
                    {"Ai4": -1, "Af8": -2.8},
                    {"Ai4": 1, "Af8": -1.7},
                ],
            },
            {"i4": 0, "f4": 0.10000000149011612, "jagged0": []},
            {"i4": 1, "f4": 1.100000023841858, "jagged0": [{"Ai4": 0, "Af8": 0.0}]},
            {
                "i4": 2,
                "f4": 2.0999999046325684,
                "jagged0": [{"Ai4": 0, "Af8": 0.0}, {"Ai4": 2, "Af8": 1.1}],
            },
            {
                "i4": 3,
                "f4": 3.0999999046325684,
                "jagged0": [
                    {"Ai4": 0, "Af8": 0.0},
                    {"Ai4": 2, "Af8": 1.1},
                    {"Ai4": 4, "Af8": 2.2},
                ],
            },
            {
                "i4": 4,
                "f4": 4.099999904632568,
                "jagged0": [
                    {"Ai4": 0, "Af8": 0.0},
                    {"Ai4": 2, "Af8": 1.1},
                    {"Ai4": 4, "Af8": 2.2},
                    {"Ai4": 6, "Af8": 3.3},
                ],
            },
            {"i4": 5, "f4": 5.099999904632568, "jagged0": []},
            {"i4": 6, "f4": 6.099999904632568, "jagged0": [{"Ai4": 5, "Af8": 5.0}]},
            {
                "i4": 7,
                "f4": 7.099999904632568,
                "jagged0": [{"Ai4": 5, "Af8": 5.0}, {"Ai4": 7, "Af8": 6.1}],
            },
            {
                "i4": 8,
                "f4": 8.100000381469727,
                "jagged0": [
                    {"Ai4": 5, "Af8": 5.0},
                    {"Ai4": 7, "Af8": 6.1},
                    {"Ai4": 9, "Af8": 7.2},
                ],
            },
            {
                "i4": 9,
                "f4": 9.100000381469727,
                "jagged0": [
                    {"Ai4": 5, "Af8": 5.0},
                    {"Ai4": 7, "Af8": 6.1},
                    {"Ai4": 9, "Af8": 7.2},
                    {"Ai4": 11, "Af8": 8.3},
                ],
            },
            {"i4": 10, "f4": 10.100000381469727, "jagged0": []},
            {"i4": 11, "f4": 11.100000381469727, "jagged0": [{"Ai4": 10, "Af8": 10.0}]},
            {
                "i4": 12,
                "f4": 12.100000381469727,
                "jagged0": [{"Ai4": 10, "Af8": 10.0}, {"Ai4": 12, "Af8": 11.1}],
            },
            {
                "i4": 13,
                "f4": 13.100000381469727,
                "jagged0": [
                    {"Ai4": 10, "Af8": 10.0},
                    {"Ai4": 12, "Af8": 11.1},
                    {"Ai4": 14, "Af8": 12.2},
                ],
            },
            {
                "i4": 14,
                "f4": 14.100000381469727,
                "jagged0": [
                    {"Ai4": 10, "Af8": 10.0},
                    {"Ai4": 12, "Af8": 11.1},
                    {"Ai4": 14, "Af8": 12.2},
                    {"Ai4": 16, "Af8": 13.3},
                ],
            },
        ]

        result = sample.arrays(
            {"i4": interp_i4, "f4": interp_f4, "Ai4": interp_Ai4, "Af8": interp_Af8},
            entry_start=5,
            entry_stop=-5,
            how="zip",
        )
        assert result.tolist() == [
            {"i4": -10, "f4": -9.899999618530273, "jagged0": []},
            {
                "i4": -9,
                "f4": -8.899999618530273,
                "jagged0": [{"Ai4": -10, "Af8": -10.0}],
            },
            {
                "i4": -8,
                "f4": -7.900000095367432,
                "jagged0": [{"Ai4": -10, "Af8": -10.0}, {"Ai4": -8, "Af8": -8.9}],
            },
            {
                "i4": -7,
                "f4": -6.900000095367432,
                "jagged0": [
                    {"Ai4": -10, "Af8": -10.0},
                    {"Ai4": -8, "Af8": -8.9},
                    {"Ai4": -6, "Af8": -7.8},
                ],
            },
            {
                "i4": -6,
                "f4": -5.900000095367432,
                "jagged0": [
                    {"Ai4": -10, "Af8": -10.0},
                    {"Ai4": -8, "Af8": -8.9},
                    {"Ai4": -6, "Af8": -7.8},
                    {"Ai4": -4, "Af8": -6.7},
                ],
            },
            {"i4": -5, "f4": -4.900000095367432, "jagged0": []},
            {
                "i4": -4,
                "f4": -3.9000000953674316,
                "jagged0": [{"Ai4": -5, "Af8": -5.0}],
            },
            {
                "i4": -3,
                "f4": -2.9000000953674316,
                "jagged0": [{"Ai4": -5, "Af8": -5.0}, {"Ai4": -3, "Af8": -3.9}],
            },
            {
                "i4": -2,
                "f4": -1.899999976158142,
                "jagged0": [
                    {"Ai4": -5, "Af8": -5.0},
                    {"Ai4": -3, "Af8": -3.9},
                    {"Ai4": -1, "Af8": -2.8},
                ],
            },
            {
                "i4": -1,
                "f4": -0.8999999761581421,
                "jagged0": [
                    {"Ai4": -5, "Af8": -5.0},
                    {"Ai4": -3, "Af8": -3.9},
                    {"Ai4": -1, "Af8": -2.8},
                    {"Ai4": 1, "Af8": -1.7},
                ],
            },
            {"i4": 0, "f4": 0.10000000149011612, "jagged0": []},
            {"i4": 1, "f4": 1.100000023841858, "jagged0": [{"Ai4": 0, "Af8": 0.0}]},
            {
                "i4": 2,
                "f4": 2.0999999046325684,
                "jagged0": [{"Ai4": 0, "Af8": 0.0}, {"Ai4": 2, "Af8": 1.1}],
            },
            {
                "i4": 3,
                "f4": 3.0999999046325684,
                "jagged0": [
                    {"Ai4": 0, "Af8": 0.0},
                    {"Ai4": 2, "Af8": 1.1},
                    {"Ai4": 4, "Af8": 2.2},
                ],
            },
            {
                "i4": 4,
                "f4": 4.099999904632568,
                "jagged0": [
                    {"Ai4": 0, "Af8": 0.0},
                    {"Ai4": 2, "Af8": 1.1},
                    {"Ai4": 4, "Af8": 2.2},
                    {"Ai4": 6, "Af8": 3.3},
                ],
            },
            {"i4": 5, "f4": 5.099999904632568, "jagged0": []},
            {"i4": 6, "f4": 6.099999904632568, "jagged0": [{"Ai4": 5, "Af8": 5.0}]},
            {
                "i4": 7,
                "f4": 7.099999904632568,
                "jagged0": [{"Ai4": 5, "Af8": 5.0}, {"Ai4": 7, "Af8": 6.1}],
            },
            {
                "i4": 8,
                "f4": 8.100000381469727,
                "jagged0": [
                    {"Ai4": 5, "Af8": 5.0},
                    {"Ai4": 7, "Af8": 6.1},
                    {"Ai4": 9, "Af8": 7.2},
                ],
            },
            {
                "i4": 9,
                "f4": 9.100000381469727,
                "jagged0": [
                    {"Ai4": 5, "Af8": 5.0},
                    {"Ai4": 7, "Af8": 6.1},
                    {"Ai4": 9, "Af8": 7.2},
                    {"Ai4": 11, "Af8": 8.3},
                ],
            },
        ]


def test_jagged():
    interpretation = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">i2")
    )

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation, library="np")
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]

        result = branch.array(
            interpretation, entry_start=3, entry_stop=-6, library="np"
        )
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
        ]


def test_jagged_awkward():
    awkward = pytest.importorskip("awkward")

    interpretation = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">i2")
    )

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation)
        assert awkward.to_list(result) == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]


def test_jagged_pandas():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("awkward")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")

    interpretation = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">i2")
    )

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation, library="pd")
        assert result.index.tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        ]

        assert result.values.tolist() == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]


def test_stl_vector():
    interpretation = uproot.interpretation.jagged.AsJagged(
        uproot.interpretation.numerical.AsDtype(">i4"), header_bytes=10
    )

    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/StlVecI32"
    ] as branch:
        result = branch.array(interpretation, library="np")
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [],
            [11],
            [12, 12],
            [13, 13, 13],
            [14, 14, 14, 14],
            [15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 16],
            [17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 19, 19],
            [],
            [21],
            [22, 22],
            [23, 23, 23],
            [24, 24, 24, 24],
            [25, 25, 25, 25, 25],
            [26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 28, 28, 28, 28, 28],
            [29, 29, 29, 29, 29, 29, 29, 29, 29],
            [],
            [31],
            [32, 32],
            [33, 33, 33],
            [34, 34, 34, 34],
            [35, 35, 35, 35, 35],
            [36, 36, 36, 36, 36, 36],
            [37, 37, 37, 37, 37, 37, 37],
            [38, 38, 38, 38, 38, 38, 38, 38],
            [39, 39, 39, 39, 39, 39, 39, 39, 39],
            [],
            [41],
            [42, 42],
            [43, 43, 43],
            [44, 44, 44, 44],
            [45, 45, 45, 45, 45],
            [46, 46, 46, 46, 46, 46],
            [47, 47, 47, 47, 47, 47, 47],
            [48, 48, 48, 48, 48, 48, 48, 48],
            [49, 49, 49, 49, 49, 49, 49, 49, 49],
            [],
            [51],
            [52, 52],
            [53, 53, 53],
            [54, 54, 54, 54],
            [55, 55, 55, 55, 55],
            [56, 56, 56, 56, 56, 56],
            [57, 57, 57, 57, 57, 57, 57],
            [58, 58, 58, 58, 58, 58, 58, 58],
            [59, 59, 59, 59, 59, 59, 59, 59, 59],
            [],
            [61],
            [62, 62],
            [63, 63, 63],
            [64, 64, 64, 64],
            [65, 65, 65, 65, 65],
            [66, 66, 66, 66, 66, 66],
            [67, 67, 67, 67, 67, 67, 67],
            [68, 68, 68, 68, 68, 68, 68, 68],
            [69, 69, 69, 69, 69, 69, 69, 69, 69],
            [],
            [71],
            [72, 72],
            [73, 73, 73],
            [74, 74, 74, 74],
            [75, 75, 75, 75, 75],
            [76, 76, 76, 76, 76, 76],
            [77, 77, 77, 77, 77, 77, 77],
            [78, 78, 78, 78, 78, 78, 78, 78],
            [79, 79, 79, 79, 79, 79, 79, 79, 79],
            [],
            [81],
            [82, 82],
            [83, 83, 83],
            [84, 84, 84, 84],
            [85, 85, 85, 85, 85],
            [86, 86, 86, 86, 86, 86],
            [87, 87, 87, 87, 87, 87, 87],
            [88, 88, 88, 88, 88, 88, 88, 88],
            [89, 89, 89, 89, 89, 89, 89, 89, 89],
            [],
            [91],
            [92, 92],
            [93, 93, 93],
            [94, 94, 94, 94],
            [95, 95, 95, 95, 95],
            [96, 96, 96, 96, 96, 96],
            [97, 97, 97, 97, 97, 97, 97],
            [98, 98, 98, 98, 98, 98, 98, 98],
            [99, 99, 99, 99, 99, 99, 99, 99, 99],
        ]


def test_pandas_merge():
    pandas = pytest.importorskip("pandas", minversion="0.21.0")

    group = uproot.interpretation.library.Pandas().group
    name_interp_branch = [("a", None), ("b", None), ("c", None)]

    a = pandas.Series([1, 2, 3, 4, 5])
    b = pandas.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    c = pandas.Series([100, 200, 300, 400, 500])

    df = group({"a": a, "b": b, "c": c}, name_interp_branch, None)
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df.a.tolist() == [1, 2, 3, 4, 5]
    assert df.b.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert df.c.tolist() == [100, 200, 300, 400, 500]

    df = group({"a": a, "b": b, "c": c}, name_interp_branch, "outer")
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df.a.tolist() == [1, 2, 3, 4, 5]
    assert df.b.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert df.c.tolist() == [100, 200, 300, 400, 500]
