# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot
from uproot.containers import AsMap, AsSet, AsString, AsVector
from uproot.interpretation.identify import parse_typename


def test_parse_typename():
    assert parse_typename("TTree") is uproot.classes["TTree"]
    assert parse_typename("string") == AsString(False)
    assert parse_typename("std::string") == AsString(False)
    assert parse_typename("std :: string") == AsString(False)
    assert parse_typename("char*") == AsString(False, length_bytes="4")
    assert parse_typename("char *") == AsString(False, length_bytes="4")
    assert parse_typename("TString") == AsString(False)
    assert parse_typename("vector<TTree>") == AsVector(True, uproot.classes["TTree"])
    assert parse_typename("vector<int>") == AsVector(True, ">i4")
    assert parse_typename("vector<bool>") == AsVector(True, "?")
    assert parse_typename("vector<string>") == AsVector(True, AsString(False))
    assert parse_typename("vector  <   string   >") == AsVector(True, AsString(False))
    assert parse_typename("std::vector<std::string>") == AsVector(True, AsString(False))
    assert parse_typename("vector<vector<int>>") == AsVector(
        True, AsVector(False, ">i4")
    )
    assert parse_typename("vector<vector<string>>") == AsVector(
        True, AsVector(False, AsString(False))
    )
    assert parse_typename("vector<vector<char*>>") == AsVector(
        True, AsVector(False, AsString(False, length_bytes="4"))
    )
    assert parse_typename("set<unsigned short>") == AsSet(True, ">u2")
    assert parse_typename("std::set<unsigned short>") == AsSet(True, ">u2")
    assert parse_typename("set<string>") == AsSet(True, AsString(False))
    assert parse_typename("set<vector<string>>") == AsSet(
        True, AsVector(False, AsString(False))
    )
    assert parse_typename("set<vector<string> >") == AsSet(
        True, AsVector(False, AsString(False))
    )
    assert parse_typename("map<int, double>") == AsMap(True, ">i4", ">f8")
    assert parse_typename("map<string, double>") == AsMap(True, AsString(True), ">f8")
    assert parse_typename("map<int, string>") == AsMap(True, ">i4", AsString(True))
    assert parse_typename("map<string, string>") == AsMap(
        True, AsString(True), AsString(True)
    )
    assert parse_typename("map<string,string>") == AsMap(
        True, AsString(True), AsString(True)
    )
    assert parse_typename("map<   string,string   >") == AsMap(
        True, AsString(True), AsString(True)
    )
    assert parse_typename("map<string,vector<int>>") == AsMap(
        True, AsString(True), AsVector(True, ">i4")
    )
    assert parse_typename("map<vector<int>, string>") == AsMap(
        True, AsVector(True, ">i4"), AsString(True)
    )
    assert parse_typename("map<vector<int>, set<float>>") == AsMap(
        True, AsVector(True, ">i4"), AsSet(True, ">f4")
    )
    assert parse_typename("map<vector<int>, set<set<float>>>") == AsMap(
        True, AsVector(True, ">i4"), AsSet(True, AsSet(False, ">f4"))
    )

    with pytest.raises(ValueError, match="invalid C\\+\\+ type name syntax at char 8"):
        parse_typename("string  <")

    with pytest.raises(ValueError, match="invalid C\\+\\+ type name syntax at char 9"):
        parse_typename("vector  <")

    with pytest.raises(ValueError, match="invalid C\\+\\+ type name syntax at char 11"):
        parse_typename("map<string<int>>")

    with pytest.raises(ValueError, match="invalid C\\+\\+ type name syntax at char 16"):
        parse_typename("map<string, int>>")


def test_strings_1():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as tree:
        result = tree["Beg"].array(library="np")
        assert result.tolist() == [f"beg-{i:03d}" for i in range(100)]

        result = tree["End"].array(library="np")
        assert result.tolist() == [f"end-{i:03d}" for i in range(100)]


def test_map_string_string_in_object():
    with uproot.open(skhep_testdata.data_path("uproot-issue431.root")) as f:
        head = f["Head"]
        assert head.member("map<string,string>") == {
            "DAQ": "394",
            "PDF": "4      58",
            "XSecFile": "",
            "can": "0 1027 888.4",
            "can_user": "0.00 1027.00  888.40",
            "coord_origin": "0 0 0",
            "cut_in": "0 0 0 0",
            "cut_nu": "100 1e+08 -1 1",
            "cut_primary": "0 0 0 0",
            "cut_seamuon": "0 0 0 0",
            "decay": "doesnt happen",
            "detector": "NOT",
            "drawing": "Volume",
            "end_event": "",
            "genhencut": "2000 0",
            "genvol": "0 1027 888.4 2.649e+09 100000",
            "kcut": "2",
            "livetime": "0 0",
            "model": "1       2       0       1      12",
            "muon_desc_file": "",
            "ngen": "0.1000E+06",
            "norma": "0 0",
            "nuflux": "0       3       0 0.500E+00 0.000E+00 0.100E+01 0.300E+01",
            "physics": "GENHEN 7.2-220514 181116 1138",
            "seed": "GENHEN 3  305765867         0         0",
            "simul": "JSirene 11012 11/17/18 07",
            "sourcemode": "diffuse",
            "spectrum": "-1.4",
            "start_run": "1",
            "target": "isoscalar",
            "usedetfile": "false",
            "xlat_user": "0.63297",
            "xparam": "OFF",
            "zed_user": "0.00 3450.00",
        }


def test_map_long_int_in_object():
    with uproot.open(skhep_testdata.data_path("uproot-issue283.root")) as f:
        map_long_int = f["config/detector"].member("ChannelIDMap")
        assert (map_long_int.keys().min(), map_long_int.keys().max()) == (
            46612627560,
            281410180683757,
        )
        assert (map_long_int.values().min(), map_long_int.values().max()) == (0, 5159)


def test_top_level_vectors():
    with uproot.open(skhep_testdata.data_path("uproot-issue38a.root"))[
        "ntupler/tree"
    ] as tree:
        assert [x.tolist() for x in tree["v_int16"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_int16"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_int32"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_int64"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_uint16"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_uint32"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_uint64"].array(library="np")] == [[1, 2, 3]]
        assert [x.tolist() for x in tree["v_bool"].array(library="np")] == [
            [False, True]
        ]
        assert [x.tolist() for x in tree["v_float"].array(library="np")] == [
            [999.0, -999.0]
        ]
        assert [x.tolist() for x in tree["v_double"].array(library="np")] == [
            [999.0, -999.0]
        ]


def test_strings_2():
    with uproot.open(skhep_testdata.data_path("uproot-issue31.root"))[
        "T/name"
    ] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["one", "two", "three", "four", "five"]


def test_strings_3():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/Str"
    ] as branch:
        result = branch.array(library="np")
        assert result.tolist() == [f"evt-{i:03d}" for i in range(100)]


def test_strings_4():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree/StdStr"
    ] as branch:
        result = branch.array(library="np")
        assert result.tolist() == [f"std-{i:03d}" for i in range(100)]
