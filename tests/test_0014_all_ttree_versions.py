# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys
from io import StringIO

import numpy
import pytest
import skhep_testdata

import uproot


@pytest.mark.parametrize(
    "version, tree_version, branch_version",
    [
        ("5.23.02", 16, 11),
        ("5.24.00", 16, 11),
        ("5.25.02", 17, 12),
        ("5.26.00", 18, 12),
        ("5.27.02", 18, 12),
        ("5.28.00", 18, 12),
        ("5.29.02", 18, 12),
        ("5.30.00", 19, 12),
        ("6.08.04", 19, 12),
        ("6.10.05", 19, 12),
        ("6.14.00", 20, 13),
        ("6.16.00", 20, 13),
        ("6.18.00", 20, 13),
        ("6.20.04", 20, 13),
    ],
)
def test_streamerless_read(version, tree_version, branch_version):
    with uproot.open(
        skhep_testdata.data_path(f"uproot-sample-{version}-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == tree_version
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == "sample"
        assert f["sample"].member("fName") == "sample"
        assert len(f["sample"].member("fBranches")) == 35
        for x in f["sample"].member("fBranches"):
            assert x.class_version == branch_version
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        assert len(f["sample"].member("fLeaves")) == 35
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None


def test_streamerless_read_vector_vector_double():
    with uproot.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root")) as f:
        assert f["t"].class_version == 19
        assert f.file._streamers is None


def test_list_streamers():
    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        assert f.file.streamer_dependencies("TNamed") == [
            ("TString", 2),
            ("TObject", 1),
            ("TNamed", 1),
        ]

        output = StringIO()
        f.file.show_streamers("TNamed", stream=output)
        assert output.getvalue() == """TString (v2)

TObject (v1)
    fUniqueID: unsigned int (TStreamerBasicType)
    fBits: unsigned int (TStreamerBasicType)

TNamed (v1): TObject (v1)
    fName: TString (TStreamerString)
    fTitle: TString (TStreamerString)
"""
