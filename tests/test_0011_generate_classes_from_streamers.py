# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot

from_ROOT = {}

from_ROOT["one"] = json.loads(r"""{
  "_typename" : "TH1F",
  "fUniqueID" : 0,
  "fBits" : 8,
  "fName" : "one",
  "fTitle" : "numero uno",
  "fLineColor" : 602,
  "fLineStyle" : 1,
  "fLineWidth" : 1,
  "fFillColor" : 0,
  "fFillStyle" : 1001,
  "fMarkerColor" : 1,
  "fMarkerStyle" : 1,
  "fMarkerSize" : 1.0,
  "fNcells" : 12,
  "fXaxis" : {
    "_typename" : "TAxis",
    "fUniqueID" : 0,
    "fBits" : 0,
    "fName" : "xaxis",
    "fTitle" : "",
    "fNdivisions" : 510,
    "fAxisColor" : 1,
    "fLabelColor" : 1,
    "fLabelFont" : 42,
    "fLabelOffset" : 0.004999999888241291,
    "fLabelSize" : 0.03500000014901161,
    "fTickLength" : 0.029999999329447746,
    "fTitleOffset" : 1.0,
    "fTitleSize" : 0.03500000014901161,
    "fTitleColor" : 1,
    "fTitleFont" : 42,
    "fNbins" : 10,
    "fXmin" : -3.0,
    "fXmax" : 3.0,
    "fXbins" : [],
    "fFirst" : 0,
    "fLast" : 0,
    "fBits2" : 0,
    "fTimeDisplay" : false,
    "fTimeFormat" : "",
    "fLabels" : null,
    "fModLabs" : null
  },
  "fYaxis" : {
    "_typename" : "TAxis",
    "fUniqueID" : 0,
    "fBits" : 0,
    "fName" : "yaxis",
    "fTitle" : "",
    "fNdivisions" : 510,
    "fAxisColor" : 1,
    "fLabelColor" : 1,
    "fLabelFont" : 42,
    "fLabelOffset" : 0.004999999888241291,
    "fLabelSize" : 0.03500000014901161,
    "fTickLength" : 0.029999999329447746,
    "fTitleOffset" : 1.0,
    "fTitleSize" : 0.03500000014901161,
    "fTitleColor" : 1,
    "fTitleFont" : 42,
    "fNbins" : 1,
    "fXmin" : 0.0,
    "fXmax" : 1.0,
    "fXbins" : [],
    "fFirst" : 0,
    "fLast" : 0,
    "fBits2" : 0,
    "fTimeDisplay" : false,
    "fTimeFormat" : "",
    "fLabels" : null,
    "fModLabs" : null
  },
  "fZaxis" : {
    "_typename" : "TAxis",
    "fUniqueID" : 0,
    "fBits" : 0,
    "fName" : "zaxis",
    "fTitle" : "",
    "fNdivisions" : 510,
    "fAxisColor" : 1,
    "fLabelColor" : 1,
    "fLabelFont" : 42,
    "fLabelOffset" : 0.004999999888241291,
    "fLabelSize" : 0.03500000014901161,
    "fTickLength" : 0.029999999329447746,
    "fTitleOffset" : 1.0,
    "fTitleSize" : 0.03500000014901161,
    "fTitleColor" : 1,
    "fTitleFont" : 42,
    "fNbins" : 1,
    "fXmin" : 0.0,
    "fXmax" : 1.0,
    "fXbins" : [],
    "fFirst" : 0,
    "fLast" : 0,
    "fBits2" : 0,
    "fTimeDisplay" : false,
    "fTimeFormat" : "",
    "fLabels" : null,
    "fModLabs" : null
  },
  "fBarOffset" : 0,
  "fBarWidth" : 1000,
  "fEntries" : 10000.0,
  "fTsumw" : 10000.0,
  "fTsumw2" : 10000.0,
  "fTsumwx" : 81.87497264376279,
  "fTsumwx2" : 10388.152621259549,
  "fMaximum" : -1111.0,
  "fMinimum" : -1111.0,
  "fNormFactor" : 0.0,
  "fContour" : [],
  "fSumw2" : [],
  "fOption" : "",
  "fFunctions" : {
    "_typename" : "TList",
    "name" : "TList",
    "arr" : [],
    "opt" : []
  },
  "fBufferSize" : 0,
  "fBuffer" : [],
  "fBinStatErrOpt" : 0,
  "fStatOverflows" : 2,
  "fArray" : [0, 68, 285, 755, 1580, 2296, 2286, 1570, 795, 289, 76, 0]
}""")


def drop_fbits(x):
    if isinstance(x, dict):
        return {
            k: drop_fbits(v)
            for k, v in x.items()
            if k not in ("fBits", "fStatOverflows", "fN", "fArray")
        }
    elif isinstance(x, list):
        return list(drop_fbits(v) for v in x)
    else:
        return x


def test():
    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        assert f.file.class_named("TH1", 7).member_names == [
            "fNcells",
            "fXaxis",
            "fYaxis",
            "fZaxis",
            "fBarOffset",
            "fBarWidth",
            "fEntries",
            "fTsumw",
            "fTsumw2",
            "fTsumwx",
            "fTsumwx2",
            "fMaximum",
            "fMinimum",
            "fNormFactor",
            "fContour",
            "fSumw2",
            "fOption",
            "fFunctions",
            "fBufferSize",
            "fBuffer",
            "fBinStatErrOpt",
        ]

        assert drop_fbits(from_ROOT["one"]) == drop_fbits(f.get("one").tojson())
