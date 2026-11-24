# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot

from_ROOT = {}

from_ROOT["TH1F"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TH1F",
  "fTitle" : "",
  "fCheckSum" : 3642409091,
  "fClassVersion" : 2,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TH1",
      "fTitle" : "1-Dim histogram base class",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 1063172259, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 7
    }, {
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TArrayF",
      "fTitle" : "Array of floats",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 1510733553, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 1
    }]
  }
}""")

from_ROOT["TH1"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TH1",
  "fTitle" : "",
  "fCheckSum" : 1063172259,
  "fClassVersion" : 7,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TNamed",
      "fTitle" : "The basis for a named object (name, title)",
      "fType" : 67,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -541636036, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 1
    }, {
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TAttLine",
      "fTitle" : "Line attributes",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -1811462839, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 2
    }, {
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TAttFill",
      "fTitle" : "Fill area attributes",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -2545006, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 2
    }, {
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TAttMarker",
      "fTitle" : "Marker attributes",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 689802220, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 2
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fNcells",
      "fTitle" : "number of bins(1D), cells (2D) +U\/Overflows",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerObject",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fXaxis",
      "fTitle" : "X axis descriptor",
      "fType" : 61,
      "fSize" : 216,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TAxis"
    }, {
      "_typename" : "TStreamerObject",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fYaxis",
      "fTitle" : "Y axis descriptor",
      "fType" : 61,
      "fSize" : 216,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TAxis"
    }, {
      "_typename" : "TStreamerObject",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fZaxis",
      "fTitle" : "Z axis descriptor",
      "fType" : 61,
      "fSize" : 216,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TAxis"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBarOffset",
      "fTitle" : "(1000*offset) for bar charts or legos",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBarWidth",
      "fTitle" : "(1000*width) for bar charts or legos",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fEntries",
      "fTitle" : "Number of entries",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTsumw",
      "fTitle" : "Total Sum of weights",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTsumw2",
      "fTitle" : "Total Sum of squares of weights",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTsumwx",
      "fTitle" : "Total Sum of weight*X",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTsumwx2",
      "fTitle" : "Total Sum of weight*X*X",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fMaximum",
      "fTitle" : "Maximum value for plotting",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fMinimum",
      "fTitle" : "Minimum value for plotting",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fNormFactor",
      "fTitle" : "Normalization factor",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerObjectAny",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fContour",
      "fTitle" : "Array to display contour levels",
      "fType" : 62,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TArrayD"
    }, {
      "_typename" : "TStreamerObjectAny",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fSumw2",
      "fTitle" : "Array of sum of squares of weights",
      "fType" : 62,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TArrayD"
    }, {
      "_typename" : "TStreamerString",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fOption",
      "fTitle" : "histogram options",
      "fType" : 65,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TString"
    }, {
      "_typename" : "TStreamerObjectPointer",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fFunctions",
      "fTitle" : "->Pointer to list of functions (fits and user)",
      "fType" : 63,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TList*"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBufferSize",
      "fTitle" : "fBuffer size",
      "fType" : 6,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerBasicPointer",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBuffer",
      "fTitle" : "[fBufferSize] entry buffer",
      "fType" : 48,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double*",
      "fCountVersion" : 7,
      "fCountName" : "fBufferSize",
      "fCountClass" : "TH1"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBinStatErrOpt",
      "fTitle" : "option for bin statistical errors",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TH1::EBinErrorOpt"
    }]
  }
}""")

from_ROOT["TNamed"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TNamed",
  "fTitle" : "",
  "fCheckSum" : 3753331260,
  "fClassVersion" : 1,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TObject",
      "fTitle" : "Basic ROOT object",
      "fType" : 66,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -1877229523, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 1
    }, {
      "_typename" : "TStreamerString",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fName",
      "fTitle" : "object identifier",
      "fType" : 65,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TString"
    }, {
      "_typename" : "TStreamerString",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTitle",
      "fTitle" : "object title",
      "fType" : 65,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TString"
    }]
  }
}""")

from_ROOT["TObject"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TObject",
  "fTitle" : "",
  "fCheckSum" : 2417737773,
  "fClassVersion" : 1,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fUniqueID",
      "fTitle" : "object unique identifier",
      "fType" : 13,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "unsigned int"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBits",
      "fTitle" : "bit field status word",
      "fType" : 15,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "unsigned int"
    }]
  }
}""")

from_ROOT["TAttLine"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TAttLine",
  "fTitle" : "",
  "fCheckSum" : 2483504457,
  "fClassVersion" : 2,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLineColor",
      "fTitle" : "Line color",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLineStyle",
      "fTitle" : "Line style",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLineWidth",
      "fTitle" : "Line width",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }]
  }
}""")

from_ROOT["TAttFill"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TAttFill",
  "fTitle" : "",
  "fCheckSum" : 4292422290,
  "fClassVersion" : 2,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fFillColor",
      "fTitle" : "Fill area color",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fFillStyle",
      "fTitle" : "Fill area style",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }]
  }
}""")

from_ROOT["TAttMarker"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TAttMarker",
  "fTitle" : "",
  "fCheckSum" : 689802220,
  "fClassVersion" : 2,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fMarkerColor",
      "fTitle" : "Marker color",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fMarkerStyle",
      "fTitle" : "Marker style",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fMarkerSize",
      "fTitle" : "Marker size",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }]
  }
}""")

from_ROOT["TAxis"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TAxis",
  "fTitle" : "",
  "fCheckSum" : 1514761840,
  "fClassVersion" : 10,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TNamed",
      "fTitle" : "The basis for a named object (name, title)",
      "fType" : 67,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -541636036, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 1
    }, {
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TAttAxis",
      "fTitle" : "Axis attributes",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 1550843710, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 4
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fNbins",
      "fTitle" : "Number of bins",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fXmin",
      "fTitle" : "low edge of first bin",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fXmax",
      "fTitle" : "upper edge of last bin",
      "fType" : 8,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "double"
    }, {
      "_typename" : "TStreamerObjectAny",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fXbins",
      "fTitle" : "Bin edges array in X",
      "fType" : 62,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TArrayD"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fFirst",
      "fTitle" : "first bin to display",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLast",
      "fTitle" : "last bin to display",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fBits2",
      "fTitle" : "second bit status word",
      "fType" : 12,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "unsigned short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTimeDisplay",
      "fTitle" : "on\/off displaying time values instead of numerics",
      "fType" : 18,
      "fSize" : 1,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "bool"
    }, {
      "_typename" : "TStreamerString",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTimeFormat",
      "fTitle" : "Date&time format, ex: 09\/12\/99 12:34:00",
      "fType" : 65,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TString"
    }, {
      "_typename" : "TStreamerObjectPointer",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLabels",
      "fTitle" : "List of labels",
      "fType" : 64,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "THashList*"
    }, {
      "_typename" : "TStreamerObjectPointer",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fModLabs",
      "fTitle" : "List of modified labels",
      "fType" : 64,
      "fSize" : 8,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TList*"
    }]
  }
}""")

from_ROOT["TAttAxis"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TAttAxis",
  "fTitle" : "",
  "fCheckSum" : 1550843710,
  "fClassVersion" : 4,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fNdivisions",
      "fTitle" : "Number of divisions(10000*n3 + 100*n2 + n1)",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fAxisColor",
      "fTitle" : "Color of the line axis",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLabelColor",
      "fTitle" : "Color of labels",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLabelFont",
      "fTitle" : "Font for labels",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLabelOffset",
      "fTitle" : "Offset of labels",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fLabelSize",
      "fTitle" : "Size of labels",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTickLength",
      "fTitle" : "Length of tick marks",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTitleOffset",
      "fTitle" : "Offset of axis title",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTitleSize",
      "fTitle" : "Size of axis title",
      "fType" : 5,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "float"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTitleColor",
      "fTitle" : "Color of axis title",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fTitleFont",
      "fTitle" : "Font for axis title",
      "fType" : 2,
      "fSize" : 2,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "short"
    }]
  }
}""")

from_ROOT["THashList"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "THashList",
  "fTitle" : "",
  "fCheckSum" : 3430828481,
  "fClassVersion" : 0,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TList",
      "fTitle" : "Doubly linked list",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 1774568379, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 5
    }]
  }
}""")

from_ROOT["TList"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TList",
  "fTitle" : "",
  "fCheckSum" : 1774568379,
  "fClassVersion" : 5,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TSeqCollection",
      "fTitle" : "Sequenceable collection ABC",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -60015674, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 0
    }]
  }
}""")

from_ROOT["TSeqCollection"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TSeqCollection",
  "fTitle" : "",
  "fCheckSum" : 4234951622,
  "fClassVersion" : 0,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TCollection",
      "fTitle" : "Collection abstract base class",
      "fType" : 0,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 1474546588, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 3
    }]
  }
}""")

from_ROOT["TCollection"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TCollection",
  "fTitle" : "",
  "fCheckSum" : 1474546588,
  "fClassVersion" : 3,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : [{
      "_typename" : "TStreamerBase",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "TObject",
      "fTitle" : "Basic ROOT object",
      "fType" : 66,
      "fSize" : 0,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, -1877229523, 0, 0, 0],
      "fTypeName" : "BASE",
      "fBaseVersion" : 1
    }, {
      "_typename" : "TStreamerString",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fName",
      "fTitle" : "name of the collection",
      "fType" : 65,
      "fSize" : 24,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "TString"
    }, {
      "_typename" : "TStreamerBasicType",
      "fUniqueID" : 0,
      "fBits" : 0,
      "fName" : "fSize",
      "fTitle" : "number of elements in collection",
      "fType" : 3,
      "fSize" : 4,
      "fArrayLength" : 0,
      "fArrayDim" : 0,
      "fMaxIndex" : [0, 0, 0, 0, 0],
      "fTypeName" : "int"
    }]
  }
}""")

from_ROOT["TString"] = json.loads(r"""{
  "_typename" : "TStreamerInfo",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fName" : "TString",
  "fTitle" : "",
  "fCheckSum" : 95257,
  "fClassVersion" : 2,
  "fElements" : {
    "_typename" : "TObjArray",
    "name" : "TObjArray",
    "arr" : []
  }
}""")


def drop_fbits(x):
    if isinstance(x, dict):
        return {k: drop_fbits(v) for k, v in x.items() if k != "fBits"}
    elif isinstance(x, list):
        return list(drop_fbits(v) for v in x)
    else:
        return x


def test():
    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        streamers = f.file.streamers
        assert len(streamers) == 14

        assert drop_fbits(from_ROOT["TH1F"]) == drop_fbits(
            streamers["TH1F"][2].tojson()
        )
        assert drop_fbits(from_ROOT["TH1"]) == drop_fbits(streamers["TH1"][7].tojson())
        assert drop_fbits(from_ROOT["TNamed"]) == drop_fbits(
            streamers["TNamed"][1].tojson()
        )
        assert drop_fbits(from_ROOT["TObject"]) == drop_fbits(
            streamers["TObject"][1].tojson()
        )
        assert drop_fbits(from_ROOT["TAttLine"]) == drop_fbits(
            streamers["TAttLine"][2].tojson()
        )
        assert drop_fbits(from_ROOT["TAttFill"]) == drop_fbits(
            streamers["TAttFill"][2].tojson()
        )
        assert drop_fbits(from_ROOT["TAttMarker"]) == drop_fbits(
            streamers["TAttMarker"][2].tojson()
        )
        assert drop_fbits(from_ROOT["TAxis"]) == drop_fbits(
            streamers["TAxis"][10].tojson()
        )
        assert drop_fbits(from_ROOT["TAttAxis"]) == drop_fbits(
            streamers["TAttAxis"][4].tojson()
        )
        assert drop_fbits(from_ROOT["THashList"]) == drop_fbits(
            streamers["THashList"][0].tojson()
        )
        assert drop_fbits(from_ROOT["TList"]) == drop_fbits(
            streamers["TList"][5].tojson()
        )
        assert drop_fbits(from_ROOT["TSeqCollection"]) == drop_fbits(
            streamers["TSeqCollection"][0].tojson()
        )
        assert drop_fbits(from_ROOT["TCollection"]) == drop_fbits(
            streamers["TCollection"][3].tojson()
        )
        assert drop_fbits(from_ROOT["TString"]) == drop_fbits(
            streamers["TString"][2].tojson()
        )
