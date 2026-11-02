# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines versioned models for ``TLeaf`` and its subclasses.
"""

from __future__ import annotations

import struct

import numpy

import uproot
import uproot._util
import uproot.behaviors.TAxis
import uproot.behaviors.TH1
import uproot.behaviors.TH2
import uproot.behaviors.TH3
import uproot.behaviors.TProfile
import uproot.behaviors.TProfile2D
import uproot.behaviors.TProfile3D
import uproot.deserialization
import uproot.model
import uproot.serialization

_rawstreamer_TCollection_v3 = (
    None,
    b"@\x00\x01\xe4\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xce\x00\t@\x00\x00\x19\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x0bTCollection\x00W\xe3\xcb\x9c\x00\x00\x00\x03@\x00\x01\xa3\xff\xff\xff\xffTObjArray\x00@\x00\x01\x91\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00y\xff\xff\xff\xffTStreamerString\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fName\x16name of the collection\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString@\x00\x00\x82\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00g\x00\x02@\x00\x00a\x00\x04@\x00\x003\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fSize number of elements in collection\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int\x00",
    "TCollection",
    3,
)
_rawstreamer_TSeqCollection_v0 = (
    None,
    b"@\x00\x00\xf5\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00\xdf\x00\t@\x00\x00\x1c\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x0eTSeqCollection\x00\xfcl;\xc6\x00\x00\x00\x00@\x00\x00\xb1\xff\xff\xff\xffTObjArray\x00@\x00\x00\x9f\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00@\x00\x00\x86\xff\xff\xff\xffTStreamerBase\x00@\x00\x00p\x00\x03@\x00\x00f\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bTCollection\x1eCollection abstract base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00W\xe3\xcb\x9c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x03\x00",
    "TSeqCollection",
    0,
)
_rawstreamer_TList_v5 = (
    None,
    b"@\x00\x00\xec\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00\xd6\x00\t@\x00\x00\x13\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x05TList\x00i\xc5\xc3\xbb\x00\x00\x00\x05@\x00\x00\xb1\xff\xff\xff\xffTObjArray\x00@\x00\x00\x9f\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00@\x00\x00\x86\xff\xff\xff\xffTStreamerBase\x00@\x00\x00p\x00\x03@\x00\x00f\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0eTSeqCollection\x1bSequenceable collection ABC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfcl;\xc6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x00\x00",
    "TList",
    5,
)
_rawstreamer_THashList_v0 = (
    None,
    b"@\x00\x00\xde\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00\xc8\x00\t@\x00\x00\x17\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\tTHashList\x00\xcc~I\xc1\x00\x00\x00\x00@\x00\x00\x9f\xff\xff\xff\xffTObjArray\x00@\x00\x00\x8d\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TList\x12Doubly linked list\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00i\xc5\xc3\xbb\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05\x00",
    "THashList",
    0,
)
_rawstreamer_TAttAxis_v4 = (
    None,
    b"@\x00\x05\xf7\xff\xff\xff\xffTStreamerInfo\x00@\x00\x05\xe1\x00\t@\x00\x00\x16\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x08TAttAxis\x00\\o\xff>\x00\x00\x00\x04@\x00\x05\xb9\xff\xff\xff\xffTObjArray\x00@\x00\x05\xa7\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00@\x00\x00\x93\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00x\x00\x02@\x00\x00r\x00\x04@\x00\x00D\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfNdivisions+Number of divisions(10000*n3 + 100*n2 + n1)\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00.\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfAxisColor\x16Color of the line axis\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00y\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00^\x00\x02@\x00\x00X\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfLabelColor\x0fColor of labels\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00x\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00]\x00\x02@\x00\x00W\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLabelFont\x0fFont for labels\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00{\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfLabelOffset\x10Offset of labels\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00w\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLabelSize\x0eSize of labels\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfTickLength\x14Length of tick marks\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00.\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfTitleOffset\x14Offset of axis title\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00{\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfTitleSize\x12Size of axis title\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00}\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00b\x00\x02@\x00\x00\\\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfTitleColor\x13Color of axis title\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfTitleFont\x13Font for axis title\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short\x00",
    "TAttAxis",
    4,
)
_rawstreamer_TAxis_v10 = (
    None,
    b"@\x00\x07\x13\xff\xff\xff\xffTStreamerInfo\x00@\x00\x06\xfd\x00\t@\x00\x00\x13\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x05TAxis\x00ZInp\x00\x00\x00\n@\x00\x06\xd8\xff\xff\xff\xffTObjArray\x00@\x00\x06\xc6\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttAxis\x0fAxis attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\\o\xff>\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x04@\x00\x00q\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00V\x00\x02@\x00\x00P\x00\x04@\x00\x00\"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fNbins\x0eNumber of bins\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00z\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00_\x00\x02@\x00\x00Y\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fXmin\x15low edge of first bin\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00{\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fXmax\x16upper edge of last bin\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00{\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fXbins\x14Bin edges array in X\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00w\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fFirst\x14first bin to display\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00u\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00Z\x00\x02@\x00\x00T\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fLast\x13last bin to display\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x84\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00i\x00\x02@\x00\x00c\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fBits2\x16second bit status word\x00\x00\x00\x0c\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0eunsigned short@\x00\x00\x9b\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x80\x00\x02@\x00\x00z\x00\x04@\x00\x00K\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfTimeDisplay1on/off displaying time values instead of numerics\x00\x00\x00\x12\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04bool@\x00\x00\x90\xff\xff\xff\xffTStreamerString\x00@\x00\x00x\x00\x02@\x00\x00r\x00\x04@\x00\x00@\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfTimeFormat'Date&time format, ex: 09/12/99 12:34:00\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString@\x00\x00}\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00^\x00\x02@\x00\x00X\x00\x04@\x00\x00#\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fLabels\x0eList of labels\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nTHashList*@\x00\x00\x83\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fModLabs\x17List of modified labels\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*\x00",
    "TAxis",
    10,
)
_rawstreamer_TAttMarker_v2 = (
    None,
    b"@\x00\x01\xd6\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xc0\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTAttMarker\x00)\x1d\x8b\xec\x00\x00\x00\x02@\x00\x01\x96\xff\xff\xff\xffTObjArray\x00@\x00\x01\x84\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00w\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfMarkerColor\x0cMarker color\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00w\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfMarkerStyle\x0cMarker style\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00u\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00Z\x00\x02@\x00\x00T\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfMarkerSize\x0bMarker size\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float\x00",
    "TAttMarker",
    2,
)
_rawstreamer_TAttFill_v2 = (
    None,
    b"@\x00\x01]\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01G\x00\t@\x00\x00\x16\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x08TAttFill\x00\xff\xd9*\x92\x00\x00\x00\x02@\x00\x01\x1f\xff\xff\xff\xffTObjArray\x00@\x00\x01\r\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00x\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00]\x00\x02@\x00\x00W\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfFillColor\x0fFill area color\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00x\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00]\x00\x02@\x00\x00W\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfFillStyle\x0fFill area style\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short\x00",
    "TAttFill",
    2,
)
_rawstreamer_TAttLine_v2 = (
    None,
    b'@\x00\x01\xca\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xb4\x00\t@\x00\x00\x16\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x08TAttLine\x00\x94\x07EI\x00\x00\x00\x02@\x00\x01\x8c\xff\xff\xff\xffTObjArray\x00@\x00\x01z\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00s\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00X\x00\x02@\x00\x00R\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLineColor\nLine color\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00s\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00X\x00\x02@\x00\x00R\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLineStyle\nLine style\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00s\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00X\x00\x02@\x00\x00R\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLineWidth\nLine width\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short\x00',
    "TAttLine",
    2,
)
_rawstreamer_TString_v2 = (
    None,
    b"@\x00\x00d\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00N\x00\t@\x00\x00\x15\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x07TString\x00\x00\x01t\x19\x00\x00\x00\x02@\x00\x00'\xff\xff\xff\xffTObjArray\x00@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    "TString",
    2,
)
_rawstreamer_TObject_v1 = (
    None,
    b"@\x00\x01s\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01]\x00\t@\x00\x00\x15\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x07TObject\x00\x90\x1b\xc0-\x00\x00\x00\x01@\x00\x016\xff\xff\xff\xffTObjArray\x00@\x00\x01$\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00\x87\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00l\x00\x02@\x00\x00f\x00\x04@\x00\x00/\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfUniqueID\x18object unique identifier\x00\x00\x00\r\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0cunsigned int@\x00\x00\x80\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00e\x00\x02@\x00\x00_\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fBits\x15bit field status word\x00\x00\x00\x0f\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0cunsigned int\x00",
    "TObject",
    1,
)
_rawstreamer_TNamed_v1 = (
    None,
    b"@\x00\x01\xc8\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xb2\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TNamed\x00\xdf\xb7J<\x00\x00\x00\x01@\x00\x01\x8c\xff\xff\xff\xffTObjArray\x00@\x00\x01z\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerString\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fName\x11object identifier\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString@\x00\x00p\xff\xff\xff\xffTStreamerString\x00@\x00\x00X\x00\x02@\x00\x00R\x00\x04@\x00\x00 \x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fTitle\x0cobject title\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00",
    "TNamed",
    1,
)
_rawstreamer_TH1_v8 = (
    None,
    b"@\x00\x0e,\xff\xff\xff\xffTStreamerInfo\x00@\x00\x0e\x16\x00\t@\x00\x00\x11\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x03TH1\x00\x1c7@\xc4\x00\x00\x00\x08@\x00\r\xf3\xff\xff\xff\xffTObjArray\x00@\x00\r\xe1\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1a\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttLine\x0fLine attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x94\x07EI\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00y\xff\xff\xff\xffTStreamerBase\x00@\x00\x00c\x00\x03@\x00\x00Y\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttFill\x14Fill area attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9*\x92\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00x\xff\xff\xff\xffTStreamerBase\x00@\x00\x00b\x00\x03@\x00\x00X\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nTAttMarker\x11Marker attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x1d\x8b\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00@\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fNcells+number of bins(1D), cells (2D) +U/Overflows\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00s\xff\xff\xff\xffTStreamerObject\x00@\x00\x00[\x00\x02@\x00\x00U\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fXaxis\x11X axis descriptor\x00\x00\x00=\x00\x00\x00\xd8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05TAxis@\x00\x00s\xff\xff\xff\xffTStreamerObject\x00@\x00\x00[\x00\x02@\x00\x00U\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fYaxis\x11Y axis descriptor\x00\x00\x00=\x00\x00\x00\xd8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05TAxis@\x00\x00s\xff\xff\xff\xffTStreamerObject\x00@\x00\x00[\x00\x02@\x00\x00U\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fZaxis\x11Z axis descriptor\x00\x00\x00=\x00\x00\x00\xd8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05TAxis@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00=\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfBarOffset%(1000*offset) for bar charts or legos\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00\x8c\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00q\x00\x02@\x00\x00k\x00\x04@\x00\x00;\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBarWidth$(1000*width) for bar charts or legos\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00y\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00^\x00\x02@\x00\x00X\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fEntries\x11Number of entries\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00z\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00_\x00\x02@\x00\x00Y\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fTsumw\x14Total Sum of weights\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x004\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumw2\x1fTotal Sum of squares of weights\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwx\x15Total Sum of weight*X\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwx2\x17Total Sum of weight*X*X\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x82\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00g\x00\x02@\x00\x00a\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum\x1aMaximum value for plotting\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x82\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00g\x00\x02@\x00\x00a\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum\x1aMinimum value for plotting\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfNormFactor\x14Normalization factor\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x88\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00m\x00\x02@\x00\x00g\x00\x04@\x00\x005\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fContour\x1fArray to display contour levels\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00\x89\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00n\x00\x02@\x00\x00h\x00\x04@\x00\x006\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fSumw2\"Array of sum of squares of weights\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00v\xff\xff\xff\xffTStreamerString\x00@\x00\x00^\x00\x02@\x00\x00X\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fOption\x11histogram options\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString@\x00\x00\x9c\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00}\x00\x02@\x00\x00w\x00\x04@\x00\x00F\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfFunctions.->Pointer to list of functions (fits and user)\x00\x00\x00?\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00t\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00Y\x00\x02@\x00\x00S\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBufferSize\x0cfBuffer size\x00\x00\x00\x06\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x99\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00{\x00\x02@\x00\x00a\x00\x04@\x00\x00/\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fBuffer\x1a[fBufferSize] entry buffer\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x08\x0bfBufferSize\x03TH1@\x00\x00\x9a\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x7f\x00\x02@\x00\x00y\x00\x04@\x00\x00=\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0efBinStatErrOpt!option for bin statistical errors\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11TH1::EBinErrorOpt@\x00\x00\xaf\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x94\x00\x02@\x00\x00\x8e\x00\x04@\x00\x00P\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0efStatOverflows4per object flag to use under/overflows in statistics\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13TH1::EStatOverflows\x00",
    "TH1",
    8,
)
_rawstreamer_TH2_v5 = (
    None,
    b"@\x00\x02\xe0\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\xca\x00\t@\x00\x00\x11\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x03TH2\x00\x01\x824\x7f\x00\x00\x00\x05@\x00\x02\xa7\xff\xff\xff\xffTObjArray\x00@\x00\x02\x95\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00x\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00]\x00\x02@\x00\x00W\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfScalefactor\x0cScale factor\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwy\x15Total Sum of weight*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwy2\x17Total Sum of weight*Y*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwxy\x17Total Sum of weight*X*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double\x00",
    "TH2",
    5,
)
_rawstreamer_TAtt3D_v1 = (
    None,
    b"@\x00\x00c\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00M\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TAtt3D\x00\x00\x00uz\x00\x00\x00\x01@\x00\x00'\xff\xff\xff\xffTObjArray\x00@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    "TAtt3D",
    1,
)
_rawstreamer_TH3_v6 = (
    None,
    b"@\x00\x04\xe1\xff\xff\xff\xffTStreamerInfo\x00@\x00\x04\xcb\x00\t@\x00\x00\x11\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x03TH3\x00B\xd2D_\x00\x00\x00\x06@\x00\x04\xa8\xff\xff\xff\xffTObjArray\x00@\x00\x04\x96\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00p\xff\xff\xff\xffTStreamerBase\x00@\x00\x00Z\x00\x03@\x00\x00P\x00\x04@\x00\x00!\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TAtt3D\r3D attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00uz\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwy\x15Total Sum of weight*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwy2\x17Total Sum of weight*Y*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwxy\x17Total Sum of weight*X*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwz\x15Total Sum of weight*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwz2\x17Total Sum of weight*Z*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwxz\x17Total Sum of weight*X*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwyz\x17Total Sum of weight*Y*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double\x00",
    "TH3",
    6,
)
_rawstreamer_TH1F_v3 = (
    None,
    b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH1F\x00\xe2\x93\x96D\x00\x00\x00\x03@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayF\x0fArray of floats\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\x0b\xf6\xf1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
    "TH1F",
    3,
)
_rawstreamer_TH1D_v3 = (
    None,
    b"@\x00\x01W\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01A\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH1D\x00\xf9\xb1V\x9f\x00\x00\x00\x03@\x00\x01\x1d\xff\xff\xff\xffTObjArray\x00@\x00\x01\x0b\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayD\x10Array of doubles\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00q9\xef4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
    "TH1D",
    3,
)
_rawstreamer_TH2D_v4 = (
    None,
    b"@\x00\x01W\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01A\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH2D\x00\x7f\xba\x82\xf0\x00\x00\x00\x04@\x00\x01\x1d\xff\xff\xff\xffTObjArray\x00@\x00\x01\x0b\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH2\x1a2-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x824\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayD\x10Array of doubles\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00q9\xef4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
    "TH2D",
    4,
)
_rawstreamer_TH3D_v4 = (
    None,
    b"@\x00\x01W\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01A\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH3D\x00d\xb9\xff\x86\x00\x00\x00\x04@\x00\x01\x1d\xff\xff\xff\xffTObjArray\x00@\x00\x01\x0b\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH3\x1a3-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\xd2D_\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x06@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayD\x10Array of doubles\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00q9\xef4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
    "TH3D",
    4,
)


class Model_TAxis_v10(uproot.behaviors.TAxis.TAxis, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAxis`` version 10.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttAxis", 4).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fNbins"],
            self._members["fXmin"],
            self._members["fXmax"],
        ) = cursor.fields(chunk, self._format0, context)
        self._members["fXbins"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fFirst"],
            self._members["fLast"],
            self._members["fBits2"],
            self._members["fTimeDisplay"],
        ) = cursor.fields(chunk, self._format1, context)
        self._members["fTimeFormat"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLabels"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )
        self._members["fModLabs"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TNamed", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TAttAxis", 4).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 2:
            self._members["fNbins"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 3:
            self._members["fXmin"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 4:
            self._members["fXmax"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 5:
            self._members["fXbins"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 6:
            self._members["fFirst"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 7:
            self._members["fLast"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 8:
            self._members["fBits2"] = cursor.field(
                chunk, self._format_memberwise5, context
            )
        if member_index == 9:
            self._members["fTimeDisplay"] = cursor.field(
                chunk, self._format_memberwise6, context
            )
        if member_index == 10:
            self._members["fTimeFormat"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 11:
            self._members["fLabels"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self
            )
        if member_index == 12:
            self._members["fModLabs"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TNamed", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttAxis", 4)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(("fNbins", numpy.dtype(">i4")))
        members.append(("fXmin", numpy.dtype(">f8")))
        members.append(("fXmax", numpy.dtype(">f8")))
        members.append(
            (
                "fXbins",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fFirst", numpy.dtype(">i4")))
        members.append(("fLast", numpy.dtype(">i4")))
        members.append(("fBits2", numpy.dtype(">u2")))
        members.append(("fTimeDisplay", numpy.dtype(numpy.bool_)))
        members.append(
            (
                "fTimeFormat",
                file.class_named("TString", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerObjectPointer of type THashList* in member fLabels of class TAxis"
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TNamed", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttAxis", 4).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fNbins"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fXmin"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fXmax"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fXbins"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fFirst"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fLast"] = uproot._util.awkward_form(numpy.dtype(">i4"), file, context)
        contents["fBits2"] = uproot._util.awkward_form(
            numpy.dtype(">u2"), file, context
        )
        contents["fTimeDisplay"] = uproot._util.awkward_form(
            numpy.dtype(numpy.bool_), file, context
        )
        contents["fTimeFormat"] = file.class_named("TString", "max").awkward_form(
            file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAxis"},
        )

    _format0 = struct.Struct(">idd")
    _format1 = struct.Struct(">iiH?")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">i")
    _format_memberwise4 = struct.Struct(">i")
    _format_memberwise5 = struct.Struct(">H")
    _format_memberwise6 = struct.Struct(">?")
    base_names_versions = [("TNamed", 1), ("TAttAxis", 4)]
    member_names = [
        "fNbins",
        "fXmin",
        "fXmax",
        "fXbins",
        "fFirst",
        "fLast",
        "fBits2",
        "fTimeDisplay",
        "fTimeFormat",
        "fLabels",
        "fModLabs",
    ]
    class_flags = {"has_read_object_any": True}

    writable = True

    def _to_writable_postprocess(self, original):
        if "fModLabs" not in self._members:
            self._members["fModLabs"] = None

    def _serialize(self, out, header, name, tobject_flags):
        import uproot.serialization

        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)

        out.append(
            self._format0.pack(
                self._members["fNbins"],
                self._members["fXmin"],
                self._members["fXmax"],
            )
        )
        self._members["fXbins"]._serialize(out, False, None, tobject_flags)
        out.append(
            self._format1.pack(
                self._members["fFirst"],
                self._members["fLast"],
                self._members["fBits2"],
                self._members["fTimeDisplay"],
            )
        )
        self._members["fTimeFormat"]._serialize(out, False, None, tobject_flags)
        uproot.serialization._serialize_object_any(out, self._members["fLabels"], None)
        uproot.serialization._serialize_object_any(out, self._members["fModLabs"], None)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 10
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TAxis(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAxis``.
    """

    known_versions = {10: Model_TAxis_v10}


class Model_TH1_v8(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1`` version 8.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fNcells"] = cursor.field(chunk, self._format0, context)
        self._members["fXaxis"] = file.class_named("TAxis").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fYaxis"] = file.class_named("TAxis").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fZaxis"] = file.class_named("TAxis").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fBarOffset"],
            self._members["fBarWidth"],
            self._members["fEntries"],
            self._members["fTsumw"],
            self._members["fTsumw2"],
            self._members["fTsumwx"],
            self._members["fTsumwx2"],
            self._members["fMaximum"],
            self._members["fMinimum"],
            self._members["fNormFactor"],
        ) = cursor.fields(chunk, self._format1, context)
        self._members["fContour"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fSumw2"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fOption"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fFunctions"] = file.class_named("TList").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBufferSize"] = cursor.field(chunk, self._format2, context)
        if context.get("speedbump", True):
            self._speedbump1 = cursor.byte(chunk, context)
        else:
            self._speedbump1 = None
        tmp = self._dtype0
        self._members["fBuffer"] = cursor.array(
            chunk, self.member("fBufferSize"), tmp, context
        )
        (
            self._members["fBinStatErrOpt"],
            self._members["fStatOverflows"],
        ) = cursor.fields(chunk, self._format3, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TNamed", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TAttLine", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 2:
            self._bases.append(
                file.class_named("TAttFill", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 3:
            self._bases.append(
                file.class_named("TAttMarker", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 4:
            self._members["fNcells"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 5:
            self._members["fXaxis"] = file.class_named("TAxis").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 6:
            self._members["fYaxis"] = file.class_named("TAxis").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 7:
            self._members["fZaxis"] = file.class_named("TAxis").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 8:
            self._members["fBarOffset"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 9:
            self._members["fBarWidth"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 10:
            self._members["fEntries"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 11:
            self._members["fTsumw"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 12:
            self._members["fTsumw2"] = cursor.field(
                chunk, self._format_memberwise5, context
            )
        if member_index == 13:
            self._members["fTsumwx"] = cursor.field(
                chunk, self._format_memberwise6, context
            )
        if member_index == 14:
            self._members["fTsumwx2"] = cursor.field(
                chunk, self._format_memberwise7, context
            )
        if member_index == 15:
            self._members["fMaximum"] = cursor.field(
                chunk, self._format_memberwise8, context
            )
        if member_index == 16:
            self._members["fMinimum"] = cursor.field(
                chunk, self._format_memberwise9, context
            )
        if member_index == 17:
            self._members["fNormFactor"] = cursor.field(
                chunk, self._format_memberwise10, context
            )
        if member_index == 18:
            self._members["fContour"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 19:
            self._members["fSumw2"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 20:
            self._members["fOption"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 21:
            self._members["fFunctions"] = file.class_named("TList").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 22:
            self._members["fBufferSize"] = cursor.field(
                chunk, self._format_memberwise11, context
            )
        if member_index == 23:
            tmp = self._dtype0
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fBuffer"] = cursor.array(
                chunk, self.member("fBufferSize"), tmp, context
            )
        if member_index == 24:
            self._members["fBinStatErrOpt"] = cursor.field(
                chunk, self._format_memberwise12, context
            )
        if member_index == 25:
            self._members["fStatOverflows"] = cursor.field(
                chunk, self._format_memberwise13, context
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TNamed", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttLine", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttFill", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttMarker", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(("fNcells", numpy.dtype(">i4")))
        members.append(
            (
                "fXaxis",
                file.class_named("TAxis", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(
            (
                "fYaxis",
                file.class_named("TAxis", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(
            (
                "fZaxis",
                file.class_named("TAxis", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fBarOffset", numpy.dtype(">i2")))
        members.append(("fBarWidth", numpy.dtype(">i2")))
        members.append(("fEntries", numpy.dtype(">f8")))
        members.append(("fTsumw", numpy.dtype(">f8")))
        members.append(("fTsumw2", numpy.dtype(">f8")))
        members.append(("fTsumwx", numpy.dtype(">f8")))
        members.append(("fTsumwx2", numpy.dtype(">f8")))
        members.append(("fMaximum", numpy.dtype(">f8")))
        members.append(("fMinimum", numpy.dtype(">f8")))
        members.append(("fNormFactor", numpy.dtype(">f8")))
        members.append(
            (
                "fContour",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(
            (
                "fSumw2",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(
            (
                "fOption",
                file.class_named("TString", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(
            (
                "fFunctions",
                file.class_named("TList", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fBufferSize", numpy.dtype(">u4")))
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fBuffer of class TH1"
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import ListOffsetForm, RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TNamed", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttLine", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttFill", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttMarker", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fNcells"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fXaxis"] = file.class_named("TAxis", "max").awkward_form(
            file, context
        )
        contents["fYaxis"] = file.class_named("TAxis", "max").awkward_form(
            file, context
        )
        contents["fZaxis"] = file.class_named("TAxis", "max").awkward_form(
            file, context
        )
        contents["fBarOffset"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fBarWidth"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fEntries"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumw"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumw2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwx"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwx2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fMaximum"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fMinimum"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fNormFactor"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fContour"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fSumw2"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fOption"] = file.class_named("TString", "max").awkward_form(
            file, context
        )
        contents["fFunctions"] = file.class_named("TList", "max").awkward_form(
            file, context
        )
        contents["fBufferSize"] = uproot._util.awkward_form(
            numpy.dtype(">u4"), file, context
        )
        contents["fBuffer"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype0, file, context),
        )
        contents["fBinStatErrOpt"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fStatOverflows"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1"},
        )

    _format0 = struct.Struct(">i")
    _format1 = struct.Struct(">hhdddddddd")
    _format2 = struct.Struct(">I")
    _format3 = struct.Struct(">ii")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">h")
    _format_memberwise2 = struct.Struct(">h")
    _format_memberwise3 = struct.Struct(">d")
    _format_memberwise4 = struct.Struct(">d")
    _format_memberwise5 = struct.Struct(">d")
    _format_memberwise6 = struct.Struct(">d")
    _format_memberwise7 = struct.Struct(">d")
    _format_memberwise8 = struct.Struct(">d")
    _format_memberwise9 = struct.Struct(">d")
    _format_memberwise10 = struct.Struct(">d")
    _format_memberwise11 = struct.Struct(">I")
    _format_memberwise12 = struct.Struct(">i")
    _format_memberwise13 = struct.Struct(">i")
    _dtype0 = numpy.dtype(">f8")
    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 2),
        ("TAttFill", 2),
        ("TAttMarker", 2),
    ]
    member_names = [
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
        "fStatOverflows",
    ]
    class_flags = {}

    writable = True

    def _to_writable_postprocess(self, original):
        self._speedbump1 = getattr(original, "_speedbump1", b"\x00")
        if "fStatOverflows" not in self._members:
            self._members["fStatOverflows"] = 0

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags | uproot.const.kMustCleanup)

        out.append(self._format0.pack(self._members["fNcells"]))
        self._members["fXaxis"]._serialize(out, header, None, tobject_flags)
        self._members["fYaxis"]._serialize(out, header, None, tobject_flags)
        self._members["fZaxis"]._serialize(out, header, None, tobject_flags)
        out.append(
            self._format1.pack(
                self._members["fBarOffset"],
                self._members["fBarWidth"],
                self._members["fEntries"],
                self._members["fTsumw"],
                self._members["fTsumw2"],
                self._members["fTsumwx"],
                self._members["fTsumwx2"],
                self._members["fMaximum"],
                self._members["fMinimum"],
                self._members["fNormFactor"],
            )
        )
        self._members["fContour"]._serialize(out, False, None, tobject_flags)
        self._members["fSumw2"]._serialize(out, False, None, tobject_flags)
        self._members["fOption"]._serialize(out, False, None, tobject_flags)
        self._members["fFunctions"]._serialize(
            out,
            header,
            None,
            (
                tobject_flags
                | uproot.const.kIsOnHeap
                | uproot.const.kNotDeleted
                | (1 << 16)  # I don't know what this is
            ),
        )
        out.append(self._format2.pack(self._members["fBufferSize"]))
        if self._speedbump1 is not None:
            out.append(self._speedbump1)
        out.append(self._members["fBuffer"].tobytes())
        out.append(
            self._format3.pack(
                self._members["fBinStatErrOpt"],
                self._members["fStatOverflows"],
            )
        )

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 8
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1``.
    """

    known_versions = {8: Model_TH1_v8}


class Model_TH2_v5(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2`` version 5.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fScalefactor"],
            self._members["fTsumwy"],
            self._members["fTsumwy2"],
            self._members["fTsumwxy"],
        ) = cursor.fields(chunk, self._format0, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._members["fScalefactor"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 2:
            self._members["fTsumwy"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 3:
            self._members["fTsumwy2"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 4:
            self._members["fTsumwxy"] = cursor.field(
                chunk, self._format_memberwise3, context
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(("fScalefactor", numpy.dtype(">f8")))
        members.append(("fTsumwy", numpy.dtype(">f8")))
        members.append(("fTsumwy2", numpy.dtype(">f8")))
        members.append(("fTsumwxy", numpy.dtype(">f8")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fScalefactor"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwy"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwy2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwxy"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2"},
        )

    _format0 = struct.Struct(">dddd")
    _format_memberwise0 = struct.Struct(">d")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">d")
    base_names_versions = [("TH1", 8)]
    member_names = ["fScalefactor", "fTsumwy", "fTsumwy2", "fTsumwxy"]
    class_flags = {}

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)

        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fScalefactor"],
                self._members["fTsumwy"],
                self._members["fTsumwy2"],
                self._members["fTsumwxy"],
            )
        )

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 5
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2``.
    """

    known_versions = {5: Model_TH2_v5}


class Model_TH3_v6(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3`` version 6.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAtt3D", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fTsumwy"],
            self._members["fTsumwy2"],
            self._members["fTsumwxy"],
            self._members["fTsumwz"],
            self._members["fTsumwz2"],
            self._members["fTsumwxz"],
            self._members["fTsumwyz"],
        ) = cursor.fields(chunk, self._format0, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TAtt3D", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 2:
            self._members["fTsumwy"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 3:
            self._members["fTsumwy2"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 4:
            self._members["fTsumwxy"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 5:
            self._members["fTsumwz"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 6:
            self._members["fTsumwz2"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 7:
            self._members["fTsumwxz"] = cursor.field(
                chunk, self._format_memberwise5, context
            )
        if member_index == 8:
            self._members["fTsumwyz"] = cursor.field(
                chunk, self._format_memberwise6, context
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAtt3D", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(("fTsumwy", numpy.dtype(">f8")))
        members.append(("fTsumwy2", numpy.dtype(">f8")))
        members.append(("fTsumwxy", numpy.dtype(">f8")))
        members.append(("fTsumwz", numpy.dtype(">f8")))
        members.append(("fTsumwz2", numpy.dtype(">f8")))
        members.append(("fTsumwxz", numpy.dtype(">f8")))
        members.append(("fTsumwyz", numpy.dtype(">f8")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAtt3D", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fTsumwy"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwy2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwxy"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwz"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwz2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwxz"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwyz"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3"},
        )

    _format0 = struct.Struct(">ddddddd")
    _format_memberwise0 = struct.Struct(">d")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">d")
    _format_memberwise4 = struct.Struct(">d")
    _format_memberwise5 = struct.Struct(">d")
    _format_memberwise6 = struct.Struct(">d")
    base_names_versions = [("TH1", 8), ("TAtt3D", 1)]
    member_names = [
        "fTsumwy",
        "fTsumwy2",
        "fTsumwxy",
        "fTsumwz",
        "fTsumwz2",
        "fTsumwxz",
        "fTsumwyz",
    ]
    class_flags = {}

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fTsumwy"],
                self._members["fTsumwy2"],
                self._members["fTsumwxy"],
                self._members["fTsumwz"],
                self._members["fTsumwz2"],
                self._members["fTsumwxz"],
                self._members["fTsumwyz"],
            )
        )

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 6
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3``.
    """

    known_versions = {6: Model_TH3_v6}


class Model_TH1C_v3(uproot.behaviors.TH1.TH1, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1C`` version 3.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayC", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayC", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayC", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayC", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1C"},
        )

    base_names_versions = [("TH1", 8), ("TArrayC", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        (
            None,
            b"@\x00\x01U\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01?\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH1C\x006\xf6\xe4\xad\x00\x00\x00\x03@\x00\x01\x1b\xff\xff\xff\xffTObjArray\x00@\x00\x01\t\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00r\xff\xff\xff\xffTStreamerBase\x00@\x00\x00\\\x00\x03@\x00\x00R\x00\x04@\x00\x00#\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayC\x0eArray of chars\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xae\x87\x996\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH1C",
            3,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1C(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1C``.
    """

    known_versions = {3: Model_TH1C_v3}


class Model_TH1D_v3(uproot.behaviors.TH1.TH1, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1D`` version 3.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayD", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayD", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayD", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayD", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1D"},
        )

    base_names_versions = [("TH1", 8), ("TArrayD", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH1D_v3,
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1D``.
    """

    known_versions = {3: Model_TH1D_v3}


class Model_TH1F_v3(uproot.behaviors.TH1.TH1, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1F`` version 3.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayF", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayF", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayF", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayF", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1F"},
        )

    base_names_versions = [("TH1", 8), ("TArrayF", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH1F_v3,
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1F(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1F``.
    """

    known_versions = {3: Model_TH1F_v3}


class Model_TH1I_v3(uproot.behaviors.TH1.TH1, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1I`` version 3.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayI", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayI", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayI", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayI", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1I"},
        )

    base_names_versions = [("TH1", 8), ("TArrayI", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        (
            None,
            b'@\x00\x01T\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01>\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH1I\x00bud\xf6\x00\x00\x00\x03@\x00\x01\x1a\xff\xff\xff\xffTObjArray\x00@\x00\x01\x08\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00q\xff\xff\xff\xffTStreamerBase\x00@\x00\x00[\x00\x03@\x00\x00Q\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayI\rArray of ints\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd9\xd5q\xc7\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00',
            "TH1I",
            3,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1I(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1I``.
    """

    known_versions = {3: Model_TH1I_v3}


class Model_TH1S_v3(uproot.behaviors.TH1.TH1, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH1S`` version 3.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1", 8).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayS", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1", 8).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayS", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1", 8)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayS", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1", 8).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayS", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH1S"},
        )

    base_names_versions = [("TH1", 8), ("TArrayS", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        (
            None,
            b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH1S\x00\x8cM\x9d\xcb\x00\x00\x00\x03@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH1\x1a1-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c7@\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x08@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayS\x0fArray of shorts\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\\\x93\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH1S",
            3,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH1S(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH1S``.
    """

    known_versions = {3: Model_TH1S_v3}


class Model_TH2C_v4(uproot.behaviors.TH2.TH2, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2C`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2", 5).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayC", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2", 5).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayC", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2", 5)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayC", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2", 5).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayC", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2C"},
        )

    base_names_versions = [("TH2", 5), ("TArrayC", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        (
            None,
            b"@\x00\x01U\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01?\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH2C\x00\xbd\x00\x10\xfe\x00\x00\x00\x04@\x00\x01\x1b\xff\xff\xff\xffTObjArray\x00@\x00\x01\t\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH2\x1a2-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x824\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05@\x00\x00r\xff\xff\xff\xffTStreamerBase\x00@\x00\x00\\\x00\x03@\x00\x00R\x00\x04@\x00\x00#\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayC\x0eArray of chars\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xae\x87\x996\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH2C",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2C(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2C``.
    """

    known_versions = {4: Model_TH2C_v4}


class Model_TH2D_v4(uproot.behaviors.TH2.TH2, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2D`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2", 5).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayD", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2", 5).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayD", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2", 5)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayD", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2", 5).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayD", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2D"},
        )

    base_names_versions = [("TH2", 5), ("TArrayD", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        _rawstreamer_TH2D_v4,
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2D``.
    """

    known_versions = {4: Model_TH2D_v4}


class Model_TH2F_v4(uproot.behaviors.TH2.TH2, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2F`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2", 5).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayF", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2", 5).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayF", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2", 5)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayF", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2", 5).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayF", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2F"},
        )

    base_names_versions = [("TH2", 5), ("TArrayF", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        (
            None,
            b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH2F\x00h\x9c\xc2\x95\x00\x00\x00\x04@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH2\x1a2-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x824\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayF\x0fArray of floats\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\x0b\xf6\xf1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH2F",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2F(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2F``.
    """

    known_versions = {4: Model_TH2F_v4}


class Model_TH2I_v4(uproot.behaviors.TH2.TH2, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2I`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2", 5).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayI", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2", 5).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayI", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2", 5)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayI", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2", 5).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayI", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2I"},
        )

    base_names_versions = [("TH2", 5), ("TArrayI", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        (
            None,
            b'@\x00\x01T\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01>\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH2I\x00\xe8~\x91G\x00\x00\x00\x04@\x00\x01\x1a\xff\xff\xff\xffTObjArray\x00@\x00\x01\x08\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH2\x1a2-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x824\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05@\x00\x00q\xff\xff\xff\xffTStreamerBase\x00@\x00\x00[\x00\x03@\x00\x00Q\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayI\rArray of ints\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd9\xd5q\xc7\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00',
            "TH2I",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2I(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2I``.
    """

    known_versions = {4: Model_TH2I_v4}


class Model_TH2S_v4(uproot.behaviors.TH2.TH2, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH2S`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2", 5).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayS", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2", 5).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayS", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2", 5)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayS", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2", 5).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayS", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH2S"},
        )

    base_names_versions = [("TH2", 5), ("TArrayS", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        (
            None,
            b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH2S\x00\x12V\xca\x1c\x00\x00\x00\x04@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH2\x1a2-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x824\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x05@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayS\x0fArray of shorts\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\\\x93\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH2S",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH2S(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH2S``.
    """

    known_versions = {4: Model_TH2S_v4}


class Model_TH3C_v4(uproot.behaviors.TH3.TH3, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3C`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3", 6).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayC", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3", 6).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayC", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3", 6)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayC", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3", 6).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayC", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3C"},
        )

    base_names_versions = [("TH3", 6), ("TArrayC", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        (
            None,
            b"@\x00\x01U\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01?\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH3C\x00\xa1\xff\x8d\x94\x00\x00\x00\x04@\x00\x01\x1b\xff\xff\xff\xffTObjArray\x00@\x00\x01\t\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH3\x1a3-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\xd2D_\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x06@\x00\x00r\xff\xff\xff\xffTStreamerBase\x00@\x00\x00\\\x00\x03@\x00\x00R\x00\x04@\x00\x00#\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayC\x0eArray of chars\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xae\x87\x996\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH3C",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3C(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3C``.
    """

    known_versions = {4: Model_TH3C_v4}


class Model_TH3D_v4(uproot.behaviors.TH3.TH3, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3D`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3", 6).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayD", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3", 6).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayD", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3", 6)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayD", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3", 6).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayD", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3D"},
        )

    base_names_versions = [("TH3", 6), ("TArrayD", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        _rawstreamer_TH3D_v4,
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3D``.
    """

    known_versions = {4: Model_TH3D_v4}


class Model_TH3F_v4(uproot.behaviors.TH3.TH3, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3F`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3", 6).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayF", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3", 6).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayF", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3", 6)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayF", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3", 6).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayF", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3F"},
        )

    base_names_versions = [("TH3", 6), ("TArrayF", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        (
            None,
            b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH3F\x00M\x9c?+\x00\x00\x00\x04@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH3\x1a3-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\xd2D_\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x06@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayF\x0fArray of floats\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\x0b\xf6\xf1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH3F",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3F(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3F``.
    """

    known_versions = {4: Model_TH3F_v4}


class Model_TH3I_v4(uproot.behaviors.TH3.TH3, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3I`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3", 6).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayI", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3", 6).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayI", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3", 6)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayI", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3", 6).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayI", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3I"},
        )

    base_names_versions = [("TH3", 6), ("TArrayI", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        (
            None,
            b'@\x00\x01T\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01>\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH3I\x00\xcd~\r\xdd\x00\x00\x00\x04@\x00\x01\x1a\xff\xff\xff\xffTObjArray\x00@\x00\x01\x08\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH3\x1a3-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\xd2D_\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x06@\x00\x00q\xff\xff\xff\xffTStreamerBase\x00@\x00\x00[\x00\x03@\x00\x00Q\x00\x04@\x00\x00"\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayI\rArray of ints\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd9\xd5q\xc7\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00',
            "TH3I",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3I(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3I``.
    """

    known_versions = {4: Model_TH3I_v4}


class Model_TH3S_v4(uproot.behaviors.TH3.TH3, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TH3S`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3", 6).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TArrayS", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3", 6).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._bases.append(
                file.class_named("TArrayS", 1).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3", 6)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TArrayS", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3", 6).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TArrayS", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TH3S"},
        )

    base_names_versions = [("TH3", 6), ("TArrayS", 1)]
    member_names = []
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        (
            None,
            b"@\x00\x01V\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01@\x00\t@\x00\x00\x12\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x04TH3S\x00\xf7VF\xb2\x00\x00\x00\x04@\x00\x01\x1c\xff\xff\xff\xffTObjArray\x00@\x00\x01\n\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00z\xff\xff\xff\xffTStreamerBase\x00@\x00\x00d\x00\x03@\x00\x00Z\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03TH3\x1a3-Dim histogram base class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\xd2D_\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x06@\x00\x00s\xff\xff\xff\xffTStreamerBase\x00@\x00\x00]\x00\x03@\x00\x00S\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TArrayS\x0fArray of shorts\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\\\x93\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01\x00",
            "TH3S",
            4,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        self._bases[0]._serialize(out, True, name, tobject_flags)
        self._bases[1]._serialize(out, False, name, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TH3S(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TH3S``.
    """

    known_versions = {4: Model_TH3S_v4}


class Model_TProfile_v7(
    uproot.behaviors.TProfile.TProfile, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TProfile`` version 7.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH1D", 3).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fBinEntries"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fErrorMode"],
            self._members["fYmin"],
            self._members["fYmax"],
            self._members["fTsumwy"],
            self._members["fTsumwy2"],
        ) = cursor.fields(chunk, self._format0, context)
        self._members["fBinSumw2"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH1D", 3).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._members["fBinEntries"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 2:
            self._members["fErrorMode"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 3:
            self._members["fYmin"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 4:
            self._members["fYmax"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 5:
            self._members["fTsumwy"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 6:
            self._members["fTsumwy2"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 7:
            self._members["fBinSumw2"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH1D", 3)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(
            (
                "fBinEntries",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fErrorMode", numpy.dtype(">i4")))
        members.append(("fYmin", numpy.dtype(">f8")))
        members.append(("fYmax", numpy.dtype(">f8")))
        members.append(("fTsumwy", numpy.dtype(">f8")))
        members.append(("fTsumwy2", numpy.dtype(">f8")))
        members.append(
            (
                "fBinSumw2",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH1D", 3).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fBinEntries"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fErrorMode"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fYmin"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fYmax"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fTsumwy"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwy2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fBinSumw2"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TProfile"},
        )

    _format0 = struct.Struct(">idddd")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">d")
    _format_memberwise4 = struct.Struct(">d")
    base_names_versions = [("TH1D", 3)]
    member_names = [
        "fBinEntries",
        "fErrorMode",
        "fYmin",
        "fYmax",
        "fTsumwy",
        "fTsumwy2",
        "fBinSumw2",
    ]
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH1D_v3,
        (
            None,
            b"@\x00\x04\xa5\xff\xff\xff\xffTStreamerInfo\x00@\x00\x04\x8f\x00\t@\x00\x00\x16\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x08TProfile\x00K\xed\xeeT\x00\x00\x00\x07@\x00\x04g\xff\xff\xff\xffTObjArray\x00@\x00\x04U\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00@\x00\x00\x8a\xff\xff\xff\xffTStreamerBase\x00@\x00\x00t\x00\x03@\x00\x00j\x00\x04@\x00\x00;\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x04TH1D)1-Dim histograms (one double per channel)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf9\xb1V\x9f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x03@\x00\x00\x85\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00j\x00\x02@\x00\x00d\x00\x04@\x00\x002\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBinEntries\x19number of entries per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfErrorMode\x18Option to compute errors\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nEErrorType@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fYmin\x19Lower limit in Y (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fYmax\x19Upper limit in Y (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwy\x15Total Sum of weight*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwy2\x17Total Sum of weight*Y*Y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x94\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00y\x00\x02@\x00\x00s\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBinSumw2*Array of sum of squares of weights per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD\x00",
            "TProfile",
            7,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        self._members["fBinEntries"]._serialize(out, False, None, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fErrorMode"],
                self._members["fYmin"],
                self._members["fYmax"],
                self._members["fTsumwy"],
                self._members["fTsumwy2"],
            )
        )
        self._members["fBinSumw2"]._serialize(out, False, None, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 7
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TProfile(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TProfile``.
    """

    known_versions = {7: Model_TProfile_v7}


class Model_TProfile2D_v8(
    uproot.behaviors.TProfile2D.TProfile2D, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TProfile2D`` version 8.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH2D", 4).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fBinEntries"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fErrorMode"],
            self._members["fZmin"],
            self._members["fZmax"],
            self._members["fTsumwz"],
            self._members["fTsumwz2"],
        ) = cursor.fields(chunk, self._format0, context)
        self._members["fBinSumw2"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH2D", 4).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._members["fBinEntries"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 2:
            self._members["fErrorMode"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 3:
            self._members["fZmin"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 4:
            self._members["fZmax"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 5:
            self._members["fTsumwz"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 6:
            self._members["fTsumwz2"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 7:
            self._members["fBinSumw2"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH2D", 4)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(
            (
                "fBinEntries",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fErrorMode", numpy.dtype(">i4")))
        members.append(("fZmin", numpy.dtype(">f8")))
        members.append(("fZmax", numpy.dtype(">f8")))
        members.append(("fTsumwz", numpy.dtype(">f8")))
        members.append(("fTsumwz2", numpy.dtype(">f8")))
        members.append(
            (
                "fBinSumw2",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH2D", 4).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fBinEntries"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fErrorMode"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fZmin"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fZmax"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fTsumwz"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwz2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fBinSumw2"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TProfile2D"},
        )

    _format0 = struct.Struct(">idddd")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">d")
    _format_memberwise4 = struct.Struct(">d")
    base_names_versions = [("TH2D", 4)]
    member_names = [
        "fBinEntries",
        "fErrorMode",
        "fZmin",
        "fZmax",
        "fTsumwz",
        "fTsumwz2",
        "fBinSumw2",
    ]
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH2_v5,
        _rawstreamer_TH2D_v4,
        (
            None,
            b"@\x00\x04\xa7\xff\xff\xff\xffTStreamerInfo\x00@\x00\x04\x91\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTProfile2D\x006\xa1B\xac\x00\x00\x00\x08@\x00\x04g\xff\xff\xff\xffTObjArray\x00@\x00\x04U\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00@\x00\x00\x8a\xff\xff\xff\xffTStreamerBase\x00@\x00\x00t\x00\x03@\x00\x00j\x00\x04@\x00\x00;\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x04TH2D)2-Dim histograms (one double per channel)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\xba\x82\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x04@\x00\x00\x85\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00j\x00\x02@\x00\x00d\x00\x04@\x00\x002\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBinEntries\x19number of entries per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfErrorMode\x18Option to compute errors\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nEErrorType@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fZmin\x19Lower limit in Z (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fZmax\x19Upper limit in Z (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwz\x15Total Sum of weight*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwz2\x17Total Sum of weight*Z*Z\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x94\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00y\x00\x02@\x00\x00s\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBinSumw2*Array of sum of squares of weights per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD\x00",
            "TProfile2D",
            8,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)

        self._members["fBinEntries"]._serialize(out, False, None, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fErrorMode"],
                self._members["fZmin"],
                self._members["fZmax"],
                self._members["fTsumwz"],
                self._members["fTsumwz2"],
            )
        )
        self._members["fBinSumw2"]._serialize(out, False, None, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 8
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TProfile2D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TProfile2D``.
    """

    known_versions = {8: Model_TProfile2D_v8}


class Model_TProfile3D_v8(
    uproot.behaviors.TProfile3D.TProfile3D, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TProfile3D`` version 8.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TH3D", 4).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fBinEntries"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fErrorMode"],
            self._members["fTmin"],
            self._members["fTmax"],
            self._members["fTsumwt"],
            self._members["fTsumwt2"],
        ) = cursor.fields(chunk, self._format0, context)
        self._members["fBinSumw2"] = file.class_named("TArrayD").read(
            chunk, cursor, context, file, self._file, self.concrete
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TH3D", 4).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            self._members["fBinEntries"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        if member_index == 2:
            self._members["fErrorMode"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 3:
            self._members["fTmin"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 4:
            self._members["fTmax"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 5:
            self._members["fTsumwt"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 6:
            self._members["fTsumwt2"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 7:
            self._members["fBinSumw2"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.extend(
            file.class_named("TH3D", 4)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(
            (
                "fBinEntries",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        members.append(("fErrorMode", numpy.dtype(">i4")))
        members.append(("fTmin", numpy.dtype(">f8")))
        members.append(("fTmax", numpy.dtype(">f8")))
        members.append(("fTsumwt", numpy.dtype(">f8")))
        members.append(("fTsumwt2", numpy.dtype(">f8")))
        members.append(
            (
                "fBinSumw2",
                file.class_named("TArrayD", "max").strided_interpretation(
                    file, header, tobject_header, breadcrumbs
                ),
            )
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        tmp_awkward_form = file.class_named("TH3D", 4).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fBinEntries"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        contents["fErrorMode"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fTmin"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fTmax"] = uproot._util.awkward_form(numpy.dtype(">f8"), file, context)
        contents["fTsumwt"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fTsumwt2"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fBinSumw2"] = file.class_named("TArrayD", "max").awkward_form(
            file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TProfile3D"},
        )

    _format0 = struct.Struct(">idddd")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _format_memberwise3 = struct.Struct(">d")
    _format_memberwise4 = struct.Struct(">d")
    base_names_versions = [("TH3D", 4)]
    member_names = [
        "fBinEntries",
        "fErrorMode",
        "fTmin",
        "fTmax",
        "fTsumwt",
        "fTsumwt2",
        "fBinSumw2",
    ]
    class_flags = {}

    class_rawstreamers = (
        _rawstreamer_TAtt3D_v1,
        _rawstreamer_TCollection_v3,
        _rawstreamer_TSeqCollection_v0,
        _rawstreamer_TList_v5,
        _rawstreamer_THashList_v0,
        _rawstreamer_TAttAxis_v4,
        _rawstreamer_TAxis_v10,
        _rawstreamer_TAttMarker_v2,
        _rawstreamer_TAttFill_v2,
        _rawstreamer_TAttLine_v2,
        _rawstreamer_TString_v2,
        _rawstreamer_TObject_v1,
        _rawstreamer_TNamed_v1,
        _rawstreamer_TH1_v8,
        _rawstreamer_TH3_v6,
        _rawstreamer_TH3D_v4,
        (
            None,
            b"@\x00\x04\xa7\xff\xff\xff\xffTStreamerInfo\x00@\x00\x04\x91\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTProfile3D\x00\xf6\x0ch\x14\x00\x00\x00\x08@\x00\x04g\xff\xff\xff\xffTObjArray\x00@\x00\x04U\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00@\x00\x00\x8a\xff\xff\xff\xffTStreamerBase\x00@\x00\x00t\x00\x03@\x00\x00j\x00\x04@\x00\x00;\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x04TH3D)3-Dim histograms (one double per channel)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xb9\xff\x86\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x04@\x00\x00\x85\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00j\x00\x02@\x00\x00d\x00\x04@\x00\x002\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBinEntries\x19number of entries per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfErrorMode\x18Option to compute errors\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nEErrorType@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fTmin\x19Lower limit in T (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00~\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00c\x00\x02@\x00\x00]\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fTmax\x19Upper limit in T (if set)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00|\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fTsumwt\x15Total Sum of weight*T\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x7f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00d\x00\x02@\x00\x00^\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fTsumwt2\x17Total Sum of weight*T*T\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x94\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00y\x00\x02@\x00\x00s\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBinSumw2*Array of sum of squares of weights per bin\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD\x00",
            "TProfile3D",
            8,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)

        self._members["fBinEntries"]._serialize(out, False, None, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fErrorMode"],
                self._members["fTmin"],
                self._members["fTmax"],
                self._members["fTsumwt"],
                self._members["fTsumwt2"],
            )
        )
        self._members["fBinSumw2"]._serialize(out, False, None, tobject_flags)

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 8
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TProfile3D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TProfile3D``.
    """

    known_versions = {8: Model_TProfile3D_v8}


uproot.classes["TAxis"] = Model_TAxis
uproot.classes["TH1"] = Model_TH1
uproot.classes["TH2"] = Model_TH2
uproot.classes["TH3"] = Model_TH3
uproot.classes["TH1C"] = Model_TH1C
uproot.classes["TH1D"] = Model_TH1D
uproot.classes["TH1F"] = Model_TH1F
uproot.classes["TH1I"] = Model_TH1I
uproot.classes["TH1S"] = Model_TH1S
uproot.classes["TH2C"] = Model_TH2C
uproot.classes["TH2D"] = Model_TH2D
uproot.classes["TH2F"] = Model_TH2F
uproot.classes["TH2I"] = Model_TH2I
uproot.classes["TH2S"] = Model_TH2S
uproot.classes["TH3C"] = Model_TH3C
uproot.classes["TH3D"] = Model_TH3D
uproot.classes["TH3F"] = Model_TH3F
uproot.classes["TH3I"] = Model_TH3I
uproot.classes["TH3S"] = Model_TH3S
uproot.classes["TProfile"] = Model_TProfile
uproot.classes["TProfile2D"] = Model_TProfile2D
uproot.classes["TProfile3D"] = Model_TProfile3D
