# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines versioned models for ``TTree``.

See :doc:`uproot.behaviors.TBranch` for definitions of ``TTree``-reading
functions.
"""

from __future__ import annotations

import struct

import numpy

import uproot
import uproot.behaviors.TTree
import uproot.models.TBranch

_ttree16_format1 = struct.Struct(">qqqqdiiiqqqqq")

_rawstreamer_TBranchRef_v1 = (
    None,
    b"@\x00\x01m\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01W\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTBranchRef\x00#`\xb3\xfd\x00\x00\x00\x01@\x00\x01-\xff\xff\xff\xffTObjArray\x00@\x00\x01\x1b\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TBranch\x11Branch descriptor\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x97\x8a\xac\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\r@\x00\x00\x89\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00j\x00\x02@\x00\x00d\x00\x04@\x00\x00/\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfRefTable\x18pointer to the TRefTable\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nTRefTable*\x00",
    "TBranchRef",
    1,
)
_rawstreamer_TRefTable_v3 = (
    None,
    b"@\x00\x03P\xff\xff\xff\xffTStreamerInfo\x00@\x00\x03:\x00\t@\x00\x00\x17\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\tTRefTable\x00\x8c\x89[\x85\x00\x00\x00\x03@\x00\x03\x11\xff\xff\xff\xffTObjArray\x00@\x00\x02\xff\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00\x82\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00g\x00\x02@\x00\x00a\x00\x04@\x00\x003\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fSize dummy for backward compatibility\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xb9\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00\x9a\x00\x02@\x00\x00\x94\x00\x04@\x00\x00_\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fParentsIarray of Parent objects  (eg TTree branch) holding the referenced objects\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nTObjArray*@\x00\x00\x88\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00i\x00\x02@\x00\x00c\x00\x04@\x00\x000\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fOwner\x1cObject owning this TRefTable\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08TObject*@\x00\x00\x9e\xff\xff\xff\xffTStreamerSTL\x00@\x00\x00\x89\x00\x03@\x00\x00{\x00\x04@\x00\x00B\x00\x01\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\rfProcessGUIDs'UUIDs of TProcessIDs used in fParentIDs\x00\x00\x01\xf4\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0evector<string>\x00\x00\x00\x01\x00\x00\x00=\x00",
    "TRefTable",
    3,
)
_rawstreamer_TTree_v20 = (
    None,
    b"@\x00\x14q\xff\xff\xff\xffTStreamerInfo\x00@\x00\x14[\x00\t@\x00\x00\x13\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x05TTree\x00rd\xe0\x7f\x00\x00\x00\x14@\x00\x146\xff\xff\xff\xffTObjArray\x00@\x00\x14$\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttLine\x0fLine attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x94\x07EI\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00y\xff\xff\xff\xffTStreamerBase\x00@\x00\x00c\x00\x03@\x00\x00Y\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttFill\x14Fill area attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9*\x92\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00x\xff\xff\xff\xffTStreamerBase\x00@\x00\x00b\x00\x03@\x00\x00X\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nTAttMarker\x11Marker attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x1d\x8b\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00{\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fEntries\x11Number of entries\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xa3\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x88\x00\x02@\x00\x00\x82\x00\x04@\x00\x00O\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfTotBytes8Total number of bytes in all branches before compression\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xa2\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x87\x00\x02@\x00\x00\x81\x00\x04@\x00\x00N\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfZipBytes7Total number of bytes in all branches after compression\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x002\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfSavedBytes\x19Number of autosaved bytes\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x8b\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00p\x00\x02@\x00\x00j\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\rfFlushedBytes\x1cNumber of auto-flushed bytes\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x89\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00n\x00\x02@\x00\x00h\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fWeight\"Tree weight (see TTree::SetWeight)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x89\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00n\x00\x02@\x00\x00h\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0efTimerInterval\x1eTimer interval in milliseconds\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00?\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfScanField'Number of runs before prompting in Scan\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x82\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00g\x00\x02@\x00\x00a\x00\x04@\x00\x003\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fUpdate\x1eUpdate frequency for EntryLoop\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xad\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x92\x00\x02@\x00\x00\x8c\x00\x04@\x00\x00^\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x16fDefaultEntryOffsetLen:Initial Length of fEntryOffset table in the basket buffers\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xb0\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x95\x00\x02@\x00\x00\x8f\x00\x04@\x00\x00a\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0efNClusterRangeENumber of Cluster range in addition to the one defined by 'AutoFlush'\x00\x00\x00\x06\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xa2\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x87\x00\x02@\x00\x00\x81\x00\x04@\x00\x00N\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfMaxEntries5Maximum number of entries in case of circular buffers\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x93\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00x\x00\x02@\x00\x00r\x00\x04@\x00\x00?\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\rfMaxEntryLoop$Maximum number of entries to process\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x9d\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x82\x00\x02@\x00\x00|\x00\x04@\x00\x00I\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0ffMaxVirtualSize,Maximum total size of buffers kept in memory\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xc1\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\xa6\x00\x02@\x00\x00\xa0\x00\x04@\x00\x00m\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfAutoSaveVAutosave tree when fAutoSave entries written or -fAutoSave (compressed) bytes produced\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xc6\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\xab\x00\x02@\x00\x00\xa5\x00\x04@\x00\x00r\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfAutoFlushZAuto-flush tree when fAutoFlush entries written or -fAutoFlush (compressed) bytes produced\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x99\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00~\x00\x02@\x00\x00x\x00\x04@\x00\x00E\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfEstimate.Number of entries to estimate histogram limits\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xbe\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\xa0\x00\x02@\x00\x00\x81\x00\x04@\x00\x00M\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x10fClusterRangeEnd/[fNClusterRange] Last entry of a cluster range.\x00\x00\x008\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tLong64_t*\x00\x00\x00\x14\x0efNClusterRange\x05TTree@\x00\x00\xd0\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\xb2\x00\x02@\x00\x00\x93\x00\x04@\x00\x00_\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfClusterSizeE[fNClusterRange] Number of entries in each cluster for a given range.\x00\x00\x008\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tLong64_t*\x00\x00\x00\x14\x0efNClusterRange\x05TTree@\x00\x00\xb3\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00\x98\x00\x02@\x00\x00\x92\x00\x04@\x00\x00V\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfIOFeatures=IO features to define for newly-written baskets and branches.\x00\x00\x00>\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11ROOT::TIOFeatures@\x00\x00y\xff\xff\xff\xffTStreamerObject\x00@\x00\x00a\x00\x02@\x00\x00[\x00\x04@\x00\x00'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBranches\x10List of Branches\x00\x00\x00=\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tTObjArray@\x00\x00\x92\xff\xff\xff\xffTStreamerObject\x00@\x00\x00z\x00\x02@\x00\x00t\x00\x04@\x00\x00@\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fLeaves+Direct pointers to individual branch leaves\x00\x00\x00=\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tTObjArray@\x00\x00\xa7\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00\x88\x00\x02@\x00\x00\x82\x00\x04@\x00\x00Q\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fAliases;List of aliases for expressions based on the tree branches.\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00\x80\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00e\x00\x02@\x00\x00_\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfIndexValues\x13Sorted index values\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayD@\x00\x00}\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00b\x00\x02@\x00\x00\\\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fIndex\x16Index of sorted values\x00\x00\x00>\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TArrayI@\x00\x00\x98\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00y\x00\x02@\x00\x00s\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfTreeIndex\"Pointer to the tree Index (if any)\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0eTVirtualIndex*@\x00\x00\x8e\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00o\x00\x02@\x00\x00i\x00\x04@\x00\x008\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fFriends\"pointer to list of friend elements\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00\xa6\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00\x87\x00\x02@\x00\x00\x81\x00\x04@\x00\x00P\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfUserInfo9pointer to a list of user objects associated to this Tree\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00\x9b\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00|\x00\x02@\x00\x00v\x00\x04@\x00\x00@\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfBranchRef(Branch supporting the TRefTable (if any)\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0bTBranchRef*\x00",
    "TTree",
    20,
)


class Model_TTree_v16(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 16.
    """

    behaviors = (uproot.behaviors.TTree.TTree,)

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
            file.class_named("TAttLine", 1).read(
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
            file.class_named("TAttFill", 1).read(
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
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree16_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree17_format1 = struct.Struct(">qqqqdiiiiqqqqq")


class Model_TTree_v17(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 17.
    """

    behaviors = (uproot.behaviors.TTree.TTree,)

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
            file.class_named("TAttLine", 1).read(
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
            file.class_named("TAttFill", 1).read(
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
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree17_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree18_format1 = struct.Struct(">qqqqqdiiiiqqqqqq")


class Model_TTree_v18(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 18.
    """

    behaviors = (uproot.behaviors.TTree.TTree,)

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
            file.class_named("TAttLine", 1).read(
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
            file.class_named("TAttFill", 1).read(
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
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree18_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree19_format1 = struct.Struct(">qqqqqdiiiiIqqqqqq")
_ttree19_dtype1 = numpy.dtype(">i8")
_ttree19_dtype2 = numpy.dtype(">i8")


class Model_TTree_v19(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 19.
    """

    behaviors = (uproot.behaviors.TTree.TTree,)

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
            file.class_named("TAttLine", 1).read(
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
            file.class_named("TAttFill", 1).read(
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
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fNClusterRange"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree19_format1, context)
        tmp = _ttree19_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterRangeEnd"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        tmp = _ttree19_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterSize"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fNClusterRange",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fClusterRangeEnd",
            "fClusterSize",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree20_format1 = struct.Struct(">qqqqqdiiiiIqqqqqq")
_ttree20_dtype1 = numpy.dtype(">i8")
_ttree20_dtype2 = numpy.dtype(">i8")


class Model_TTree_v20(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 20.
    """

    behaviors = (uproot.behaviors.TTree.TTree,)

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
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fNClusterRange"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree20_format1, context)
        tmp = _ttree20_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterRangeEnd"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        tmp = _ttree20_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterSize"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        self._members["fIOFeatures"] = file.class_named("ROOT::TIOFeatures").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fNClusterRange",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fClusterRangeEnd",
            "fClusterSize",
            "fIOFeatures",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 2),
        ("TAttFill", 2),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None

    class_rawstreamers = (
        _rawstreamer_TRefTable_v3,
        uproot.models.TBranch._rawstreamer_TBranch_v13,
        _rawstreamer_TBranchRef_v1,
        uproot.models.TH._rawstreamer_TList_v5,
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TObjArray._rawstreamer_TObjArray_v3,
        uproot.models.TBranch._rawstreamer_ROOT_3a3a_TIOFeatures_v1,
        uproot.models.TH._rawstreamer_TAttMarker_v2,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TAttLine_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TTree_v20,
    )


class Model_TTree(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TTree``.
    """

    known_versions = {
        16: Model_TTree_v16,
        17: Model_TTree_v17,
        18: Model_TTree_v18,
        19: Model_TTree_v19,
        20: Model_TTree_v20,
    }


_tiofeatures_format1 = struct.Struct(">B")


class Model_ROOT_3a3a_TIOFeatures(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::TIOFeatures``.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        cursor.skip(4)
        self._members["fIOBits"] = cursor.field(chunk, _tiofeatures_format1, context)


uproot.classes["TTree"] = Model_TTree
uproot.classes["ROOT::TIOFeatures"] = Model_ROOT_3a3a_TIOFeatures

fEntriesStruct = struct.Struct(">q")


class Model_TTree_NumEntries(uproot.model.Model):
    """
    A helper :doc:`uproot.model.Model` for :doc:`uproot.num_entries`.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        cursor.skip_over(chunk, context)  # TNamed
        cursor.skip_over(chunk, context)  # TAttLine
        cursor.skip_over(chunk, context)  # TAttFill
        cursor.skip_over(chunk, context)  # TAttMarker
        self._members["fEntries"] = cursor.fields(chunk, fEntriesStruct, context)
        cursor.skip_after(self)

    @property
    def member_names(self):
        return ["fEntries"]

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]


def num_entries(paths):
    """
    Args:
        paths (str): The filesystem path or remote URL of
            the TTree to find the number of entries in. It must have a file path
            as well as an object path which points to the location of the TTree
            inside the file. If the file names have colons in them, you can also
            pass in a dictionary in the format of { file_path : object_path }.
            Other examples: ``"rel/file.root:ttree"``, ``"C:\\abs\\file.root:ttree"``,
            ``"http://where/what.root:ttree"``,
            ``"https://username:password@where/secure.root:ttree"``,
            ``"rel/file.root:tdirectory/ttree"``, iterables of the previous examples.

    Returns an iterator over the number of entries over each TTree in the input.
    This is a shortcut method and reads lesser data than normal file opening.
    """
    paths2 = uproot._util.regularize_files(paths, steps_allowed=False)

    for file_path, object_path in paths2:
        with uproot.open(
            {file_path: object_path}, custom_classes={"TTree": Model_TTree_NumEntries}
        ) as f:
            yield file_path, object_path, f.all_members["fEntries"][0]
