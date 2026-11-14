# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines versioned models for ``TBranch`` and its subclasses.

See :doc:`uproot.behaviors.TBranch` for definitions of ``TTree``-reading
functions.
"""

from __future__ import annotations

import struct

import numpy

import uproot
import uproot.models.TH

_tbranch10_format1 = struct.Struct(">iiiiqiIiqqq")
_tbranch10_dtype1 = numpy.dtype(">i4")
_tbranch10_dtype2 = numpy.dtype(">i8")
_tbranch10_dtype3 = numpy.dtype(">i8")

_rawstreamer_ROOT_3a3a_TIOFeatures_v1 = (
    None,
    b"@\x00\x00\xe0\xff\xff\xff\xffTStreamerInfo\x00@\x00\x00\xca\x00\t@\x00\x00\x1f\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x11ROOT::TIOFeatures\x00\x1a\xa1/\x10\x00\x00\x00\x01@\x00\x00\x99\xff\xff\xff\xffTObjArray\x00@\x00\x00\x87\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00@\x00\x00n\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00S\x00\x02@\x00\x00M\x00\x04@\x00\x00\x15\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fIOBits\x00\x00\x00\x00\x0b\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\runsigned char\x00",
    "ROOT::TIOFeatures",
    1,
)
_rawstreamer_TBranch_v13 = (
    None,
    b'@\x00\rf\xff\xff\xff\xffTStreamerInfo\x00@\x00\rP\x00\t@\x00\x00\x15\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x07TBranch\x00\x10\x97\x8a\xac\x00\x00\x00\r@\x00\r)\xff\xff\xff\xffTObjArray\x00@\x00\r\x17\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00y\xff\xff\xff\xffTStreamerBase\x00@\x00\x00c\x00\x03@\x00\x00Y\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttFill\x14Fill area attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9*\x92\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x85\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00j\x00\x02@\x00\x00d\x00\x04@\x00\x006\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfCompress\x1fCompression level and algorithm\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x86\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00k\x00\x02@\x00\x00e\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBasketSize\x1eInitial Size of  Basket Buffer\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xa6\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x8b\x00\x02@\x00\x00\x85\x00\x04@\x00\x00W\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0ffEntryOffsetLen:Initial Length of fEntryOffset table in the basket buffers\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x83\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00h\x00\x02@\x00\x00b\x00\x04@\x00\x004\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfWriteBasket\x1aLast basket number written\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\xa3\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x88\x00\x02@\x00\x00\x82\x00\x04@\x00\x00O\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfEntryNumber5Current entry number (last one filled in this branch)\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x9c\xff\xff\xff\xffTStreamerObjectAny\x00@\x00\x00\x81\x00\x02@\x00\x00{\x00\x04@\x00\x00?\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfIOFeatures&IO features for newly-created baskets.\x00\x00\x00>\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11ROOT::TIOFeatures@\x00\x00y\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00^\x00\x02@\x00\x00X\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fOffset\x15Offset of this branch\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x88\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00m\x00\x02@\x00\x00g\x00\x04@\x00\x009\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfMaxBaskets Maximum number of Baskets so far\x00\x00\x00\x06\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00z\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00_\x00\x02@\x00\x00Y\x00\x04@\x00\x00+\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfSplitLevel\x12Branch split level\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00{\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00`\x00\x02@\x00\x00Z\x00\x04@\x00\x00\'\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fEntries\x11Number of entries\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x95\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00z\x00\x02@\x00\x00t\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfFirstEntry(Number of the first entry in this branch\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xa1\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x86\x00\x02@\x00\x00\x80\x00\x04@\x00\x00M\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfTotBytes6Total number of bytes in all leaves before compression\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\xa0\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x85\x00\x02@\x00\x00\x7f\x00\x04@\x00\x00L\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfZipBytes5Total number of bytes in all leaves after compression\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x8b\xff\xff\xff\xffTStreamerObject\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x009\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfBranches"-> List of Branches of this branch\x00\x00\x00=\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tTObjArray@\x00\x00\x87\xff\xff\xff\xffTStreamerObject\x00@\x00\x00o\x00\x02@\x00\x00i\x00\x04@\x00\x005\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fLeaves -> List of leaves of this branch\x00\x00\x00=\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tTObjArray@\x00\x00\x89\xff\xff\xff\xffTStreamerObject\x00@\x00\x00q\x00\x02@\x00\x00k\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fBaskets!-> List of baskets of this branch\x00\x00\x00=\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tTObjArray@\x00\x00\xac\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x8e\x00\x02@\x00\x00p\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfBasketBytes\'[fMaxBaskets] Length of baskets on file\x00\x00\x00+\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04int*\x00\x00\x00\r\x0bfMaxBaskets\x07TBranch@\x00\x00\xbb\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x9d\x00\x02@\x00\x00\x7f\x00\x04@\x00\x00K\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0cfBasketEntry1[fMaxBaskets] Table of first entry in each basket\x00\x00\x008\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tLong64_t*\x00\x00\x00\r\x0bfMaxBaskets\x07TBranch@\x00\x00\xb3\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x95\x00\x02@\x00\x00w\x00\x04@\x00\x00C\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBasketSeek*[fMaxBaskets] Addresses of baskets on file\x00\x00\x008\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tLong64_t*\x00\x00\x00\r\x0bfMaxBaskets\x07TBranch@\x00\x00\xb0\xff\xff\xff\xffTStreamerString\x00@\x00\x00\x98\x00\x02@\x00\x00\x92\x00\x04@\x00\x00`\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\tfFileNameIName of file where buffers are stored ("" if in same file as Tree header)\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00',
    "TBranch",
    13,
)


class Model_TBranch_v10(uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranch`` version 10.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

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
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch10_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._cursor_baskets = cursor.copy()
        if file.options["minimal_ttree_metadata"]:
            if not cursor.skip_over(chunk, context):
                file.class_named("TObjArray").read(
                    chunk, cursor, context, file, self._file, self.concrete
                )
        else:
            self._members["fBaskets"] = file.class_named("TObjArray").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        tmp = _tbranch10_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch10_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch10_dtype3
        if context.get("speedbump", True) and cursor.bytes(chunk, 1, context)[0] == 2:
            tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch11_format1 = struct.Struct(">iiiiqiIiqqqq")
_tbranch11_dtype1 = numpy.dtype(">i4")
_tbranch11_dtype2 = numpy.dtype(">i8")
_tbranch11_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v11(uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranch`` version 11.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

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
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch11_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._cursor_baskets = cursor.copy()
        if file.options["minimal_ttree_metadata"]:
            if not cursor.skip_over(chunk, context):
                file.class_named("TObjArray").read(
                    chunk, cursor, context, file, self._file, self.concrete
                )
        else:
            self._members["fBaskets"] = file.class_named("TObjArray").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        tmp = _tbranch11_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch11_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch11_dtype3
        if context.get("speedbump", True) and cursor.bytes(chunk, 1, context)[0] == 2:
            tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch12_format1 = struct.Struct(">iiiiqiIiqqqq")
_tbranch12_dtype1 = numpy.dtype(">i4")
_tbranch12_dtype2 = numpy.dtype(">i8")
_tbranch12_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v12(uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranch`` version 12.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

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
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch12_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._cursor_baskets = cursor.copy()
        if file.options["minimal_ttree_metadata"]:
            if not cursor.skip_over(chunk, context):
                file.class_named("TObjArray").read(
                    chunk, cursor, context, file, self._file, self.concrete
                )
        else:
            self._members["fBaskets"] = file.class_named("TObjArray").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        tmp = _tbranch12_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch12_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch12_dtype3
        if context.get("speedbump", True) and cursor.bytes(chunk, 1, context)[0] == 2:
            tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch13_format1 = struct.Struct(">iiiiq")
_tbranch13_format2 = struct.Struct(">iIiqqqq")
_tbranch13_dtype1 = numpy.dtype(">i4")
_tbranch13_dtype2 = numpy.dtype(">i8")
_tbranch13_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v13(uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranch`` version 13.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

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
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
        ) = cursor.fields(chunk, _tbranch13_format1, context)
        self._members["fIOFeatures"] = file.class_named("ROOT::TIOFeatures").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch13_format2, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._cursor_baskets = cursor.copy()
        if file.options["minimal_ttree_metadata"]:
            if not cursor.skip_over(chunk, context):
                file.class_named("TObjArray").read(
                    chunk, cursor, context, file, self._file, self.concrete
                )
        else:
            self._members["fBaskets"] = file.class_named("TObjArray").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
        tmp = _tbranch13_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch13_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch13_dtype3
        if context.get("speedbump", True) and cursor.bytes(chunk, 1, context)[0] == 2:
            tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = file.class_named("TString").read(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fIOFeatures",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 2)]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TObjArray._rawstreamer_TObjArray_v3,
        _rawstreamer_ROOT_3a3a_TIOFeatures_v1,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TBranch_v13,
    )


class Model_TBranch(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TBranch``.
    """

    known_versions = {
        10: Model_TBranch_v10,
        11: Model_TBranch_v11,
        12: Model_TBranch_v12,
        13: Model_TBranch_v13,
    }


_tbranchelement8_format1 = struct.Struct(">Iiiiii")


class Model_TBranchElement_v8(
    uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranchElement`` version 8.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TBranch", 10).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fParentName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fClonesName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement8_format1, context)
        self._members["fBranchCount"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBranchCount2"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [("TBranch", 10)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_tbranchelement9_format1 = struct.Struct(">Iiiiii")


class Model_TBranchElement_v9(
    uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranchElement`` version 9.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TBranch", 12).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fParentName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fClonesName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement9_format1, context)
        self._members["fBranchCount"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBranchCount2"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [("TBranch", 12)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_tbranchelement10_format1 = struct.Struct(">Ihiiii")


class Model_TBranchElement_v10(
    uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranchElement`` version 10.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TBranch", 12).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fParentName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fClonesName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement10_format1, context)
        self._members["fBranchCount"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBranchCount2"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [("TBranch", 12)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


class Model_TBranchElement(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TBranchElement``.
    """

    known_versions = {
        8: Model_TBranchElement_v8,
        9: Model_TBranchElement_v9,
        10: Model_TBranchElement_v10,
    }


class Model_TBranchObject_v1(
    uproot.behaviors.TBranch.TBranch, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TBranchObject`` version 1.
    """

    behaviors = (uproot.behaviors.TBranch.TBranch,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            file.class_named("TBranch", 13).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fClassName"] = file.class_named("TString").read(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [("TBranch", 13)]
    member_names = ["fClassName"]
    class_flags = {}


class Model_TBranchObject(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TBranchObject``.
    """

    known_versions = {
        1: Model_TBranchObject_v1,
    }


uproot.classes["TBranch"] = Model_TBranch
uproot.classes["TBranchElement"] = Model_TBranchElement
uproot.classes["TBranchObject"] = Model_TBranchObject
