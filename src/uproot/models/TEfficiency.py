# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versioned model for ``TEfficiency``.

``TEfficiency`` has a ``std::vector<std::pair<double,double>>`` member
(``fBeta_bin_params``) that ROOT serializes memberwise, and recent versions of
ROOT do not write a ``TStreamerInfo`` for ``std::pair<double,double>``. This
hand-written model reads (and writes) that member directly, so that
``TEfficiency`` objects can be read without relying on the file's streamers.
"""

from __future__ import annotations

import struct

import numpy

import uproot
import uproot._awkwardforth
import uproot.behaviors.TEfficiency
import uproot.const
import uproot.deserialization
import uproot.model
import uproot.models.TH
import uproot.serialization

_rawstreamer_TEfficiency_v2 = (
    None,
    b"@\x00\x082\xff\xff\xff\xffTStreamerInfo\x00@\x00\x08\x1c\x00\t@\x00\x00\x19\x00\x01\x00\x01\x00\x00\x00\x00\x03\x03\x00\x00\x0bTEfficiency\x00R\x93\x1a\xeb\x00\x00\x00\x02@\x00\x07\xf1\xff\xff\xff\xffTObjArray\x00@\x00\x07\xdf\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttLine\x0fLine attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x94\x07EI\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00y\xff\xff\xff\xffTStreamerBase\x00@\x00\x00c\x00\x03@\x00\x00Y\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttFill\x14Fill area attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9*\x92\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00x\xff\xff\xff\xffTStreamerBase\x00@\x00\x00b\x00\x03@\x00\x00X\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nTAttMarker\x11Marker attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x1d\x8b\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\xa5\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x8a\x00\x02@\x00\x00\x84\x00\x04@\x00\x00S\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfBeta_alpha:global parameter for prior beta distribution (default = 1)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\xa4\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x89\x00\x02@\x00\x00\x83\x00\x04@\x00\x00R\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfBeta_beta:global parameter for prior beta distribution (default = 1)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\xc2\xff\xff\xff\xffTStreamerSTL\x00@\x00\x00\xad\x00\x03@\x00\x00\x9f\x00\x04@\x00\x00X\x00\x01\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x10fBeta_bin_params:parameter for prior beta distribution different bin by bin\x00\x00\x01\xf4\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1cvector<pair<double,double> >\x00\x00\x00\x01\x00\x00\x00=@\x00\x00\x95\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00z\x00\x02@\x00\x00t\x00\x04@\x00\x00C\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfConfLevel+confidence level (default = 0.683, 1 sigma)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x8c\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00m\x00\x02@\x00\x00g\x00\x04@\x00\x006\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfFunctions\x1e->pointer to list of functions\x00\x00\x00?\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00\xa4\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00\x85\x00\x02@\x00\x00\x7f\x00\x04@\x00\x00P\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x10fPassedHistogram2histogram for events which passed certain criteria\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04TH1*@\x00\x00\xb5\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\x9a\x00\x02@\x00\x00\x94\x00\x04@\x00\x00Q\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x10fStatisticOption3defines how the confidence intervals are determined\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18TEfficiency::EStatOption@\x00\x00\x95\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00v\x00\x02@\x00\x00p\x00\x04@\x00\x00A\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0ffTotalHistogram$histogram for total number of events\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04TH1*@\x00\x00\x8a\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00o\x00\x02@\x00\x00i\x00\x04@\x00\x008\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fWeight#weight for all events (default = 1)\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double\x00",
    "TEfficiency",
    2,
)

_beta_bin_params_dtype = numpy.dtype(
    [("first", numpy.float64), ("second", numpy.float64)]
)

# ROOT's memberwise header for vector<pair<double,double>>: the collection's
# streamer version with kStreamedMemberWise set, then the version (0) and
# checksum of the "pair<double,double>" class.
_beta_bin_params_collection_version = 9
_pair_double_double_version_checksum = b"\x00\x00\x00\xd7\xbe\xd2"

_format_int32 = struct.Struct(">i")
_format_double = struct.Struct(">d")
_format_double2 = struct.Struct(">dd")
_format_short = struct.Struct(">h")

# TEfficiency::EStatusBits: kIsBayesian, kPosteriorMode, kShortestInterval,
# kUseBinPrior, kUseWeights
_tefficiency_bits_mask = (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17) | (1 << 18)


def _read_beta_bin_params(chunk, cursor, context, file, selffile, parent):
    start_cursor = cursor.copy()
    num_bytes, _, is_memberwise = uproot.deserialization.numbytes_version(
        chunk, cursor, context
    )

    if not is_memberwise:
        # objectwise: each pair<double,double> has its own byte count and
        # version (followed by a checksum if the version is 0)
        length = cursor.field(chunk, _format_int32, context)
        out = numpy.empty(length, dtype=_beta_bin_params_dtype)
        for i in range(length):
            _, version, _ = uproot.deserialization.numbytes_version(
                chunk, cursor, context
            )
            if version <= 0:
                cursor.skip(4)
            out[i] = cursor.fields(chunk, _format_double2, context)
        uproot.deserialization.numbytes_check(
            chunk,
            start_cursor,
            cursor,
            num_bytes,
            "vector<pair<double,double>>",
            context,
            file.file_path,
        )
        return out

    # version of the value class (pair<double,double>), followed by its checksum
    # if the version is 0 (as is the case for emulated/foreign classes)
    pair_version = cursor.field(chunk, _format_short, context)
    if pair_version <= 0:
        cursor.skip(4)

    length = cursor.field(chunk, _format_int32, context)
    first = cursor.array(chunk, length, numpy.dtype(">f8"), context)
    second = cursor.array(chunk, length, numpy.dtype(">f8"), context)

    out = numpy.empty(length, dtype=_beta_bin_params_dtype)
    out["first"] = first
    out["second"] = second

    uproot.deserialization.numbytes_check(
        chunk,
        start_cursor,
        cursor,
        num_bytes,
        "vector<pair<double,double>>",
        context,
        file.file_path,
    )
    return out


def _serialize_beta_bin_params(out, params):
    params = numpy.asarray(params)
    if params.dtype.names is None:
        params = numpy.asarray(params, dtype=numpy.float64).reshape(-1, 2)
        first, second = params[:, 0], params[:, 1]
    else:
        first, second = params["first"], params["second"]

    payload = (
        _pair_double_double_version_checksum
        + _format_int32.pack(len(first))
        + numpy.asarray(first, dtype=">f8").tobytes()
        + numpy.asarray(second, dtype=">f8").tobytes()
    )
    out.append(
        uproot.serialization.numbytes_version(
            len(payload),
            _beta_bin_params_collection_version | int(uproot.const.kStreamedMemberWise),
        )
    )
    out.append(payload)


class Model_TEfficiency_v2(
    uproot.behaviors.TEfficiency.TEfficiency, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TEfficiency`` version 2.
    """

    behaviors = (uproot.behaviors.TEfficiency.TEfficiency,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
            )
        for classname, version in self.base_names_versions:
            self._bases.append(
                file.class_named(classname, version).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        self._members["fBeta_alpha"], self._members["fBeta_beta"] = cursor.fields(
            chunk, _format_double2, context
        )
        self._members["fBeta_bin_params"] = _read_beta_bin_params(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fConfLevel"] = cursor.field(chunk, _format_double, context)
        self._members["fFunctions"] = file.class_named("TList").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fPassedHistogram"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )
        self._members["fStatisticOption"] = cursor.field(chunk, _format_int32, context)
        self._members["fTotalHistogram"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )
        self._members["fWeight"] = cursor.field(chunk, _format_double, context)

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 2),
        ("TAttFill", 2),
        ("TAttMarker", 2),
    ]
    member_names = [
        "fBeta_alpha",
        "fBeta_beta",
        "fBeta_bin_params",
        "fConfLevel",
        "fFunctions",
        "fPassedHistogram",
        "fStatisticOption",
        "fTotalHistogram",
        "fWeight",
    ]
    class_flags = {"has_read_object_any": True}

    _base_rawstreamers = (
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TH._rawstreamer_TList_v5,
        uproot.models.TH._rawstreamer_TAttMarker_v2,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TAttLine_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TEfficiency_v2,
    )
    writable = True

    @property
    def class_rawstreamers(self):
        """
        The streamers needed to write this ``TEfficiency``, including those of
        its passed and total histograms (whose classes vary).
        """
        out = []
        for hist in (
            self._members.get("fPassedHistogram"),
            self._members.get("fTotalHistogram"),
        ):
            if hist is not None:
                for rawstreamer in hist.class_rawstreamers:
                    if rawstreamer not in out:
                        out.append(rawstreamer)
        for rawstreamer in self._base_rawstreamers:
            if rawstreamer not in out:
                out.append(rawstreamer)
        return tuple(out)

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)

        # TEfficiency stores its options (Bayesian, weights, ...) in TObject::fBits
        tobject_flags = numpy.uint32(tobject_flags) | numpy.uint32(
            self.member("@fBits") & _tefficiency_bits_mask
        )
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)

        out.append(
            _format_double2.pack(
                self._members["fBeta_alpha"], self._members["fBeta_beta"]
            )
        )
        _serialize_beta_bin_params(out, self._members["fBeta_bin_params"])
        out.append(_format_double.pack(self._members["fConfLevel"]))

        self._members["fFunctions"]._serialize(
            out, True, None, numpy.uint32(0x00000000)
        )

        # like TEfficiency::SetName, the histograms are named after the efficiency
        # (or just "passed" and "total" if the efficiency has no name)
        if name is None:
            name = self.member("fName")
        prefix = "" if name is None else f"{name}_"
        uproot.serialization._serialize_object_any(
            out, self._members["fPassedHistogram"], f"{prefix}passed"
        )
        out.append(_format_int32.pack(self._members["fStatisticOption"]))
        uproot.serialization._serialize_object_any(
            out, self._members["fTotalHistogram"], f"{prefix}total"
        )
        out.append(_format_double.pack(self._members["fWeight"]))

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TEfficiency(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TEfficiency``.
    """

    known_versions = {2: Model_TEfficiency_v2}


uproot.classes["TEfficiency"] = Model_TEfficiency
