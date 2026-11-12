# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines low-level methods for serialization, including :doc:`uproot.serialization.string`,
which prepends string lengths following ROOT's convention, :doc:`uproot.serialization.numbytes_version`,
the opposite of :doc:`uproot.deserialization.numbytes_version`, and :doc:`uproot.serialization.serialize_object_any`,
the opposite of :doc:`uproot.deserialization.read_object_any`.
"""

from __future__ import annotations

import struct

import numpy

import uproot.const
import uproot.deserialization


def string(data):
    """
    Converts a Python string into bytes, ready to be written to a file.

    If the string's byte representation (UTF-8) has fewer than 255 bytes, it
    is preceded by a 1-byte length; otherwise, it is preceded by ``b'\xff'`` and a
    4-byte length.
    """
    return bytestring(data.encode(errors="surrogateescape"))


def bytestring(data):
    """
    Converts Python bytes into a length-prefixed bytestring, ready to be written to a file.

    If the string's byte representation (UTF-8) has fewer than 255 bytes, it
    is preceded by a 1-byte length; otherwise, it is preceded by ``b'\xff'`` and a
    4-byte length.
    """
    length = len(data)
    if length < 255:
        return struct.pack(f">B{length}s", length, data)
    else:
        return struct.pack(f">BI{length}s", 255, length, data)


def numbytes_version(num_bytes, version):
    """
    Args:
        num_bytes (int): The first value to include in the header.
        version (int): The second value to include in the header.

    Returns a 6-byte bytes object consisting of ``num_bytes`` and ``version``,
    with the appropriate offset (+2) and ``uproot.const.kByteCountMask`` applied
    to ``num_bytes``.

    This function is the opposite of :doc:`uproot.deserialization.numbytes_version`,
    but only generates one case (whereas the deserializer must handle special flags
    in the ``num_bytes``).
    """
    return uproot.deserialization._numbytes_version_1.pack(
        numpy.uint32(num_bytes + 2) | uproot.const.kByteCountMask, version
    )


_serialize_object_any_format1 = struct.Struct(">II")


def _serialize_object_any(out, model, name):
    if model is None:
        out.append(b"\x00\x00\x00\x00")

    else:
        where = len(out)
        model._serialize(out, True, name, numpy.uint32(0x00000000))

        classname = model.classname.encode(errors="surrogateescape") + b"\x00"
        num_bytes = sum(len(x) for x in out[where:]) + len(classname) + 4
        bcnt = numpy.uint32(num_bytes) | uproot.const.kByteCountMask
        tag = uproot.const.kNewClassTag

        out.insert(where, _serialize_object_any_format1.pack(bcnt, tag) + classname)


def serialize_object_any(model, name=None):
    """
    Args:
        model (:doc:`uproot.model.Model`): Object to serialize.
        name (str or None): If not None, overrides the object's name.

    Serializes the object in a form that could be read by :doc:`uproot.deserialization.read_object_any`.
    """
    out = []
    _serialize_object_any(out, model, name)
    return b"".join(out)
