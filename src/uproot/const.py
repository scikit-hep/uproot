# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines integer constants used by serialization and deserialization routines.
"""

from __future__ import annotations

from enum import IntFlag

import numpy

# determines when a file is "big"
kStartBigFile = 2000000000

kMaxTBasketBytes = numpy.iinfo(numpy.int32).max

# used in unmarshaling
kByteCountMask = numpy.int64(0x40000000)
kByteCountVMask = numpy.int64(0x4000)
kClassMask = numpy.int64(0x80000000)
kNewClassTag = numpy.int64(0xFFFFFFFF)

kIsOnHeap = numpy.uint32(0x01000000)
kNotDeleted = numpy.uint32(0x02000000)
kMustCleanup = numpy.uint32(1 << 3)
kIsReferenced = numpy.uint32(1 << 4)

kMapOffset = 2

# not used?
kNullTag = 0
kZombie = numpy.uint32(0x04000000)
kBitMask = numpy.uint32(0x00FFFFFF)
kDisplacementMask = numpy.uint32(0xFF000000)

############# core/zip/inc/Compression.h

kZLIB = 1
kLZMA = 2
kOldCompressionAlgo = 3
kLZ4 = 4
kZSTD = 5
kUndefinedCompressionAlgorithm = 6

############# constants for streamers

kBase = 0
kChar = 1
kShort = 2
kInt = 3
kLong = 4
kFloat = 5
kCounter = 6
kCharStar = 7
kDouble = 8
kDouble32 = 9
kLegacyChar = 10
kUChar = 11
kUShort = 12
kUInt = 13
kULong = 14
kBits = 15
kLong64 = 16
kULong64 = 17
kBool = 18
kFloat16 = 19
kOffsetL = 20
kOffsetP = 40
kObject = 61
kAny = 62
kObjectp = 63
kObjectP = 64
kTString = 65
kTObject = 66
kTNamed = 67
kAnyp = 68
kAnyP = 69
kAnyPnoVT = 70
kSTLp = 71

kSkip = 100
kSkipL = 120
kSkipP = 140

kConv = 200
kConvL = 220
kConvP = 240

kSTL = 300
kSTLstring = 365

kStreamer = 500
kStreamLoop = 501

############# constants from core/foundation/inc/ESTLType.h

kNotSTL = 0
kSTLvector = 1
kSTLlist = 2
kSTLdeque = 3
kSTLmap = 4
kSTLmultimap = 5
kSTLset = 6
kSTLmultiset = 7
kSTLbitset = 8
kSTLforwardlist = 9
kSTLunorderedset = 10
kSTLunorderedmultiset = 11
kSTLunorderedmap = 12
kSTLunorderedmultimap = 13
kSTLend = 14
kSTLany = 300

############# IOFeatures

kGenerateOffsetMap = numpy.uint8(1)

############# other

kStreamedMemberWise = numpy.uint16(1 << 14)

############ RNTuple https://github.com/root-project/root/blob/0b9cdbcfd326ba50ee6c2f202675656129eafbe7/tree/ntuple/v7/doc/BinaryFormatSpecification.md

rntuple_version_for_writing = (1, 0, 0, 1)

# https://github.com/root-project/root/blob/2c80bed03dbe28610c5825d8cd417db2ffc3e1d6/tree/ntuple/doc/BinaryFormatSpecification.md#attribute-schema-version
rntuple_attribute_schema_version_major = 1
rntuple_attribute_meta_field_names = ("_rangeStart", "_rangeLen", "_userData")

rntuple_col_num_to_dtype_dict = {
    0x00: "bit",
    0x01: "uint8",  # uninterpreted byte
    0x02: "uint8",  # char
    0x03: "int8",
    0x04: "uint8",
    0x05: "int16",
    0x06: "uint16",
    0x07: "int32",
    0x08: "uint32",
    0x09: "int64",
    0x0A: "uint64",
    0x0B: "float16",
    0x0C: "float32",
    0x0D: "float64",
    0x0E: "int32",  # Index32
    0x0F: "int64",  # Index64
    0x10: "switch",  # Switch: (uint64, uint32)
    0x11: "int16",  # SplitInt16: split + zigzag encoding
    0x12: "uint16",  # SplitUInt16: split encoding
    0x13: "int32",  # SplitInt32: split + zigzag encoding
    0x14: "uint32",  # SplitUInt32: split encoding
    0x15: "int64",  # SplitInt64: split + zigzag encoding
    0x16: "uint64",  # SplitUInt64: split encoding
    0x17: "float16",  # SplitReal16: split encoding
    0x18: "float32",  # SplitReal32: split encoding
    0x19: "float64",  # SplitReal64: split encoding
    0x1A: "int32",  # SplitIndex32: split + delta encoding
    0x1B: "int64",  # SplitIndex64: split + delta encoding
    0x1C: "real32trunc",  # Real32Trunc: float32 with truncated mantissa
    0x1D: "real32quant",  # Real32Quant: float32 with quantized integer representation
}
rntuple_col_num_to_size_dict = {
    0x00: 1,
    0x01: 8,
    0x02: 8,
    0x03: 8,
    0x04: 8,
    0x05: 16,
    0x06: 16,
    0x07: 32,
    0x08: 32,
    0x09: 64,
    0x0A: 64,
    0x0B: 16,
    0x0C: 32,
    0x0D: 64,
    0x0E: 32,
    0x0F: 64,
    0x10: 96,
    0x11: 16,
    0x12: 16,
    0x13: 32,
    0x14: 32,
    0x15: 64,
    0x16: 64,
    0x17: 16,
    0x18: 32,
    0x19: 64,
    0x1A: 32,
    0x1B: 64,
    0x1C: 32,  # from 10 to 31 in storage, but 32 in memory
    0x1D: 32,  # from 1 to 32 in storage, but 32 in memory
}
rntuple_col_type_to_num_dict = {
    "bit": 0x00,
    "byte": 0x01,
    "char": 0x02,
    "int8": 0x03,
    "uint8": 0x04,
    "int16": 0x05,
    "uint16": 0x06,
    "int32": 0x07,
    "uint32": 0x08,
    "int64": 0x09,
    "uint64": 0x0A,
    "real16": 0x0B,
    "real32": 0x0C,
    "real64": 0x0D,
    "index32": 0x0E,
    "index64": 0x0F,
    "switch": 0x10,
    "splitint16": 0x11,
    "splituint16": 0x12,
    "splitint32": 0x13,
    "splituint32": 0x14,
    "splitint64": 0x15,
    "splituint64": 0x16,
    "splitreal16": 0x17,
    "splitreal32": 0x18,
    "splitreal64": 0x19,
    "splitindex32": 0x1A,
    "splitindex64": 0x1B,
    "real32trunc": 0x1C,
    "real32quant": 0x1D,
}
rntuple_index_types = (
    rntuple_col_type_to_num_dict["index32"],
    rntuple_col_type_to_num_dict["index64"],
    rntuple_col_type_to_num_dict["splitindex32"],
    rntuple_col_type_to_num_dict["splitindex64"],
)
rntuple_split_types = (
    rntuple_col_type_to_num_dict["splitint16"],
    rntuple_col_type_to_num_dict["splituint16"],
    rntuple_col_type_to_num_dict["splitint32"],
    rntuple_col_type_to_num_dict["splituint32"],
    rntuple_col_type_to_num_dict["splitint64"],
    rntuple_col_type_to_num_dict["splituint64"],
    rntuple_col_type_to_num_dict["splitreal16"],
    rntuple_col_type_to_num_dict["splitreal32"],
    rntuple_col_type_to_num_dict["splitreal64"],
    rntuple_col_type_to_num_dict["splitindex32"],
    rntuple_col_type_to_num_dict["splitindex64"],
)
rntuple_zigzag_types = (
    rntuple_col_type_to_num_dict["splitint16"],
    rntuple_col_type_to_num_dict["splitint32"],
    rntuple_col_type_to_num_dict["splitint64"],
)
rntuple_delta_types = (
    rntuple_col_type_to_num_dict["splitindex32"],
    rntuple_col_type_to_num_dict["splitindex64"],
)
rntuple_custom_float_types = (
    rntuple_col_type_to_num_dict["real32trunc"],
    rntuple_col_type_to_num_dict["real32quant"],
)


class RNTupleLocatorType(IntFlag):
    STANDARD = 0x00
    LARGE = 0x01


class RNTupleEnvelopeType(IntFlag):
    RESERVED = 0x00
    HEADER = 0x01
    FOOTER = 0x02
    PAGELIST = 0x03


class RNTupleFieldRole(IntFlag):
    LEAF = 0x00
    COLLECTION = 0x01
    RECORD = 0x02
    VARIANT = 0x03
    STREAMER = 0x04


class RNTupleFieldFlags(IntFlag):
    NOFLAG = 0x00
    REPETITIVE = 0x01
    PROJECTED = 0x02
    CHECKSUM = 0x04


class RNTupleColumnFlags(IntFlag):
    NOFLAG = 0x00
    DEFERRED = 0x01
    RANGE = 0x02


class RNTupleExtraTypeIdentifier(IntFlag):
    ROOT = 0x00


class RNTupleClusterFlags(IntFlag):
    NOFLAG = 0x00
    SHARDED = 0x01
