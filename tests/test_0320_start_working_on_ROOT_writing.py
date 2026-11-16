# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os
import shutil
import struct

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


_file_header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
_file_header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")

_directory_format_small = struct.Struct(">hIIiiiii")
_directory_format_big = struct.Struct(">hIIiiqqq")
_directory_format_num_keys = struct.Struct(">i")

_key_format_small = struct.Struct(">ihiIhhii")
_key_format_big = struct.Struct(">ihiIhhqq")

_string_size_format_1 = struct.Struct(">B")
_string_size_format_4 = struct.Struct(">I")

_free_format_small = struct.Struct(">HII")
_free_format_big = struct.Struct(">HQQ")

# /home/jpivarski/storage/data/uproot4-big/issue-288.root
# /home/jpivarski/storage/data/uproot4-big/issue-90.root


@pytest.mark.parametrize(
    "source", ["uproot-Zmumu.root", "uproot-issue243-new.root", "uproot-issue64.root"]
)
def test(tmp_path, source):
    # print(source)

    filename = os.path.join(tmp_path, "update-me.root")
    shutil.copyfile(skhep_testdata.data_path(source), filename)

    # print("BEFORE")
    # for version, fFirst, fLast in get_frees(filename):
    #     print(version, fFirst, fLast)

    with open(filename, "r+b") as rawfile:
        rawfile.seek(0)
        (
            magic,
            fVersion,
            fBEGIN,
            fEND,
            fSeekFree,
            fNbytesFree,
            nfree,
            fNbytesName,
            fUnits,
            fCompress,
            fSeekInfo,
            fNbytesInfo,
            fUUID_version,
            fUUID,
        ) = _file_header_fields_small.unpack(
            rawfile.read(_file_header_fields_small.size)
        )

        if fVersion > 1000000:
            rawfile.seek(0)
            (
                magic,
                fVersion,
                fBEGIN,
                fEND,
                fSeekFree,
                fNbytesFree,
                nfree,
                fNbytesName,
                fUnits,
                fCompress,
                fSeekInfo,
                fNbytesInfo,
                fUUID_version,
                fUUID,
            ) = _file_header_fields_big.unpack(
                rawfile.read(_file_header_fields_big.size)
            )

        rawfile.seek(fSeekFree)
        freesegments = read_key(rawfile)

        rawfile.seek(freesegments["fSeekKey"] + freesegments["fKeylen"])

        frees = []
        for _ in range(nfree):
            version, fFirst, fLast = _free_format_small.unpack(
                rawfile.read(_free_format_small.size)
            )
            if version > 1000:
                rawfile.seek(rawfile.tell() - _free_format_small.size)
                version, fFirst, fLast = _free_format_big.unpack(
                    rawfile.read(_free_format_big.size)
                )
            frees.append((version, fFirst, fLast))

        directory_start = fBEGIN + fNbytesName

        rawfile.seek(directory_start)
        directory = read_directory(rawfile)

        rawfile.seek(directory["fSeekKeys"])

        beginning_of_scribble_1 = rawfile.tell()

        header_key = read_key(rawfile)
        (num_keys,) = _directory_format_num_keys.unpack(
            rawfile.read(_directory_format_num_keys.size)
        )
        keys = [read_key(rawfile) for _ in range(num_keys)]

        end_of_scribble_1 = rawfile.tell()

        beginning_of_scribble_2 = fSeekFree
        end_of_scribble_2 = fSeekFree + fNbytesFree

        # scribble over the directory and freesegments data

        rawfile.seek(beginning_of_scribble_1)
        rawfile.write(b"X" * (end_of_scribble_1 - beginning_of_scribble_1))

        rawfile.seek(beginning_of_scribble_2)
        rawfile.write(b"x" * (end_of_scribble_2 - beginning_of_scribble_2))

        # now make a new directory with the same contents

        _, beginning_of_new_directory, _ = frees[-1]

        rawfile.seek(beginning_of_new_directory)
        header_key["fSeekKey"] = beginning_of_new_directory

        write_key(rawfile, header_key)
        rawfile.write(_directory_format_num_keys.pack(num_keys))
        for key in keys:
            write_key(rawfile, key)

        end_of_new_directory = rawfile.tell()

        nfree = len(frees[:-1]) + 3
        fNbytesFree = (
            _key_format_small.size  # assuming small
            + length_string(freesegments["fClassName"])
            + length_string(freesegments["fName"])
            + length_string(freesegments["fTitle"])
            + nfree * _free_format_small.size  # assuming small
        )
        frees = frees[:-1] + [
            (
                1,
                beginning_of_scribble_1,
                end_of_scribble_1 - 1,
            ),  # fLast is really *last*
            (
                1,
                beginning_of_scribble_2,
                end_of_scribble_2 - 1,
            ),  # *not* the first unfree
            (1, end_of_new_directory + fNbytesFree, 2000000000),
        ]

        fSeekFree = rawfile.tell()

        freesegments["fSeekKey"] = fSeekFree
        freesegments["fNbytes"] = fNbytesFree
        freesegments["fObjlen"] = nfree * _free_format_small.size  # assuming small

        write_key(rawfile, freesegments)
        for free in frees:
            version, fFirst, fLast = free
            if version <= 1000:
                rawfile.write(_free_format_small.pack(version, fFirst, fLast))
            else:
                rawfile.write(_free_format_big.pack(version, fFirst, fLast))

        fEND = rawfile.tell()

        directory["fSeekKeys"] = beginning_of_new_directory

        rawfile.seek(directory_start)
        write_directory(rawfile, directory)

        rawfile.seek(0)
        if fVersion <= 1000000:
            rawfile.write(
                _file_header_fields_small.pack(
                    magic,
                    fVersion,
                    fBEGIN,
                    fEND,
                    fSeekFree,
                    fNbytesFree,
                    nfree,
                    fNbytesName,
                    fUnits,
                    fCompress,
                    fSeekInfo,
                    fNbytesInfo,
                    fUUID_version,
                    fUUID,
                )
            )
        else:
            rawfile.write(
                _file_header_fields_big.pack(
                    magic,
                    fVersion,
                    fBEGIN,
                    fEND,
                    fSeekFree,
                    fNbytesFree,
                    nfree,
                    fNbytesName,
                    fUnits,
                    fCompress,
                    fSeekInfo,
                    fNbytesInfo,
                    fUUID_version,
                    fUUID,
                )
            )

    # print("AFTER MY UPDATE")
    # for version, fFirst, fLast in get_frees(filename):
    #     print(version, fFirst, fLast)

    # read it back in ROOT and add another object

    f1 = ROOT.TFile(filename, "UPDATE")

    if source == "uproot-Zmumu.root":
        t = f1.Get("events")
        assert str(t.GetBranch("px1").GetTitle()) == "px1/D"
    elif source == "uproot-issue243-new.root":
        t = f1.Get("sig")
        assert str(t.GetBranch("m_hh").GetTitle()) == "m_hh/D"
    elif source == "uproot-issue64.root":
        t = f1.Get("events/events")
        assert str(t.GetBranch("eventid").GetTitle()) == "eventid/I"

    x = ROOT.TObjString("hello")
    x.Write()
    f1.Close()

    # print("AFTER ROOT'S UPDATE")
    # for version, fFirst, fLast in get_frees(filename):
    #     print(version, fFirst, fLast)

    f2 = ROOT.TFile(filename)
    assert str(f2.Get("hello")) == "hello"
    f2.Close()

    with uproot.open(filename) as f3:
        assert str(f3["hello"]) == "hello"
        key = f3.key("hello")
        # print(key.fSeekKey, key.fKeylen, key.fNbytes)


def read_string(rawfile):
    (string_size,) = _string_size_format_1.unpack(rawfile.read(1))
    if string_size == 255:
        (string_size,) = _string_size_format_4.unpack(rawfile.read(4))
    return rawfile.read(string_size).decode(errors="surrogateescape")


def length_string(string):
    bytestring = string.encode(errors="surrogateescape")
    length = len(bytestring)
    if len(bytestring) < 255:
        return len(bytestring) + 1
    else:
        return len(bytestring) + 5


def write_string(rawfile, string):
    bytestring = string.encode(errors="surrogateescape")
    length = len(bytestring)
    if len(bytestring) < 255:
        rawfile.write(struct.pack(">B%ds" % length, length, bytestring))
    else:
        rawfile.write(struct.pack(">BI%ds" % length, 255, length, bytestring))


def read_directory(rawfile):
    (
        fVersion,
        fDatimeC,
        fDatimeM,
        fNbytesKeys,
        fNbytesName,
        fSeekDir,
        fSeekParent,
        fSeekKeys,
    ) = _directory_format_small.unpack(rawfile.read(_directory_format_small.size))

    if fVersion > 1000:
        rawfile.seek(directory_start)
        (
            fVersion,
            fDatimeC,
            fDatimeM,
            fNbytesKeys,
            fNbytesName,
            fSeekDir,
            fSeekParent,
            fSeekKeys,
        ) = _directory_format_big.unpack(rawfile.read(_directory_format_big.size))

    return {
        "fVersion": fVersion,
        "fDatimeC": fDatimeC,
        "fDatimeM": fDatimeM,
        "fNbytesKeys": fNbytesKeys,
        "fNbytesName": fNbytesName,
        "fSeekDir": fSeekDir,
        "fSeekParent": fSeekParent,
        "fSeekKeys": fSeekKeys,
    }


def write_directory(rawfile, directory):
    if directory["fVersion"] <= 1000:
        rawfile.write(
            _directory_format_small.pack(
                directory["fVersion"],
                directory["fDatimeC"],
                directory["fDatimeM"],
                directory["fNbytesKeys"],
                directory["fNbytesName"],
                directory["fSeekDir"],
                directory["fSeekParent"],
                directory["fSeekKeys"],
            )
        )
    else:
        rawfile.write(
            _directory_format_big.pack(
                directory["fVersion"],
                directory["fDatimeC"],
                directory["fDatimeM"],
                directory["fNbytesKeys"],
                directory["fNbytesName"],
                directory["fSeekDir"],
                directory["fSeekParent"],
                directory["fSeekKeys"],
            )
        )


def read_key(rawfile):
    position = rawfile.tell()

    (
        fNbytes,
        fVersion,
        fObjlen,
        fDatime,
        fKeylen,
        fCycle,
        fSeekKey,
        fSeekPdir,
    ) = _key_format_small.unpack(rawfile.read(_key_format_small.size))

    if fVersion > 1000:
        rawfile.seek(position)
        (
            fNbytes,
            fVersion,
            fObjlen,
            fDatime,
            fKeylen,
            fCycle,
            fSeekKey,
            fSeekPdir,
        ) = _key_format_big.unpack(rawfile.read(_key_format_big.size))

    fClassName = read_string(rawfile)
    fName = read_string(rawfile)
    fTitle = read_string(rawfile)

    return {
        "fNbytes": fNbytes,
        "fVersion": fVersion,
        "fObjlen": fObjlen,
        "fDatime": fDatime,
        "fKeylen": fKeylen,
        "fCycle": fCycle,
        "fSeekKey": fSeekKey,
        "fSeekPdir": fSeekPdir,
        "fClassName": fClassName,
        "fName": fName,
        "fTitle": fTitle,
    }


def write_key(rawfile, key):
    if key["fVersion"] <= 1000:
        rawfile.write(
            _key_format_small.pack(
                key["fNbytes"],
                key["fVersion"],
                key["fObjlen"],
                key["fDatime"],
                key["fKeylen"],
                key["fCycle"],
                key["fSeekKey"],
                key["fSeekPdir"],
            )
        )
    else:
        rawfile.write(
            _key_format_big.pack(
                key["fNbytes"],
                key["fVersion"],
                key["fObjlen"],
                key["fDatime"],
                key["fKeylen"],
                key["fCycle"],
                key["fSeekKey"],
                key["fSeekPdir"],
            )
        )

    write_string(rawfile, key["fClassName"])
    write_string(rawfile, key["fName"])
    write_string(rawfile, key["fTitle"])


def get_frees(filename):
    with open(filename, "rb") as rawfile:
        rawfile.seek(0)
        (
            magic,
            fVersion,
            fBEGIN,
            fEND,
            fSeekFree,
            fNbytesFree,
            nfree,
            fNbytesName,
            fUnits,
            fCompress,
            fSeekInfo,
            fNbytesInfo,
            fUUID_version,
            fUUID,
        ) = _file_header_fields_small.unpack(
            rawfile.read(_file_header_fields_small.size)
        )

        if fVersion > 1000000:
            rawfile.seek(0)
            (
                magic,
                fVersion,
                fBEGIN,
                fEND,
                fSeekFree,
                fNbytesFree,
                nfree,
                fNbytesName,
                fUnits,
                fCompress,
                fSeekInfo,
                fNbytesInfo,
                fUUID_version,
                fUUID,
            ) = _file_header_fields_big.unpack(
                rawfile.read(_file_header_fields_big.size)
            )

        rawfile.seek(fSeekFree)
        freesegments = read_key(rawfile)

        rawfile.seek(freesegments["fSeekKey"] + freesegments["fKeylen"])

        frees = []
        for _ in range(nfree):
            version, fFirst, fLast = _free_format_small.unpack(
                rawfile.read(_free_format_small.size)
            )
            if version > 1000:
                rawfile.seek(rawfile.tell() - _free_format_small.size)
                version, fFirst, fLast = _free_format_big.unpack(
                    rawfile.read(_free_format_big.size)
                )
            frees.append((version, fFirst, fLast))

        return frees
