# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module defining the "cascading" file writer, which is like a
reactive programming environment. A tree of data structures whose leaves are
:doc:`uproot.writing._cascade.CascadeLeaf` each control a part of the ROOT file,
and when a change is needed, they rewrite their section in an organized flow that
updates pointers after updating content, so that pointers are always valid.

Modifying data in a ROOT file has many non-local effects. For instance, if you add
a new object, the space for that object must be claimed from the FreeSegments record
at the end of the file, and the FreeSegments record may even need to be rewritten
because it has to move to make space for the new object. Then a pointer must be added
to a TDirectory to point to the new object, and that might exceed the space allocated
for the TDirectory, in which case it, too needs to claim space from FreeSegments
(possibly moving it). When a TDirectory moves, the pointer to it must be updated
as well, which might be in another TDirectory if it's a subdirectory, etc. If these
updates were not organized, redundant overwrites or even infinite cycles could occur.
Thus, the structure of the cascade tree must be carefully laid out.
"""

from __future__ import annotations

import datetime
import math
import os.path
import struct
import uuid

import numpy

import uproot.compression
import uproot.const
import uproot.models.TList
import uproot.reading
import uproot.serialization
import uproot.source.chunk
import uproot.source.cursor
import uproot.streamers


class CascadeLeaf:
    """
    A leaf node in the tree of cascading, low-level writables.

    Only leaf nodes actually write to the file. The general strategy is that non-leaf
    nodes modify their children's attributes, and any modification causes a leaf's
    ``file_dirty`` property to become True. When all variables are updated, the
    writing process is triggered in a single pass, though a particular leaf will
    only actually write if its ``file_dirty`` is True.

    This is similar to the way that graphical elements update pixels in a graphical
    user interface: each item owns a region of pixels and only updates those pixels
    if they are marked as "dirty" (would change). The difference here is that a
    file is one-dimensional. Leaves own a one-dimensional interval from ``location``
    to ``location + num_bytes``. Some of these attributes might not be known (and
    therefore set to None) before it is time to write.

    See :doc:`uproot.writing._cascade` for a general overview.
    """

    def __init__(self, location, allocation):
        self._location = location
        self._allocation = allocation
        self._file_dirty = True

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        if self._location != value:
            self._file_dirty = True
            self._location = value

    @property
    def allocation(self):
        if self._allocation is None:
            self._allocation = self.num_bytes
        return self._allocation

    @property
    def num_bytes(self):
        return len(self.serialize())

    def serialize(self):
        raise AssertionError("CascadeLeaf is abstract; 'serialize' must be overloaded")

    @classmethod
    def deserialize(cls, raw_bytes, location):
        raise AssertionError(
            "CascadeLeaf is abstract; 'deserialize' must be overloaded"
        )

    def write(self, sink):
        if self._file_dirty:
            if self._location is None:
                raise RuntimeError(
                    "can't write object because location is unknown:\n\n    "
                    + repr(self)
                )
            tmp = self.serialize()
            # print(f"writing {self._location}:{self._location + len(tmp)} ({len(tmp)}) {type(self).__name__} {self.name if hasattr(self, 'name') else ''} {self.title if hasattr(self, 'title') else ''}")
            sink.write(self._location, tmp)
            self._file_dirty = False


class CascadeNode:
    """
    A non-leaf node in the tree of cascading, low-level writables.

    Non-leaf nodes contain other nodes as attributes and propagate their information
    downward toward the leaves, which do the actual writing.

    A non-leaf node defines a ``dependencies`` order in which the writing is to take
    place. The order is such that referents are written before references. If writing
    is interrupted after writing a referent but before writing its references, the
    resulting file will contain unreachable data, but all references would be valid,
    providing some resilience against incomplete writes.

    See :doc:`uproot.writing._cascade` for a general overview.
    """

    def __init__(self, *dependencies):
        self._dependencies = dependencies

    def more_dependencies(self, *dependencies):
        self._dependencies = self._dependencies + dependencies

    def write(self, sink):
        for dependency in self._dependencies:
            dependency.write(sink)


_string_size_format_4 = struct.Struct(">I")


class String(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a string, such
    as a name, a title, or a class name.

    If the string's byte representation (UTF-8) has fewer than 255 bytes, it
    is preceded by a 1-byte length; otherwise, it is preceded by ``b'\xff'`` and a
    4-byte length.
    """

    def __init__(self, location, string):
        super().__init__(location, None)
        self._string = string
        self._serialization = uproot.serialization.string(self._string)

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._string!r})"

    @property
    def string(self):
        return self._string

    def copy_to(self, location):
        return String(location, self._string)

    def serialize(self):
        return self._serialization

    @classmethod
    def deserialize(cls, raw_bytes, location):
        num_bytes = ord(raw_bytes[:1])
        position = 1
        if num_bytes == 255:
            (num_bytes,) = _string_size_format_4.unpack(raw_bytes[1:5])
            position = 5
        out = raw_bytes[position : position + num_bytes].decode(
            errors="surrogateescape"
        )
        return String(location, out), location + position + num_bytes


class Key(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a ROOT TKey.

    TKeys may have 4-byte pointers (small) or 8-byte pointers (big), which changes
    their size depending on where their referents are located (before or after the
    ``kStartBigFile`` boundary of 2 GB, not 2 GiB). Their size also depends on
    three strings: name, title, and class name, but these do not change as
    an object's position moves within the file.

    See `ROOT TKey specification <https://github.com/root-project/root/blob/master/io/doc/TFile/datarecord.md>`__.
    """

    class_version = 4

    def __init__(
        self,
        location,
        uncompressed_bytes,
        compressed_bytes,
        classname,
        name,
        title,
        cycle,
        parent_location,
        seek_location,
        created_on=None,
        big=None,
    ):
        super().__init__(location, None)
        self._uncompressed_bytes = uncompressed_bytes
        self._compressed_bytes = compressed_bytes
        self._classname = classname
        self._name = name
        self._title = title
        self._cycle = cycle
        self._parent_location = parent_location
        self._seek_location = seek_location
        self._created_on = datetime.datetime.now() if created_on is None else created_on
        self._big = big

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._uncompressed_bytes}, {self._compressed_bytes}, {self._classname!r}, {self._name!r}, {self._title!r}, {self._cycle}, {self._parent_location}, {self._seek_location})"

    @property
    def allocation(self):
        return self.num_bytes

    @property
    def uncompressed_bytes(self):
        return self._uncompressed_bytes

    @uncompressed_bytes.setter
    def uncompressed_bytes(self, value):
        if self._uncompressed_bytes != value:
            self._file_dirty = True
            self._uncompressed_bytes = value

    @property
    def compressed_bytes(self):
        return self._compressed_bytes

    @compressed_bytes.setter
    def compressed_bytes(self, value):
        if self._compressed_bytes != value:
            self._file_dirty = True
            self._compressed_bytes = value

    @property
    def classname(self):
        return self._classname

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def cycle(self):
        return self._cycle

    @property
    def parent_location(self):
        return self._parent_location

    @parent_location.setter
    def parent_location(self, value):
        if self._parent_location != value:
            self._file_dirty = True
            self._big = None
            self._parent_location = value

    @property
    def seek_location(self):
        return self._seek_location

    @seek_location.setter
    def seek_location(self, value):
        if self._seek_location != value:
            self._file_dirty = True
            self._big = None
            self._seek_location = value

    def copy_to(self, location):
        position = location + self.num_bytes
        classname = self._classname.copy_to(position)

        position += classname.num_bytes
        name = self._name.copy_to(position)

        position += name.num_bytes
        title = self._title.copy_to(position)

        if self._seek_location is not None:
            location = self._seek_location
        else:
            location = self._location

        return Key(
            location,
            self._uncompressed_bytes,
            self._compressed_bytes,
            classname,
            name,
            title,
            self._cycle,
            self._parent_location,
            location,
            created_on=self._created_on,
            big=self._big,
        )

    @property
    def big(self):
        if self._big is not None:
            return self._big
        elif self._seek_location is not None:
            return (
                self._seek_location >= uproot.const.kStartBigFile
                or self._parent_location >= uproot.const.kStartBigFile
            )
        else:
            return (
                self._location is None
                or self._location >= uproot.const.kStartBigFile
                or self._parent_location >= uproot.const.kStartBigFile
            )

    @property
    def num_bytes(self):
        if self.big:
            return (
                uproot.reading._key_format_big.size
                + self._classname.allocation
                + self._name.allocation
                + self._title.allocation
            )
        else:
            return (
                uproot.reading._key_format_small.size
                + self._classname.allocation
                + self._name.allocation
                + self._title.allocation
            )

    def serialize(self):
        if self._seek_location is None and self._location is None:
            raise RuntimeError(
                "can't serialize key because location is unknown:\n\n    " + repr(self)
            )

        if self.big:
            format = uproot.reading._key_format_big
            version = self.class_version + 1000
        else:
            format = uproot.reading._key_format_small
            version = self.class_version

        if self._seek_location is not None:
            location = self._seek_location
        else:
            location = self._location

        return (
            format.pack(
                self._compressed_bytes + self.num_bytes,  # fNbytes
                version,  # fVersion
                self._uncompressed_bytes,  # fObjlen
                uproot._util.datetime_to_code(self._created_on),  # fDatime
                self.num_bytes,  # fKeylen
                self._cycle,  # fCycle
                location,  # fSeekKey
                self._parent_location,  # fSeekPdir
            )
            + self._classname.serialize()
            + self._name.serialize()
            + self._title.serialize()
        )

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path, is_directory_key=False):
        (
            fNbytes,
            version,
            fObjlen,
            fDatime,
            fKeylen,
            fCycle,
            fSeekKey,
            fSeekPdir,
        ) = uproot.reading._key_format_small.unpack(
            raw_bytes[: uproot.reading._key_format_small.size]
        )
        position = location + uproot.reading._key_format_small.size
        big = False

        if version >= 1000:
            (
                fNbytes,
                version,
                fObjlen,
                fDatime,
                fKeylen,
                fCycle,
                fSeekKey,
                fSeekPdir,
            ) = uproot.reading._key_format_big.unpack(
                raw_bytes[: uproot.reading._key_format_big.size]
            )
            version -= 1000
            position = location + uproot.reading._key_format_big.size
            big = True

        if version != cls.class_version:
            raise ValueError(
                f"Uproot can't read TKey version {version} for writing, only version {cls.class_version}{in_path}"
            )

        assert 0 < fNbytes <= fKeylen + fObjlen
        assert fCycle > 0
        if not is_directory_key:
            assert fSeekKey == location, f"fSeekKey {fSeekKey} location {location}"
            fSeekKey = None

        classname, position = String.deserialize(
            raw_bytes[position - location :], position
        )
        name, position = String.deserialize(raw_bytes[position - location :], position)
        title, position = String.deserialize(raw_bytes[position - location :], position)

        assert fKeylen == position - location

        return Key(
            location,
            fObjlen,  # uncompressed_bytes
            fNbytes - fKeylen,  # compressed_bytes
            classname,
            name,
            title,
            fCycle,  # cycle
            fSeekPdir,  # parent_location
            fSeekKey,  # may be location
            created_on=uproot._util.code_to_datetime(fDatime),
            big=big,
        )


_free_format_small = struct.Struct(">HII")
_free_format_big = struct.Struct(">HQQ")


def _slices_bytes(slices):
    total = 0
    for _, stop in slices:
        if stop - 1 >= uproot.const.kStartBigFile:
            total += _free_format_big.size
        else:
            total += _free_format_small.size
    return total


def _free_segments_num_bytes(slices, file_end, data_location=0):
    total = _slices_bytes(slices)

    if file_end is None:
        file_end = (data_location or 0) + total + _free_format_small.size

    if file_end >= uproot.const.kStartBigFile:
        return total + _free_format_big.size
    else:
        return total + _free_format_small.size


class FreeSegmentsData(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for the FreeSegments record
    """

    class_version = 1

    def __init__(self, location, slices, end):
        super().__init__(location, None)
        self._slices = slices
        self._end = end

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._slices}, {self._end})"

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, value):
        if self._slices != value:
            self._file_dirty = True
            self._allocation = None
            self._slices = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if self._end != value:
            self._file_dirty = True
            self._allocation = None
            self._end = value

    @property
    def allocation(self):
        if self._allocation is None:
            self._allocation = self.num_bytes
        return self._allocation

    @allocation.setter
    def allocation(self, value):
        if self._allocation != value:
            self._file_dirty = True
            self._allocation = value

    @property
    def num_bytes(self):
        return _free_segments_num_bytes(self._slices, self._end, self._location)

    def required_end(self, data_location):
        return data_location + _free_segments_num_bytes(
            self._slices, None, data_location
        )

    def serialize(self):
        pairs = []
        for start, stop in self._slices:
            if stop - 1 < uproot.const.kStartBigFile:
                pairs.append(
                    _free_format_small.pack(self.class_version, start, stop - 1)
                )
            else:
                pairs.append(
                    _free_format_big.pack(self.class_version + 1000, start, stop - 1)
                )

        if self._end < uproot.const.kStartBigFile:
            pairs.append(
                _free_format_small.pack(
                    self.class_version, self._end, uproot.const.kStartBigFile
                )
            )
        else:
            infinity = uproot.const.kStartBigFile
            while not self._end < infinity:
                infinity *= 2
            pairs.append(
                _free_format_big.pack(self.class_version + 1000, self._end, infinity)
            )

        return b"".join(pairs)

    @classmethod
    def deserialize(cls, raw_bytes, location, num_bytes, num_slices, in_path):
        slices = []
        position = 0

        for _ in range(num_slices + 1):
            version, fFirst, fLast = _free_format_small.unpack(
                raw_bytes[position : position + _free_format_small.size]
            )
            if version >= 1000:
                version, fFirst, fLast = _free_format_big.unpack(
                    raw_bytes[position : position + _free_format_big.size]
                )
                version -= 1000
                position += _free_format_big.size
            else:
                position += _free_format_small.size

            if version != cls.class_version:
                raise ValueError(
                    f"Uproot can't read TFree version {version} for writing, only version {cls.class_version}{in_path}"
                )

            slices.append((fFirst, fLast + 1))

        end = slices.pop()[0]

        assert position == num_bytes

        out = FreeSegmentsData(location, tuple(slices), end)
        out._allocation = num_bytes
        return out


class FreeSegments(CascadeNode):
    """
    A :doc:`uproot.writing._cascade.CascadeNode` for writing a ROOT FreeSegments record.

    The FreeSegments describes which parts of the ROOT file are unused by valid objects;
    they may have been invalidated by moving/rewriting an object in a new space. Every
    request for a new allocation or release of an old one goes through this object: the
    disk I/O equivalent of malloc and free.

    Since allocation and release usually change the size of the FreeSegments record,
    forcing it to be rewritten and possibly requiring another allocation, our ``allocate``
    method attempts to keep the FreeSegments record at the end of the file, where it can
    grow freely without bumping into an object after it (which might itself invoke
    another allocation!). The exceptions to this rule are when we are updating an
    existing ROOT file, the FreeSegments record is not at the end of that file, and
    updating it does not require more space.

    This is different from ROOT's allocation algorithm, but compatible with it because
    Uproot and ROOT can both accept a FreeSegments record at any position in a file.

    See `ROOT TFree specification <https://github.com/root-project/root/blob/master/io/doc/TFile/freesegments.md>`__.
    """

    def __init__(self, key, data, fileheader):
        super().__init__(key, data, fileheader)
        self._key = key
        self._data = data
        self._fileheader = fileheader

    def __repr__(self):
        return f"{type(self).__name__}({self._key}, {self._data}, {self._fileheader})"

    @property
    def key(self):
        return self._key

    @property
    def data(self):
        return self._data

    @property
    def fileheader(self):
        return self._fileheader

    @property
    def at_end(self):
        end_of_record = self._key.location + self._key.num_bytes + self._data.allocation
        assert end_of_record <= self._data.end
        return end_of_record == self._data.end

    def allocate(self, num_bytes, dry_run=False):
        slices = self._data.slices
        for i, (start, stop) in enumerate(slices):
            if stop - start == num_bytes:
                # This will reduce the num_bytes of the FreeSegments record,
                # but the allocation can stay the same size.
                if not dry_run:
                    self._data.slices = tuple(
                        slices[j] for j in range(len(slices)) if i != j
                    )
                return start

            elif stop - start > num_bytes:
                # This will not change the num_bytes of the FreeSegments record.
                if not dry_run:
                    self._data.slices = tuple(
                        slices[j] if i != j else (start + num_bytes, stop)
                        for j in range(len(slices))
                    )
                return start

        if self.at_end:
            # The new object can take FreeSegments's spot; FreeSegments will
            # move to stay at the end.
            out = self._key.location
            if not dry_run:
                self._key.location = self._key.location + num_bytes
                self._data.end = self._data.required_end(
                    self._key.location + self._key.num_bytes
                )
            return out

        else:
            # FreeSegments is not changing size and not at the end; it can
            # stay where it is.
            out = self._data.end
            if not dry_run:
                self._data.end = self._data.end + num_bytes
            return out

    @staticmethod
    def _another_slice(slices, original_start, original_stop):
        for start, stop in slices:
            if start <= original_start < stop or start < original_stop <= stop:
                raise RuntimeError(
                    "segment of data to release overlaps one already marked as free: "
                    f"releasing [{original_start}, {original_stop}) but [{start}, {stop}) is free"
                )

        for i in range(len(slices) - 1):
            if slices[i][1] == original_start and original_stop == slices[i + 1][0]:
                # These two slices need to be merged, including the newly released interval.
                return (*slices[:i], (slices[i][0], slices[i + 1][1]), *slices[i + 2 :])

        for i, (start, stop) in enumerate(slices):
            if original_start == stop:
                # This slice needs to grow to the right.
                return tuple(
                    slices[j] if i != j else (start, original_stop)
                    for j in range(len(slices))
                )

            elif original_stop == start:
                # This slice needs to grow to the left.
                return tuple(
                    slices[j] if i != j else (original_start, stop)
                    for j in range(len(slices))
                )

        # The FreeSegments record will have to grow.
        return tuple(sorted((*slices, (original_start, original_stop))))

    @staticmethod
    def _slices_bytes(slices):
        return _slices_bytes(slices)

    def release(self, start, stop):
        new_slices = self._another_slice(self._data.slices, start, stop)

        if self.at_end:
            self._data.slices = new_slices
            self._data.end = self._data.required_end(
                self._key.location + self._key.num_bytes
            )
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes

        elif self._slices_bytes(new_slices) <= self._slices_bytes(self._data.slices):
            # Wherever the FreeSegments record is, it's not getting bigger.
            # It can stay there.
            self._data.slices = new_slices
            self._data.allocation = None
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes

        else:
            # The FreeSegments record needs to move, opening up yet another slice.
            # Move it to the end (regardless of whether there's now enough room
            # to put it elsewhere; we like keeping it at the end).
            self._data.slices = self._another_slice(
                new_slices,
                self._key.location,
                self._key.location + self._key.allocation + self._data.allocation,
            )
            self._key.location = self._data.end
            self._data.end = self._data.required_end(
                self._key.location + self._key.num_bytes
            )
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes

    def write(self, sink):
        self._key.uncompressed_bytes = self._data.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._data.location = self._key.location + self._key.allocation
        self._fileheader.free_location = self._key.location
        self._fileheader.free_num_bytes = self._key.allocation + self._data.allocation
        self._fileheader.free_num_slices = len(self._data.slices)
        self._fileheader.end = self._data.end
        super().write(sink)


_tlistheader_format = struct.Struct(">IHHIIBI")


class TListHeader(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a ROOT TList header.

    A TList contains the TStreamerInfo records, so it must be writable in the cascade.
    """

    class_version = 5

    def __init__(self, location, data_bytes, num_entries):
        super().__init__(location, _tlistheader_format.size)
        self._data_bytes = data_bytes
        self._num_entries = num_entries

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._data_bytes}, {self._num_entries})"

    @property
    def data_bytes(self):
        return self._data_bytes

    @data_bytes.setter
    def data_bytes(self, value):
        if self._data_bytes != value:
            self._file_dirty = True
            self._data_bytes = value

    @property
    def num_entries(self):
        return self._num_entries

    @num_entries.setter
    def num_entries(self, value):
        if self._num_entries != value:
            self._file_dirty = True
            self._num_entries = value

    @property
    def num_bytes(self):
        return _tlistheader_format.size

    def serialize(self):
        return _tlistheader_format.pack(
            numpy.uint32(self._data_bytes - 4) | uproot.const.kByteCountMask,
            self.class_version,
            1,  # TObject version
            0,  # TObject::fUniqueID
            uproot.const.kNotDeleted,  # TObject::fBits
            0,
            self._num_entries,
        )


class RawStreamerInfo(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a TStreamerInfo record
    defined by a raw byte stream, rather than a fully interpreted
    :doc:`uproot.streamers.Model_TStreamerInfo`.
    """

    def __init__(self, location, serialization, name, class_version):
        super().__init__(location, len(serialization))
        self._serialization = serialization
        self._name = name
        self._class_version = class_version

    @property
    def name(self):
        return self._name

    @property
    def class_version(self):
        return self._class_version

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._serialization}, {self._name!r}, {self._class_version})"

    def copy_to(self, location):
        return RawStreamerInfo(
            location, self._serialization, self._name, self._class_version
        )

    def serialize(self):
        return self._serialization


class RawTListOfStrings(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a TList of ROOT TStrings.

    Sometimes, a TList of TStrings is included in a TList of TStreamerInfo to specify
    schema evolution rules as snippets of C++ code. Uproot can't interpret them, but it
    does need to pass them on when updating a preexisting file.
    """

    def __init__(self, location, serialization):
        super().__init__(location, len(serialization))
        self._serialization = serialization

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._serialization})"

    def copy_to(self, location):
        return RawTListOfStrings(location, self._serialization)

    def serialize(self):
        return self._serialization


class TListOfStreamers(CascadeNode):
    """
    A :doc:`uproot.writing._cascade.CascadeNode` for writing a TList of ROOT TStreamerInfos.

    (And sometimes a nested TList of TStrings, see :doc:`uproot.writing._cascade.RawTListOfStrings`.)
    """

    def __init__(self, allocation, key, header, rawstreamers, freesegments):
        super().__init__(freesegments, key, header, *rawstreamers)
        self._allocation = allocation
        self._key = key
        self._header = header
        self._rawstreamers = rawstreamers
        self._freesegments = freesegments
        self._lookup = {
            (x.name, x.class_version)
            for x in self._rawstreamers
            if not isinstance(x, RawTListOfStrings)
        }

    def __repr__(self):
        return f"{type(self).__name__}({self._allocation}, {self._key}, {self._header}, {self._rawstreamers}, {self._freesegments})"

    @property
    def allocation(self):
        return self._allocation

    @property
    def num_bytes(self):
        return self._header.num_bytes + sum(x.num_bytes for x in self._rawstreamers)

    @property
    def key(self):
        return self._key

    @property
    def header(self):
        return self._header

    @property
    def freesegments(self):
        return self._freesegments

    def update_streamers(self, sink, streamers, flush=True):
        where = len(self._rawstreamers)

        for streamer in streamers:
            pair = (streamer.name, streamer.class_version)
            if pair not in self._lookup:
                self._lookup.add(pair)

                if isinstance(streamer, uproot.streamers.Model_TStreamerInfo):
                    self._rawstreamers.append(
                        RawStreamerInfo(
                            self._key.location + self.num_bytes,
                            uproot.serialization.serialize_object_any(
                                streamer, streamer.name
                            )
                            + b"\x00",
                            streamer.name,
                            streamer.class_version,
                        )
                    )

                elif isinstance(streamer, RawStreamerInfo):
                    self._rawstreamers.append(
                        streamer.copy_to(self._key.location + self.num_bytes)
                    )

        self.more_dependencies(*self._rawstreamers[where:])

        if flush:
            self.write(sink)
            sink.flush()

    def _reallocate(self, self_num_bytes):
        original_start = self._key.location
        original_stop = self._key.location + self._key.allocation + self._allocation

        requested_num_bytes = self._key.num_bytes + self_num_bytes

        self._key.location = self._freesegments.allocate(requested_num_bytes)
        self._key.seek_location = self._key.location
        self._allocation = self_num_bytes

        self._freesegments.release(original_start, original_stop)

    def write(self, sink):
        self_num_bytes = self.num_bytes
        if self_num_bytes > self.allocation:
            self._reallocate(self_num_bytes)

        position = afterkey = self._key.location + self._key.num_bytes

        self._header.location = position
        position += self._header.num_bytes

        for rawstreamer in self._rawstreamers:
            rawstreamer.location = position
            position += rawstreamer.num_bytes

        self._header.data_bytes = position - afterkey
        self._header.num_entries = len(self._rawstreamers)

        self._key.uncompressed_bytes = self._allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._freesegments.fileheader.info_location = self._key.location
        self._freesegments.fileheader.info_num_bytes = (
            self._key.allocation + self._allocation
        )

        super().write(sink)

    @classmethod
    def deserialize(cls, raw_bytes, location, key, freesegments, file_path, uuid):
        readforupdate = _ReadForUpdate(file_path, uuid)

        chunk = uproot.source.chunk.Chunk.wrap(readforupdate, raw_bytes)

        if key.compressed_bytes == key.uncompressed_bytes:
            uncompressed = chunk
        else:
            uncompressed = uproot.compression.decompress(
                chunk,
                uproot.source.cursor.Cursor(0),
                {},
                key.compressed_bytes,
                key.uncompressed_bytes,
            )

        tlist = uproot.models.TList.Model_TList.read(
            uncompressed,
            uproot.source.cursor.Cursor(0, origin=-key.num_bytes),
            {},
            readforupdate,
            readforupdate,
            None,
        )

        header = TListHeader(location, key.uncompressed_bytes, len(tlist))

        rawstreamers = []

        for (start, stop), streamer in zip(tlist.byte_ranges, tlist, strict=True):
            if isinstance(streamer, uproot.streamers.Model_TStreamerInfo):
                rawstreamers.append(
                    RawStreamerInfo(
                        location + start,
                        uncompressed.raw_data[start:stop].tobytes(),
                        streamer.name,
                        streamer.class_version,
                    )
                )

            elif isinstance(streamer, uproot.models.TList.Model_TList):
                rawstreamers.append(
                    RawTListOfStrings(
                        location + start,
                        uncompressed.raw_data[start:stop].tobytes(),
                    )
                )

        return (
            TListOfStreamers(
                key.compressed_bytes,
                key,
                header,
                rawstreamers,
                freesegments,
            ),
            tlist,
        )


class _ReadForUpdate:
    def __init__(self, file_path, uuid, get_chunk=None, tlist_of_streamers=None):
        self.file_path = file_path
        self.uuid = uuid
        self.object_cache = None
        self._get_chunk = get_chunk
        self._tlist_of_streamers = tlist_of_streamers
        self._custom_classes = None
        self.source = self

    @property
    def detached(self):
        return self

    def chunk(self, start, stop):
        return self._get_chunk(start, stop)

    def class_named(self, classname, version=None):
        return uproot.reading.ReadOnlyFile.class_named(self, classname, version=version)

    def streamers_named(self, classname):
        if self._tlist_of_streamers is None:
            return []
        else:
            return [x for x in self._tlist_of_streamers if x.name == classname]

    def streamer_named(self, classname, version="max"):
        out = None
        if self._tlist_of_streamers is not None:
            for x in self._tlist_of_streamers:
                if x.name == classname:
                    if version == "max":
                        if out is None or x.class_version > out.class_version:
                            out = x
                    elif version == "min":
                        if out is None or x.class_version < out.class_version:
                            out = x
                    elif x.class_version == version:
                        return x
        return out

    @property
    def custom_classes(self):
        return self._custom_classes


class DirectoryData(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing the data part of a ROOT
    TDirectory.

    TDirectory (and TFileDirectory) objects have two parts: a fixed size header that
    does not move, providing a stable reference point, and a list of TKeys that would
    need to be rewritten if it grows beyond its initial allocation. This "data" part
    is the list of TKeys.

    See `ROOT TDirectory specification <https://github.com/root-project/root/blob/master/io/doc/TFile/tdirectory.md>`__.
    """

    def __init__(self, location, allocation, keys):
        super().__init__(location, allocation)
        self._dirty_keys = []
        self._dirty_keys_start = None

        self._keys_num_bytes = sum(x.allocation for x in keys)
        self._keys_by_name = {}
        for key in keys:
            self._index_key(key)
        self._keys = keys

    def __repr__(self):
        return (
            f"{type(self).__name__}({self._location}, {self._allocation}, {self._keys})"
        )

    @property
    def allocation(self):
        if self._allocation is None:
            self._allocation = self.num_bytes
        return self._allocation

    @allocation.setter
    def allocation(self, value):
        if self._allocation != value:
            self._file_dirty = True
            self._allocation = value

    def next_cycle(self, name):
        cycle = 1
        for key in self._keys_by_name.get(name, []):
            cycle = max(cycle, key.cycle + 1)
        return cycle

    def _index_key(self, key):
        self._keys_by_name.setdefault(key.name.string, []).append(key)

    def _unindex_key(self, key):
        keys = self._keys_by_name[key.name.string]
        keys[:] = [k for k in keys if k is not key]
        if not keys:
            del self._keys_by_name[key.name.string]

    def add_key(self, key, incremental=False):
        if incremental:
            self._dirty_keys.append(key)
        else:
            self._file_dirty = True

        self._keys_num_bytes += key.allocation
        self._index_key(key)
        self._keys.append(key)

    def replace_key(self, key):
        self._file_dirty = True
        old = self.get_key(key.name.string, key.cycle)
        if old is None:
            raise AssertionError(f"no key {key.name.string};{key.cycle} to replace")
        self._keys_num_bytes += key.allocation - old.allocation
        self._keys[self._keys.index(old)] = key
        self._unindex_key(old)
        self._index_key(key)

    def remove_key(self, key):
        self._file_dirty = True
        old = self.get_key(key.name.string, key.cycle)
        if old is None:
            raise AssertionError(f"no key {key.name.string};{key.cycle} to remove")
        self._keys_num_bytes -= old.allocation
        self._keys.remove(old)
        self._unindex_key(old)

    @property
    def num_keys(self):
        return len(self._keys)

    def haskey(self, name):
        return name in self._keys_by_name

    def get_key(self, name, cycle=None):
        keys = self._keys_by_name.get(name, [])
        if cycle is None:
            if len(keys) == 0:
                return None
            else:
                return max(keys, key=lambda x: x.cycle)
        else:
            for key in keys:
                if key.cycle == cycle:
                    return key
            else:
                return None

    @property
    def key_names(self):
        return [x.name.string for x in self._keys]

    @property
    def key_triples(self):
        return [(x.name.string, x.cycle, x.classname.string) for x in self._keys]

    @property
    def dir_keys(self):
        return [
            x
            for x in self._keys
            if x.classname.string == "TDirectory"
            or x.classname.string == "TDirectoryFile"
        ]

    @property
    def dir_names(self):
        return [x.name.string for x in self.dir_keys]

    def classname_of(self, name):
        keys = self._keys_by_name.get(name, [])
        if len(keys) == 0:
            return None
        else:
            return max(keys, key=lambda x: x.cycle).classname.string

    @property
    def num_bytes(self):
        return uproot.reading._directory_format_num_keys.size + self._keys_num_bytes

    @property
    def next_location(self):
        return self._location + self.num_bytes

    def write(self, sink):
        if self._file_dirty:
            if self._location is None:
                raise RuntimeError(
                    "can't write object because location is unknown:\n\n    "
                    + repr(self)
                )

            tmp = self.serialize()
            self._dirty_keys_start = self._location + len(tmp)

            sink.write(self._location, tmp)
            self._file_dirty = False

        else:
            num = uproot.reading._directory_format_num_keys.pack(len(self._keys))
            sink.write(self._location, num)

            for key in self._dirty_keys:
                copied = key.copy_to(self._dirty_keys_start)
                copied.location = self._dirty_keys_start
                copied.write(sink)
                self._dirty_keys_start += copied.num_bytes

        self._dirty_keys = []

    def serialize(self):
        out = [uproot.reading._directory_format_num_keys.pack(len(self._keys))]
        for key in self._keys:
            out.append(key.serialize())
        return b"".join(out)

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path):
        (num_keys,) = uproot.reading._directory_format_num_keys.unpack(raw_bytes[:4])
        position = location + 4

        keys = []
        for _ in range(num_keys):
            keys.append(
                Key.deserialize(
                    raw_bytes[position - location :],
                    position,
                    in_path,
                    is_directory_key=True,
                )
            )
            position += keys[-1].num_bytes

        return DirectoryData(location, len(raw_bytes), keys)


class DirectoryHeader(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing the header part of a ROOT
    TDirectory.

    TDirectory (and TFileDirectory) objects have two parts: a fixed size header that
    does not move, providing a stable reference point, and a list of TKeys that would
    need to be rewritten if it grows beyond its initial allocation. This is the "header"
    part.

    See `ROOT TDirectory specification <https://github.com/root-project/root/blob/master/io/doc/TFile/tdirectory.md>`__.
    """

    class_version = 5

    def __init__(
        self,
        location,
        begin_location,
        begin_num_bytes,
        data_location,
        data_num_bytes,
        parent_location,
        uuid,
    ):
        super().__init__(
            location, uproot.reading._directory_format_big.size + 2 + len(uuid.bytes)
        )
        self._begin_location = begin_location
        self._begin_num_bytes = begin_num_bytes
        self._data_location = data_location
        self._data_num_bytes = data_num_bytes
        self._parent_location = parent_location
        self._uuid = uuid
        self._created_on = datetime.datetime.now()
        self._modified_on = self._created_on

    def __repr__(self):
        return f"{type(self).__name__}({self._location}, {self._begin_location}, {self._begin_num_bytes}, {self._data_location}, {self._data_num_bytes}, {self._parent_location}, {self._uuid!r})"

    @property
    def begin_location(self):
        return self._begin_location

    @begin_location.setter
    def begin_location(self, value):
        if self._begin_location != value:
            self._file_dirty = True
            self._begin_location = value

    @property
    def begin_num_bytes(self):
        return self._begin_num_bytes

    @begin_num_bytes.setter
    def begin_num_bytes(self, value):
        if self._begin_num_bytes != value:
            self._file_dirty = True
            self._begin_num_bytes = value

    @property
    def data_location(self):
        return self._data_location

    @data_location.setter
    def data_location(self, value):
        if self._data_location != value:
            self._file_dirty = True
            self._data_location = value

    @property
    def data_num_bytes(self):
        return self._data_num_bytes

    @data_num_bytes.setter
    def data_num_bytes(self, value):
        if self._data_num_bytes != value:
            self._file_dirty = True
            self._data_num_bytes = value

    @property
    def parent_location(self):
        return self._parent_location

    @property
    def uuid(self):
        return self._uuid

    @property
    def modified_on(self):
        return self._modified_on

    @modified_on.setter
    def modified_on(self, value):
        if self._modified_on != value:
            self._file_dirty = True
            self._modified_on = value

    @property
    def big(self):
        return (
            self._begin_location >= uproot.const.kStartBigFile
            or self._data_location >= uproot.const.kStartBigFile
            or self._parent_location >= uproot.const.kStartBigFile
        )

    @property
    def num_bytes(self):
        if self.big:
            return uproot.reading._directory_format_big.size
        else:
            return uproot.reading._directory_format_small.size

    def serialize(self):
        if self.big:
            format = uproot.reading._directory_format_big
            version = self.class_version + 1000
            extra = b""
        else:
            format = uproot.reading._directory_format_small
            version = self.class_version
            extra = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        return (
            format.pack(
                version,  # fVersion
                uproot._util.datetime_to_code(self._created_on),  # fDatimeC
                uproot._util.datetime_to_code(self._modified_on),  # fDatimeM
                self._data_num_bytes,  # fNbytesKeys
                self._begin_num_bytes,  # fNbytesName
                self._begin_location,  # fSeekDir
                self._parent_location,  # fSeekParent
                self._data_location,  # fSeekKeys
            )
            + b"\x00\x01"  # TUUID version 1
            + self._uuid.bytes
            + extra
        )

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path):
        (
            version,
            fDatimeC,
            fDatimeM,
            fNbytesKeys,
            fNbytesName,
            fSeekDir,
            fSeekParent,
            fSeekKeys,
        ) = uproot.reading._directory_format_small.unpack(
            raw_bytes[: uproot.reading._directory_format_small.size]
        )
        position = location + uproot.reading._directory_format_small.size

        if version >= 1000:
            (
                version,
                fDatimeC,
                fDatimeM,
                fNbytesKeys,
                fNbytesName,
                fSeekDir,
                fSeekParent,
                fSeekKeys,
            ) = uproot.reading._directory_format_big.unpack(
                raw_bytes[: uproot.reading._directory_format_big.size]
            )
            version -= 1000
            position = location + uproot.reading._directory_format_big.size

        if version != cls.class_version:
            raise ValueError(
                f"Uproot can't read TDirectory version {version} for writing, only version {cls.class_version}{in_path}"
            )

        # TUUID version 1
        assert raw_bytes[position - location : position - location + 2] == b"\x00\x01"
        uuid_bytes = raw_bytes[position - location + 2 : position - location + 18]

        out = DirectoryHeader(
            location,
            fSeekDir,  # begin_location
            fNbytesName,  # begin_num_bytes
            fSeekKeys,  # data_location
            fNbytesKeys,  # data_num_bytes
            fSeekParent,  # parent_location
            uuid.UUID(bytes=uuid_bytes),
        )
        out._created_on = uproot._util.code_to_datetime(fDatimeC)
        out._modified_on = uproot._util.code_to_datetime(fDatimeM)
        return out


class Directory(CascadeNode):
    """
    A :doc:`uproot.writing._cascade.CascadeNode` for writing a TDirectory.

    TDirectory (and TFileDirectory) objects have two parts: a fixed size header that
    does not move, providing a stable reference point, and a list of TKeys that would
    need to be rewritten if it grows beyond its initial allocation. This node manages
    both parts.

    See `ROOT TDirectory specification <https://github.com/root-project/root/blob/master/io/doc/TFile/tdirectory.md>`__.

    This class is an abstract superclass; the concrete subclasses are
    :doc:`uproot.writing._cascade.RootDirectory` and :doc:`uproot.writing._cascade.SubDirectory`.
    """

    def _reallocate_data(self, new_data_size):
        original_start = self._datakey.location
        original_stop = original_start + self._datakey.num_bytes + self._data.allocation

        self._datakey.location = None  # let it assume the key might be big
        requested_num_bytes = self._datakey.num_bytes + new_data_size
        self._datakey.location = self._freesegments.allocate(requested_num_bytes)
        self._header.data_location = self._datakey.location
        self._data.location = self._datakey.location + self._datakey.num_bytes
        might_be_slightly_more = requested_num_bytes - self._datakey.num_bytes
        self._data.allocation = might_be_slightly_more

        self._freesegments.release(original_start, original_stop)

    def add_object(
        self,
        sink,
        classname,
        name,
        title,
        raw_data,
        uncompressed_bytes,
        replaces=None,
        big=None,
    ):
        cycle = self._data.next_cycle(name) if replaces is None else replaces.cycle

        classname_asbytes = classname.encode(errors="surrogateescape")
        name_asbytes = name.encode(errors="surrogateescape")
        title_asbytes = title.encode(errors="surrogateescape")
        strings_size = 0
        strings_size += (1 if len(classname_asbytes) < 255 else 5) + len(
            classname_asbytes
        )
        strings_size += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        strings_size += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        parent_location = self._key.location  # FIXME: is this correct?

        location = None
        if not big and parent_location < uproot.const.kStartBigFile:
            requested_bytes = (
                uproot.reading._key_format_small.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=True)
            position = location + uproot.reading._key_format_small.size
            if location < uproot.const.kStartBigFile:
                self._freesegments.allocate(requested_bytes, dry_run=False)
            else:
                location = None

        if location is None:
            requested_bytes = (
                uproot.reading._key_format_big.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=False)
            position = location + uproot.reading._key_format_big.size

        writable_classname = String(position, classname)
        position += writable_classname.num_bytes

        writable_name = String(position, name)
        position += writable_name.num_bytes

        writable_title = String(position, title)
        position += writable_title.num_bytes

        key = Key(
            location,
            uncompressed_bytes,
            len(raw_data),
            writable_classname,
            writable_name,
            writable_title,
            cycle,
            parent_location,
            location,
            big=big,
        )

        if replaces is None:
            next_key = key.copy_to(self._data.next_location)
            if self._data.num_bytes + next_key.num_bytes > self._data.allocation:
                requested_num_bytes = math.ceil(
                    1.5 * (self._data.allocation + next_key.num_bytes + 8)
                )
                self._reallocate_data(requested_num_bytes)
                next_key = key.copy_to(self._data.next_location)
            next_key._location = self._data.next_location
            self._data.add_key(next_key, incremental=True)

        else:
            original_key = self._data.get_key(replaces.name.string, replaces.cycle)
            assert original_key is not None
            new_key = key.copy_to(original_key.location)
            if (
                self._data.num_bytes + new_key.num_bytes - original_key.num_bytes
                > self._data.allocation
            ):
                requested_num_bytes = math.ceil(
                    1.5 * (self._data.allocation + new_key.num_bytes + 8)
                )
                self._reallocate_data(requested_num_bytes)
                original_key = self._data.get_key(replaces.name.string, replaces.cycle)
                assert original_key is not None
                new_key = key.copy_to(original_key.location)
            self._data.replace_key(new_key)

        self._header.modified_on = datetime.datetime.now()

        key.write(sink)
        sink.write(location + key.num_bytes, raw_data)
        self.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return key

    def add_directory(self, sink, name, initial_directory_bytes, uuid, flush=True):
        cycle = self._data.next_cycle(name)

        subdirectory_key = Key(
            None,
            None,
            None,
            String(None, "TDirectory"),
            String(None, name),
            String(None, name),
            cycle,
            self._key.location,
            None,
        )
        subdirectory_header = DirectoryHeader(
            None,
            None,
            None,
            None,
            None,
            self._key.location,
            uuid,
        )
        subdirectory_datakey = Key(
            None,
            None,
            None,
            String(None, "TDirectory"),
            String(None, name),
            String(None, name),
            cycle,
            self._key.location,
            None,
        )

        requested_num_bytes = (
            subdirectory_key.num_bytes
            + subdirectory_header.allocation
            + subdirectory_datakey.num_bytes
            + initial_directory_bytes
        )
        subdirectory_key.location = self._freesegments.allocate(requested_num_bytes)
        subdirectory_datakey.location = (
            subdirectory_key.location
            + subdirectory_key.num_bytes
            + subdirectory_header.allocation
        )
        might_be_slightly_more = requested_num_bytes - (
            subdirectory_key.num_bytes  # because Key.num_bytes depends on location
            + subdirectory_header.allocation
            + subdirectory_datakey.num_bytes  # including this Key, too
        )
        subdirectory_data = DirectoryData(None, might_be_slightly_more, [])

        subdirectory = SubDirectory(
            subdirectory_key,
            subdirectory_header,
            subdirectory_datakey,
            subdirectory_data,
            self,
            self._freesegments,
        )

        subdirectory_key.uncompressed_bytes = subdirectory_header.allocation
        subdirectory_key.compressed_bytes = subdirectory_key.uncompressed_bytes

        next_key = subdirectory_key.copy_to(self._data.next_location)
        if self._data.num_bytes + next_key.num_bytes > self._data.allocation:
            self._reallocate_data(
                math.ceil(1.5 * (self._data.allocation + next_key.num_bytes + 8))
            )
            next_key = subdirectory_key.copy_to(self._data.next_location)
        next_key._location = self._data.next_location
        self._data.add_key(next_key, incremental=True)
        self._header.modified_on = datetime.datetime.now()

        if flush:
            subdirectory.write(sink)
            self.write(sink)
            sink.set_file_length(self._freesegments.fileheader.end)
            sink.flush()

        return subdirectory

    def add_tree(
        self,
        sink,
        name,
        title,
        branch_types,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        import uproot.writing._cascadetree

        tree = uproot.writing._cascadetree.Tree(
            self,
            name,
            title,
            branch_types,
            self._freesegments,
            counter_name,
            field_name,
            initial_basket_capacity,
            resize_factor,
        )
        tree.write_anew(sink)
        return tree

    def add_rntuple(self, sink, name, description, akform):
        import uproot.writing._cascadentuple

        anchor = uproot.writing._cascadentuple.NTuple_Anchor(
            None,
            *uproot.const.rntuple_version_for_writing,
            None,
            None,
            None,
            None,
            None,
            None,
            0,  # TODO: Fix this
        )

        header = uproot.writing._cascadentuple.NTuple_Header(
            None, name, description, akform
        )

        footer = uproot.writing._cascadentuple.NTuple_Footer(None, header._checksum)

        ntuple = uproot.writing._cascadentuple.NTuple(
            self, akform, self._freesegments, header, footer, [], anchor
        )

        ntuple.write(sink)
        return ntuple


class RootDirectory(Directory):
    """
    A :doc:`uproot.writing._cascade.Directory` for the root (first) directory in a file.

    Subdirectories are :doc:`uproot.writing._cascade.SubDirectory` objects. Two classes
    are needed because their structure is slightly different.
    """

    def __init__(self, key, name, title, header, datakey, data, freesegments):
        super().__init__(freesegments, datakey, data, key, name, title, header)
        self._key = key
        self._name = name
        self._title = title
        self._header = header
        self._datakey = datakey
        self._data = data
        self._freesegments = freesegments

    def __repr__(self):
        return f"{type(self).__name__}({self._key}, {self._name}, {self._title}, {self._header}, {self._datakey}, {self._data}, {self._freesegments})"

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def header(self):
        return self._header

    @property
    def datakey(self):
        return self._datakey

    @property
    def data(self):
        return self._data

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def location(self):
        return self._key.location

    @property
    def begin_num_bytes(self):
        return self._key.allocation + self._name.allocation + self._title.allocation

    def write(self, sink):
        self._key.uncompressed_bytes = (
            self._name.allocation + self._title.allocation + self._header.allocation
        )
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._name.location = self._key.location + self._key.allocation
        self._title.location = self._name.location + self._name.allocation
        self._header.location = self._title.location + self._title.allocation
        self._header.begin_location = self._key.location
        self._header.begin_num_bytes = self.begin_num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.allocation + self._data.allocation
        self._datakey.uncompressed_bytes = self._data.allocation
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        self._data.location = self._datakey.location + self._datakey.allocation
        self._freesegments.fileheader.begin_num_bytes = self.begin_num_bytes
        super().write(sink)


class SubDirectory(Directory):
    """
    A :doc:`uproot.writing._cascade.Directory` for any subdirectories in a file.

    The root (first) directory is a :doc:`uproot.writing._cascade.RootDirectory` object.
    Two classes are needed because their structure is slightly different.
    """

    def __init__(self, key, header, datakey, data, parent, freesegments):
        super().__init__(freesegments, datakey, data, key, header, parent)
        self._key = key
        self._header = header
        self._datakey = datakey
        self._data = data
        self._parent = parent
        self._freesegments = freesegments

    def __repr__(self):
        return f"{type(self).__name__}({self._key}, {self._header}, {self._datakey}, {self._data}, {self._parent}, {self._freesegments})"

    @property
    def key(self):
        return self._key

    @property
    def header(self):
        return self._header

    @property
    def datakey(self):
        return self._datakey

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def location(self):
        return self._key.location

    def write(self, sink):
        self._key.uncompressed_bytes = self._header.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._header.location = self._key.location + self._key.allocation
        self._header.begin_location = self._key.location
        self._header.begin_num_bytes = self._key.num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.allocation + self._data.allocation
        self._datakey.uncompressed_bytes = self._data.allocation
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        self._data.location = self._datakey.location + self._datakey.allocation
        super().write(sink)


class FileHeader(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing the header of a ROOT file.

    The header is in the first 100 bytes of the ROOT file, starting with the magic
    ``b'root'`` sequence. It contains a pointer to the root (first) directory (:doc:`uproot.writing._cascade.RootDirectory`),
    the FreeSegments (:doc:`uproot.writing._cascade.FreeSegments`) record, the list of
    TStreamerInfos (:doc:`uproot.writing._cascade.TListOfStreamers`), preferred compression
    codec, UUID, and the version of ROOT used to make the file.

    Uproot declares its ROOT version to be 6.24/00 and versions of all self-generated
    classes are consistent with that version.

    See `ROOT header specification <https://github.com/root-project/root/blob/master/io/doc/TFile/header.md>`__.
    """

    magic = b"root"
    class_version = 62400  # ROOT 6.24/00 is our model

    def __init__(
        self,
        end,
        free_location,
        free_num_bytes,
        free_num_slices,
        begin_num_bytes,
        compression,
        info_location,
        info_num_bytes,
        uuid,
    ):
        super().__init__(0, 100)
        self._end = end
        self._free_location = free_location
        self._free_num_bytes = free_num_bytes
        self._free_num_slices = free_num_slices
        self._begin_num_bytes = begin_num_bytes
        self._compression = compression
        self._info_location = info_location
        self._info_num_bytes = info_num_bytes
        self._uuid = uuid
        self._version = None
        self._begin = 100

    def __repr__(self):
        return f"{type(self).__name__}({self._end}, {self._free_location}, {self._free_num_bytes}, {self._free_num_slices}, {self._begin_num_bytes}, {self._compression!r}, {self._info_location}, {self._info_num_bytes}, {self._uuid!r})"

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if self._end != value:
            self._file_dirty = True
            self._end = value

    @property
    def free_location(self):
        return self._free_location

    @free_location.setter
    def free_location(self, value):
        if self._free_location != value:
            self._file_dirty = True
            self._free_location = value

    @property
    def free_num_bytes(self):
        return self._free_num_bytes

    @free_num_bytes.setter
    def free_num_bytes(self, value):
        if self._free_num_bytes != value:
            self._file_dirty = True
            self._free_num_bytes = value

    @property
    def free_num_slices(self):
        return self._free_num_slices

    @free_num_slices.setter
    def free_num_slices(self, value):
        if self._free_num_slices != value:
            self._file_dirty = True
            self._free_num_slices = value

    @property
    def begin_num_bytes(self):
        return self._begin_num_bytes

    @begin_num_bytes.setter
    def begin_num_bytes(self, value):
        if self._begin_num_bytes != value:
            self._file_dirty = True
            self._begin_num_bytes = value

    @property
    def compression(self):
        return self._compression

    @compression.setter
    def compression(self, value):
        if self._compression != value:
            self._file_dirty = True
        self._compression = value

    @property
    def info_location(self):
        return self._info_location

    @info_location.setter
    def info_location(self, value):
        if self._info_location != value:
            self._file_dirty = True
            self._info_location = value

    @property
    def info_num_bytes(self):
        return self._info_num_bytes

    @info_num_bytes.setter
    def info_num_bytes(self, value):
        if self._info_num_bytes != value:
            self._file_dirty = True
            self._info_num_bytes = value

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        if self._uuid != value:
            self._file_dirty = True
        self._uuid = value

    @property
    def version(self):
        if self._version is None:
            return self.class_version
        else:
            return self._version

    @property
    def begin(self):
        return self._begin

    @property
    def big(self):
        return (
            self._end is None
            or self._end >= uproot.const.kStartBigFile
            or self._free_location >= uproot.const.kStartBigFile
            or self._info_location >= uproot.const.kStartBigFile
        )

    @property
    def num_bytes(self):
        if self.big:
            return uproot.reading._file_header_fields_big.size
        else:
            return uproot.reading._file_header_fields_small.size

    def serialize(self):
        if self.big:
            format = uproot.reading._file_header_fields_big
            version = self.version + 1000000
            units = 8
        else:
            format = uproot.reading._file_header_fields_small
            version = self.version
            units = 4

        if self._compression is None:
            fCompress = uproot.compression.ZLIB(0).code
        else:
            fCompress = self._compression.code

        return format.pack(
            self.magic,
            version,  # fVersion
            self.begin,  # fBEGIN
            self._end,  # fEND
            self._free_location,  # fSeekFree
            self._free_num_bytes,  # fNbytesFree
            self._free_num_slices + 1,  # nfree
            self._begin_num_bytes,  # fNbytesName
            units,  # fUnits
            fCompress,
            self._info_location,  # fSeekInfo
            self._info_num_bytes,  # fNbytesInfo
            1,  # TUUID version
            self._uuid.bytes,  # fUUID
        )

    @classmethod
    def deserialize(cls, raw_bytes, location):
        (
            _magic,
            version,
            begin,
            end,
            free_location,
            free_num_bytes,
            free_num_slices_plus_1,
            begin_num_bytes,
            units,
            compression_code,
            info_location,
            info_num_bytes,
            uuid_version,
            uuid_bytes,
        ) = uproot.reading._file_header_fields_small.unpack(
            raw_bytes[: uproot.reading._file_header_fields_small.size]
        )
        if version >= 1000000:
            (
                _magic,
                version,
                begin,
                end,
                free_location,
                free_num_bytes,
                free_num_slices_plus_1,
                begin_num_bytes,
                units,
                compression_code,
                info_location,
                info_num_bytes,
                uuid_version,
                uuid_bytes,
            ) = uproot.reading._file_header_fields_big.unpack(raw_bytes)
            assert units == 8
            outversion = version - 1000000
        else:
            assert units == 4
            outversion = version

        assert begin >= uproot.reading._file_header_fields_small.size
        assert free_location >= 0
        assert free_num_bytes >= 0
        assert free_num_slices_plus_1 >= 1
        assert begin_num_bytes >= 0
        assert compression_code >= 0
        assert info_location >= 0
        assert info_num_bytes >= 0
        assert 0 <= uuid_version <= 8

        out = FileHeader(
            end,
            free_location,
            free_num_bytes,
            free_num_slices_plus_1 - 1,
            begin_num_bytes,
            uproot.compression.Compression.from_code(compression_code),
            info_location,
            info_num_bytes,
            uuid.UUID(bytes=uuid_bytes),
        )
        out._version = outversion
        out._begin = begin
        return out


class CascadingFile:
    """
    An object that represents an entire file, the root of the cascading-node tree.

    See :doc:`uproot.writing._cascade` for a general overview.
    """

    def __init__(
        self,
        fileheader,
        streamers,
        freesegments,
        rootdirectory,
        tlist_of_streamers,
    ):
        self._fileheader = fileheader
        self._streamers = streamers
        self._freesegments = freesegments
        self._rootdirectory = rootdirectory
        self._tlist_of_streamers = tlist_of_streamers

    def __repr__(self):
        return f"{type(self).__name__}({self._fileheader}, {self._streamers}, {self._freesegments}, {self._rootdirectory})"

    @property
    def fileheader(self):
        return self._fileheader

    @property
    def streamers(self):
        return self._streamers

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def rootdirectory(self):
        return self._rootdirectory

    @property
    def tlist_of_streamers(self):
        return self._tlist_of_streamers


def create_empty(
    sink,
    compression,
    initial_directory_bytes,
    initial_streamers_bytes,
    uuid_function,
):
    """
    Function to create an empty ROOT file, returning a :doc:`uproot.writing._cascade.CascadingFile`.
    """
    filename = sink.file_path
    filename = "dynamic.root" if filename is None else os.path.split(filename)[-1]
    if len(filename) >= 256:
        raise ValueError("ROOT file names must be less than 256 bytes")

    fileheader = FileHeader(
        None,
        None,
        None,
        None,
        None,
        compression,
        None,
        None,
        uuid_function(),
    )

    freesegments_key = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        fileheader.begin,
        None,
    )
    freesegments_data = FreeSegmentsData(None, (), None)
    freesegments = FreeSegments(freesegments_key, freesegments_data, fileheader)

    streamers_key = Key(
        None,
        None,
        None,
        String(None, "TList"),
        String(None, "StreamerInfo"),
        String(None, "Doubly linked list"),
        1,
        fileheader.begin,
        None,
    )
    streamers_header = TListHeader(None, None, None)
    streamers = TListOfStreamers(
        initial_streamers_bytes, streamers_key, streamers_header, [], freesegments
    )

    directory_key = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        0,
        None,
    )
    directory_name = String(None, filename)
    directory_title = String(None, "")
    directory_header = DirectoryHeader(
        None, fileheader.begin, None, None, None, 0, uuid_function()
    )
    directory_datakey = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        fileheader.begin,
        None,
    )
    directory_data = DirectoryData(None, initial_directory_bytes, [])
    rootdirectory = RootDirectory(
        directory_key,
        directory_name,
        directory_title,
        directory_header,
        directory_datakey,
        directory_data,
        freesegments,
    )

    directory_key.location = fileheader.begin
    streamers_key.location = (
        directory_key.location
        + directory_key.allocation
        + directory_name.allocation
        + directory_title.allocation
        + directory_header.allocation
    )
    directory_datakey.location = (
        streamers_key.location + streamers_key.allocation + streamers.allocation
    )
    directory_data.location = directory_datakey.location + directory_datakey.allocation
    freesegments_key.location = directory_data.location + directory_data.allocation
    freesegments_data.end = (
        freesegments_key.location
        + freesegments_key.allocation
        + freesegments_data.allocation
    )
    fileheader.info_location = streamers_key.location
    fileheader.info_num_bytes = streamers_key.allocation + streamers.allocation

    rootdirectory.write(sink)
    streamers.write(sink)

    return CascadingFile(fileheader, streamers, freesegments, rootdirectory, None)


def update_existing(sink, initial_directory_bytes, uuid_function):
    """
    Function to derive a :doc:`uproot.writing._cascade.CascadingFile` from an existing
    file.

    This function *does* write to the file, overwriting the TStreamerInfo.
    """
    raw_bytes = sink.read(
        0,
        uproot.reading._file_header_fields_big.size,
        insist=uproot.reading._file_header_fields_small.size,
    )
    if raw_bytes[:4] != b"root":
        raise ValueError(
            f"not a ROOT file: first four bytes are {raw_bytes[:4]!r}{sink.in_path}"
        )
    fileheader = FileHeader.deserialize(raw_bytes, 0)

    raw_bytes = sink.read(fileheader.free_location, fileheader.free_num_bytes)
    freesegments_key = Key.deserialize(
        raw_bytes, fileheader.free_location, sink.in_path
    )

    freesegments_data = FreeSegmentsData.deserialize(
        raw_bytes[freesegments_key.num_bytes :],
        fileheader.free_location + freesegments_key.num_bytes,
        fileheader.free_num_bytes - freesegments_key.num_bytes,
        fileheader.free_num_slices,
        sink.in_path,
    )

    freesegments = FreeSegments(freesegments_key, freesegments_data, fileheader)

    raw_bytes = sink.read(fileheader.info_location, fileheader.info_num_bytes)
    streamers_key = Key.deserialize(raw_bytes, fileheader.info_location, sink.in_path)
    streamers, tlist_of_streamers = TListOfStreamers.deserialize(
        raw_bytes[streamers_key.num_bytes :],
        fileheader.info_location + streamers_key.num_bytes,
        streamers_key,
        freesegments,
        sink.file_path,
        fileheader.uuid,
    )

    raw_bytes = sink.read(
        fileheader.begin,
        fileheader.begin_num_bytes + uproot.reading._directory_format_big.size + 18,
    )
    directory_key = Key.deserialize(raw_bytes, fileheader.begin, sink.in_path)
    position = fileheader.begin + directory_key.num_bytes

    directory_name, position = String.deserialize(
        raw_bytes[position - fileheader.begin :], position
    )
    directory_title, position = String.deserialize(
        raw_bytes[position - fileheader.begin :], position
    )

    assert fileheader.begin_num_bytes == position - fileheader.begin

    directory_header = DirectoryHeader.deserialize(
        raw_bytes[position - fileheader.begin :], position, sink.in_path
    )
    assert directory_header.begin_location == fileheader.begin
    assert directory_header.begin_num_bytes == fileheader.begin_num_bytes
    assert directory_header.parent_location == 0

    raw_bytes = sink.read(
        directory_header.data_location, directory_header.data_num_bytes
    )
    directory_datakey = Key.deserialize(
        raw_bytes, directory_header.data_location, sink.in_path
    )
    directory_data = DirectoryData.deserialize(
        raw_bytes[directory_datakey.num_bytes :],
        directory_header.data_location + directory_datakey.num_bytes,
        sink.in_path,
    )

    rootdirectory = RootDirectory(
        directory_key,
        directory_name,
        directory_title,
        directory_header,
        directory_datakey,
        directory_data,
        freesegments,
    )

    streamers.write(sink)
    sink.flush()

    return CascadingFile(
        fileheader, streamers, freesegments, rootdirectory, tlist_of_streamers
    )
