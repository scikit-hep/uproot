# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module provides an interface between Uproot and PyROOT.

PyROOT is a *complete* set of ROOT bindings generated automatically from ROOT's
reflected C++ code. Uproot is a pure Python reimplementation of ROOT I/O, with
incomplete coverage of ROOT's suite of classes, but without an install-time
dependence on ROOT.

The only point of contact between Uproot and PyROOT is that they both recognize
the same *serialized* form of ROOT objects, so conversions in this module proceed
by serializing and deserializing the objects through a TMessage. This makes it
possible to

- convert any PyROOT object into its Uproot :doc:`uproot.model.Model` *if Uproot can read it* (which is possible for most classes)
- convert any Uproot :doc:`uproot.model.Model` into the corresponding PyROOT object *if Uproot can write it* (which is considerably more constrained; mostly just histograms).

This module also makes it possible for PyROOT objects to be added to ROOT files
that Uproot is writing (regardless of whether Uproot could read such objects).
"""

from __future__ import annotations

import threading
import uuid

import numpy

import uproot


def to_pyroot(obj, name=None):
    """
    Args:
        obj (:doc:`uproot.model.Model` or other object that Uproot can write):
            The object to convert, such as an Uproot model, a boost-histogram or
            hist histogram, or a NumPy histogram.
        name (str or None): A name for the new PyROOT object.

    Converts an :doc:`uproot.model.Model` (or any object that
    :doc:`uproot.writing.identify.to_writable` recognizes) into a PyROOT object
    *if it is writable by Uproot*. A minority of Uproot models are writable,
    mostly just histograms. Writability is necessary for conversion to PyROOT
    because it is serialized through a ROOT TMessage.
    """
    import ROOT

    if type(obj).__module__ == "cppyy.gbl":
        # already a PyROOT object
        return obj.Clone() if name is None else obj.Clone(name)

    # Objects read from a file have the class versions of the file that wrote
    # them, which Uproot may not be able to serialize; convert them to the
    # versions that Uproot writes, as writing them to a file does. This also
    # converts other objects, such as hist histograms.
    writable = uproot.to_writable(obj)

    with to_pyroot._lock:
        if to_pyroot._Uproot_FromTMessage is None:
            ROOT.gInterpreter.Declare("""
#include "TMessage.h"
#include "TStreamerInfo.h"

class _Uproot_FromTMessage : public TMessage {
public:
    _Uproot_FromTMessage(void* buffer, Int_t size): TMessage(buffer, size) { }
};

void _uproot_register_streamer_info(void* buffer, Int_t size) {
    _Uproot_FromTMessage message(buffer, size);
    TObject* obj = message.ReadObject(message.GetClass());
    message.DetachBuffer();
    TStreamerInfo* info = dynamic_cast<TStreamerInfo*>(obj);
    if (info == nullptr) {
        delete obj;
        return;
    }
    // as TFile::ReadStreamerInfo does: the TClass takes ownership of a new
    // StreamerInfo and marks a duplicate as deletable
    info->BuildCheck();
    if (info->TestBit(kCanDelete)) {
        delete info;
    }
}
""")
            to_pyroot._Uproot_FromTMessage = ROOT._Uproot_FromTMessage

        # A TMessage, unlike a file, carries no StreamerInfo, so ROOT can only read
        # the class versions it knows. Give it the StreamerInfo of the versions
        # that Uproot writes, as a file would.
        for rawstreamer in writable.class_rawstreamers or ():
            if not isinstance(rawstreamer, tuple):
                continue
            _, serialization, classname, class_version = rawstreamer
            key = (classname, class_version)
            if key in to_pyroot._registered_streamers:
                continue
            tclass = ROOT.TClass.GetClass(classname)
            if tclass and tclass.GetClassVersion() != class_version:
                buffer = _tmessage_buffer(ROOT, serialization)
                ROOT._uproot_register_streamer_info(buffer, len(buffer))
            to_pyroot._registered_streamers.add(key)

    buffer = _tmessage_buffer(
        ROOT, uproot.serialization.serialize_object_any(writable, name)
    )
    message = to_pyroot._Uproot_FromTMessage(buffer, len(buffer))
    try:
        return message.ReadObject(message.GetClass())
    finally:
        # The buffer belongs to NumPy, so the TMessage must not free it.
        message.DetachBuffer()


def _tmessage_buffer(ROOT, serialized):
    buffer = numpy.empty(len(serialized) + 8, numpy.uint8)
    buffer[:8].view(numpy.uint64)[0] = ROOT.kMESS_OBJECT
    buffer[8:] = numpy.frombuffer(serialized, numpy.uint8)
    return buffer


to_pyroot._Uproot_FromTMessage = None
to_pyroot._registered_streamers = set()
to_pyroot._lock = threading.Lock()


def pyroot_to_buffer(obj):
    """
    Args:
        obj (PyROOT object inheriting from TObject): PyROOT object to serialize.

    Serializes a PyROOT object into a NumPy array.
    """
    import ROOT

    with pyroot_to_buffer._lock:
        if pyroot_to_buffer._copy is None:
            ROOT.gInterpreter.Declare("""
void _uproot_copy_tmessage(TMessage& message, size_t out_addr) {
    memcpy((void*)out_addr, message.Buffer() + 8, message.Length() - 8);
}
""")
            pyroot_to_buffer._copy = ROOT._uproot_copy_tmessage

        message = ROOT.TMessage(ROOT.kMESS_OBJECT)
        message.SetCompressionLevel(0)
        message.WriteObject(obj)
        # TMessage prepends an 8-byte header (4-byte length + 4-byte kMESS_OBJECT
        # type tag) before the serialized object data; skip it in both the copy
        # offset and the output length.
        length = message.Length() - 8
        buffer = numpy.empty(length, numpy.uint8)
        pyroot_to_buffer._copy(message, buffer.ctypes.data)
        return buffer


pyroot_to_buffer._lock = threading.Lock()
pyroot_to_buffer._copy = None


def _serialize_in_key(obj, key_num_bytes):
    """
    Serializes a PyROOT object as ROOT does in a TKey of ``key_num_bytes``,
    which matters because references to other objects within it (e.g. axis labels
    or list items) are relative to the start of the key.
    """
    import ROOT

    with pyroot_to_buffer._lock:
        if _serialize_in_key._serialize is None:
            ROOT.gInterpreter.Declare("""
#include "TBufferFile.h"

Long64_t _uproot_serialize_in_key(TObject* obj, Int_t keylen, std::vector<char>& out) {
    // as TKey::TKey(const TObject*, ...) does, after writing the key itself
    TBufferFile buffer(TBuffer::kWrite, keylen + 10000);
    buffer.SetBufferOffset(keylen);
    buffer.MapObject(obj);
    obj->Streamer(buffer);
    out.assign(buffer.Buffer() + keylen, buffer.Buffer() + buffer.Length());
    return out.size();
}

void _uproot_copy_vector(std::vector<char>& data, size_t out_addr) {
    memcpy((void*)out_addr, data.data(), data.size());
}
""")
            _serialize_in_key._serialize = ROOT._uproot_serialize_in_key
            _serialize_in_key._copy = ROOT._uproot_copy_vector

        data = ROOT.std.vector("char")()
        buffer = numpy.empty(
            _serialize_in_key._serialize(obj, key_num_bytes, data), numpy.uint8
        )
        _serialize_in_key._copy(data, buffer.ctypes.data)
        return buffer


_serialize_in_key._serialize = None
_serialize_in_key._copy = None


class _GetStreamersOnce:
    _custom_classes = None
    _streamers = {}
    _parsed = {}
    _memfile_copy = None
    _lock = threading.Lock()

    def __init__(self, obj):
        self._obj = obj
        self._dumped = False

    def class_named(self, classname, version=None):
        return uproot.reading.ReadOnlyFile.class_named(self, classname, version)

    def streamers_named(self, classname):
        out = uproot.reading.ReadOnlyFile.streamers_named(self, classname)
        if len(out) == 0 and not self._dumped:
            # the StreamerInfo may have come from earlier objects of the same
            # class, which didn't contain this one (e.g. in fFunctions)
            self._dump()
            out = uproot.reading.ReadOnlyFile.streamers_named(self, classname)
        return out

    def streamer_named(self, classname, version):
        return uproot.reading.ReadOnlyFile.streamer_named(
            self, classname, version=version
        )

    @property
    def custom_classes(self):
        return self._custom_classes

    @property
    def file_path(self):
        return None

    class ArrayFile:
        def __init__(self, array):
            self.array = array
            self.current = 0

        def seek(self, position):
            self.current = position

        def read(self, num_bytes):
            position = self.current + num_bytes
            out = self.array[self.current : position]
            self.current = position
            return out

    @property
    def streamers(self):
        tclass = self._obj.IsA()
        obj_classname = tclass.GetName()
        obj_version = tclass.GetClassVersion()
        if self._streamers.get(obj_classname, {}).get(obj_version, None) is None:
            self._dump()
        return self._streamers

    def _dump(self):
        # gets (and returns) the StreamerInfo of this object and all it contains
        import ROOT

        with _GetStreamersOnce._lock:
            if _GetStreamersOnce._memfile_copy is None:
                ROOT.gInterpreter.Declare("""
#include "TDirectory.h"
#include "TMemFile.h"

void _uproot_dump_streamer_info(
    TObject* obj, std::vector<char>& out, std::vector<Long64_t>& info
) {
    TDirectory::TContext context;  // TMemFile would change gDirectory
    TMemFile memfile("noname.root", "new");
    memfile.SetCompressionLevel(0);
    memfile.WriteObjectAny(obj, obj->IsA(), "noname");
    memfile.WriteStreamerInfo();
    memfile.Close();
    info = {memfile.GetSeekInfo(), memfile.GetNbytesInfo()};
    out.resize(memfile.GetEND());
    memfile.CopyTo(out.data(), out.size());
}

void _uproot_copy_chars(std::vector<char>& data, size_t out_addr) {
    memcpy((void*)out_addr, data.data(), data.size());
}
""")
                _GetStreamersOnce._memfile_copy = (
                    ROOT._uproot_dump_streamer_info,
                    ROOT._uproot_copy_chars,
                )

            data = ROOT.std.vector("char")()
            info = ROOT.std.vector("Long64_t")()
            dump, copy = _GetStreamersOnce._memfile_copy
            dump(self._obj, data, info)
            buffer = numpy.empty(data.size(), numpy.uint8)
            copy(data, buffer.ctypes.data)

            # parsing the StreamerInfo record is the slow part, so reuse it for
            # the same record (from its data, after its key, which has a date)
            seek, nbytes = int(info[0]), int(info[1])
            keylen = int(buffer[seek + 14 : seek + 16].view(">i2")[0])
            record = buffer[seek + keylen : seek + nbytes].tobytes()
            dependencies = _GetStreamersOnce._parsed.get(record)
            if dependencies is None:
                dependencies = []
                with uproot.open(_GetStreamersOnce.ArrayFile(buffer)) as file:
                    for classname, versions in file.file.streamers.items():
                        if classname not in self._streamers:
                            self._streamers[classname] = {}
                        for version, streamerinfo in versions.items():
                            self._streamers[classname][version] = streamerinfo
                            dependencies.append(streamerinfo)
                _GetStreamersOnce._parsed[record] = dependencies

        self._dumped = True
        return dependencies


class _NoFile:
    def __init__(self):
        import ROOT

        self._file_path = ""
        self._options = {}
        self._fVersion = ROOT.gROOT.GetVersionInt()
        self._fBEGIN = 0
        self._fEND = 0
        self._fSeekFree = 0
        self._fNbytesFree = 0
        self._nfree = 0
        self._fNbytesName = 0
        self._fUnits = 0
        self._fCompress = 0
        self._fSeekInfo = 0
        self._fNbytesInfo = 0
        self._fUUID_version = 1
        self._fUUID = uuid.UUID("00000000-0000-0000-0000-000000000000")

    @property
    def streamers(self):
        return _GetStreamersOnce._streamers

    @property
    def custom_classes(self):
        return _GetStreamersOnce._custom_classes


def from_pyroot(obj):
    """
    Args:
        obj (PyROOT object inheriting from TObject): PyROOT object to convert to
            an Uproot :doc:`uproot.model.Model`.

    Converts a PyROOT object into its corresponding Uproot :doc:`uproot.model.Model`
    *if it is readable by Uproot*. Most ROOT classes are readable by Uproot. Readability
    is necessary for conversion from PyROOT because the object is serialized through a
    ROOT TMessage.
    """
    if obj.InheritsFrom("TTree"):
        raise TypeError(
            "TTrees can't be converted from PyROOT, since their data are in baskets "
            "in a file: write the TTree to a file and open it with uproot.open"
        )
    import ROOT

    if obj.IsA().GetState() == ROOT.TClass.kInterpreted:
        raise TypeError(
            f"ROOT can't serialize {obj.ClassName()}, which is an interpreted "
            "class without a dictionary, so it can't be converted"
        )
    buffer = pyroot_to_buffer(obj)
    chunk = uproot.source.chunk.Chunk.wrap(None, buffer)
    # references in the TMessage are relative to its start, which includes the
    # 8-byte header that pyroot_to_buffer removes
    cursor = uproot.source.cursor.Cursor(0, origin=-8)
    maybestreamers = _GetStreamersOnce(obj)
    detatched = _NoFile()
    return uproot.deserialization.read_object_any(
        chunk, cursor, {}, maybestreamers, detatched, None
    )


class _PyROOTWritable:
    def __init__(self, obj):
        self._obj = obj
        self._class_rawstreamers = None

    @property
    def class_rawstreamers(self):
        # the StreamerInfo of this object and all that it contains, for the file
        if self._class_rawstreamers is None:
            self._class_rawstreamers = _GetStreamersOnce(self._obj)._dump()
        return self._class_rawstreamers

    @property
    def classname(self):
        tclass = self._obj.IsA()
        return tclass.GetName()

    @property
    def fTitle(self):
        return self._obj.GetTitle()

    # see uproot.writing.identify.add_to_directory
    references_relative_to_key = True

    def serialize(self, name=None, key_num_bytes=0):
        if name is None or name == self._obj.GetName():
            obj = self._obj
        else:
            obj = self._obj.Clone(name)

        return _serialize_in_key(obj, key_num_bytes).tobytes()
