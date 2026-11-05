# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Small models of basic ROOT classes: what they show, how they serialize, and which
Awkward forms and strided interpretations they ask for.
"""

from __future__ import annotations

import struct

import numpy
import pytest
import skhep_testdata

import uproot

kByteCount = 0x40000000


@pytest.fixture(scope="module")
def file():
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        yield f.file


def form_context(**kwargs):
    return {
        "index_format": "i64",
        "header": False,
        "tobject_header": True,
        "breadcrumbs": (),
        **kwargs,
    }


def read(cls, data, file, context=None):
    chunk = uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(data, numpy.uint8)
    )
    cursor = uproot.source.cursor.Cursor(0)
    out = cls.read(
        chunk, cursor, {} if context is None else context, file, file.detached, None
    )
    return out, cursor


def tobject(fbits=0x03000000):
    return struct.pack(">HII", 1, 0, fbits)


def versioned(cls, version):
    # Custom models of these classes may be versioned, unlike the built-in ones.
    return type(f"{cls.__name__}_v{version}", (cls,), {})


def test_tobject_with_byte_count_in_version(file):
    # the version's byte-count bit announces 4 bytes to ignore
    data = struct.pack(">HI", 0x4001, 0xFFFFFFFF) + struct.pack(">II", 7, 0x03000000)
    obj, cursor = read(uproot.models.TObject.Model_TObject, data, file)
    assert cursor.index == 14
    assert obj.member("@fUniqueID") == 7
    assert repr(obj).startswith(f"<TObject 7 {0x03000000} at 0x")


def test_tobject_is_referenced(file):
    data = tobject(uproot.const.kIsReferenced | 0x03000000) + b"\x00\x05"
    obj, cursor = read(uproot.models.TObject.Model_TObject, data, file)
    assert cursor.index == 12
    assert obj.member("@fBits") & int(uproot.const.kIsReferenced)


@pytest.mark.parametrize(
    "tobject_header, names",
    [
        (False, []),
        (True, ["@instance_version", "@num_bytes", "@fUniqueID", "@fBits", "@pidf"]),
    ],
)
def test_tobject_forms(tobject_header, names):
    cls = uproot.models.TObject.Model_TObject
    interp = cls.strided_interpretation(None, tobject_header=tobject_header)
    assert [x[0] for x in interp.members] == names

    form = cls.awkward_form(None, form_context(tobject_header=tobject_header))
    assert form.fields == names
    assert form.parameters == {"__record__": "TObject"}


@pytest.mark.parametrize(
    "header, names",
    [(False, ["fDatime"]), (True, ["@num_bytes", "@instance_version", "fDatime"])],
)
def test_tdatime_forms(header, names):
    cls = uproot.models.TDatime.Model_TDatime
    interp = cls.strided_interpretation(None, header=header)
    assert [x[0] for x in interp.members] == names

    form = cls.awkward_form(None, form_context(header=header))
    assert form.fields == names
    assert str(form.content("fDatime").primitive) == "uint32"


def test_ttime_forms_and_memberwise(file):
    cls = uproot.models.TTime.Model_TTime_v2
    names = ["@num_bytes", "@instance_version", "fMilliSec"]

    interp = cls.strided_interpretation(None)
    assert [x[0] for x in interp.members] == names
    assert interp.members[-1][1] == numpy.dtype(">i8")

    assert cls.awkward_form(None, form_context()).fields == names[2:]
    form = cls.awkward_form(None, form_context(header=True))
    assert form.fields == names
    assert form.parameters == {"__record__": "TTime"}

    with pytest.raises(
        uproot.interpretation.objects.CannotBeStrided,
        match="contain members of the same type cannot be strided",
    ):
        cls.strided_interpretation(None, breadcrumbs=(cls,))
    with pytest.raises(
        uproot.interpretation.objects.CannotBeAwkward,
        match="contain members of the same type cannot be Awkward Arrays",
    ):
        cls.awkward_form(None, form_context(breadcrumbs=(cls,)))

    obj = cls.empty()
    chunk = uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(struct.pack(">q", 1234), numpy.uint8)
    )
    cursor = uproot.source.cursor.Cursor(0)
    obj.read_member_n(chunk, cursor, {}, file, 1)
    assert obj.members == {}
    assert cursor.index == 0
    obj.read_member_n(chunk, cursor, {}, file, 0)
    assert obj.members == {"fMilliSec": 1234}
    assert cursor.index == 8


def test_ttime_from_file():
    with uproot.open(skhep_testdata.data_path("uproot-issue-861.root")) as f:
        real, live = f["RealTime_0"], f["LiveTime_0"]
        assert real.member_names == ["fMilliSec"]
        assert real.member("fMilliSec") >= live.member("fMilliSec") > 0


def test_ttime_in_awkward_forth(tmp_path):
    # Any 8 bytes per entry can be seen as TTimes (with no headers) through AsObjects.
    awkward = pytest.importorskip("awkward")

    filename = tmp_path / "ttime.root"
    with uproot.recreate(filename) as f:
        tree = f.mktree("tree", {"x": "var * int64"})
        tree.extend({"x": awkward.Array([[1], [2], [2**40]])})

    with uproot.open(filename) as f:
        branch = f["tree"]["x"]
        interpretation = uproot.interpretation.objects.AsObjects(
            uproot.classes["TTime"], branch
        )
        array = branch.array(interpretation)

    assert array.fMilliSec.tolist() == [1, 2, 2**40]
    assert "stream !q-> node" in interpretation._complete_forth_code


def read_tarray(file):
    data = struct.pack(">i", 10) + numpy.arange(10, dtype=">i4").tobytes()
    return read(uproot.models.TArray.Model_TArrayI, data, file)[0]


def test_tarray(file):
    array = read_tarray(file)
    assert array.nbytes == 40
    assert repr(array).startswith("<TArrayI [0, 1, 2, ..., 7, 8, 9] at 0x")
    assert numpy.asarray(array, dtype=numpy.float32).dtype == numpy.float32
    assert array.tojson() == list(range(10))


def test_tarray_serialize(file):
    array = read_tarray(file)
    payload = struct.pack(">i", 10) + numpy.arange(10, dtype=">i4").tobytes()

    assert array.serialize() == (
        struct.pack(">IH", kByteCount | (2 + len(payload)), 1) + payload
    )
    out = []
    array._serialize(out, False, None, numpy.uint32(0))
    assert b"".join(out) == payload


@pytest.mark.parametrize("header", [True, False])
def test_serialize_with_bases(header):
    """TArray and TString don't have bases in ROOT, but they would be serialized."""
    array = uproot.models.TArray.Model_TArrayI.empty()
    array._members["fN"] = 1
    array._data = numpy.array([5], ">i4")
    array._bases.append(uproot.models.TObject.Model_TObject.empty())

    string = uproot.models.TString.Model_TString("hi")
    string._bases = [uproot.models.TObject.Model_TObject.empty()]

    expected_tobject = b"\x00\x01" + struct.pack(">II", 0, 0)
    for obj, payload, version in [
        (array, expected_tobject + struct.pack(">ii", 1, 5), 1),
        (string, expected_tobject + b"\x02hi", 2),
    ]:
        out = []
        obj._serialize(out, header, None, numpy.uint32(0))
        prefix = b""
        if header:
            prefix = struct.pack(">IH", kByteCount | (2 + len(payload)), version)
        assert b"".join(out) == prefix + payload


TATT_MEMBERS = {
    "TAttLine_v2": (
        {"fLineColor": 2, "fLineStyle": 3, "fLineWidth": 4},
        struct.pack(">hhh", 2, 3, 4),
        2,
    ),
    "TAttFill_v2": (
        {"fFillColor": 5, "fFillStyle": 1001},
        struct.pack(">hh", 5, 1001),
        2,
    ),
    "TAttMarker_v2": (
        {"fMarkerColor": 6, "fMarkerStyle": 20, "fMarkerSize": 1.5},
        struct.pack(">hhf", 6, 20, 1.5),
        2,
    ),
    "TAttAxis_v4": (
        {
            "fNdivisions": 510,
            "fAxisColor": 1,
            "fLabelColor": 2,
            "fLabelFont": 42,
            "fLabelOffset": 0.5,
            "fLabelSize": 0.25,
            "fTickLength": 0.75,
            "fTitleOffset": 1.0,
            "fTitleSize": 0.125,
            "fTitleColor": 3,
            "fTitleFont": 62,
        },
        struct.pack(">ihhhfffffhh", 510, 1, 2, 42, 0.5, 0.25, 0.75, 1.0, 0.125, 3, 62),
        4,
    ),
    "TAtt3D_v1": ({}, b"", 1),
}


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("name", sorted(TATT_MEMBERS))
def test_tatt_serialize_with_bases(name, header):
    """The TAtt classes don't have bases in ROOT, but they would be serialized first."""
    members, payload, version = TATT_MEMBERS[name]
    obj = getattr(uproot.models.TAtt, "Model_" + name).empty()
    obj._members.update(members)
    base = uproot.models.TObject.Model_TObject.empty()
    base._members["@fUniqueID"] = 7
    obj._bases.append(base)

    out = []
    obj._serialize(out, header, None, numpy.uint32(0))
    payload = b"\x00\x01" + struct.pack(">II", 7, 0) + payload
    prefix = b""
    if header:
        prefix = struct.pack(">IH", kByteCount | (2 + len(payload)), version)
    assert b"".join(out) == prefix + payload


def test_tstring(file):
    data = b"\x05hello"
    obj, cursor = read(uproot.models.TString.Model_TString, data, file)
    assert cursor.index == 6
    assert obj == "hello"
    assert repr(obj).startswith("<TString 'hello' at 0x")
    assert obj.tojson() == "hello"


def test_tobjstring(file):
    payload = tobject() + b"\x05hello"
    data = struct.pack(">IH", kByteCount | (2 + len(payload)), 1) + payload
    obj, _ = read(uproot.models.TObjString.Model_TObjString, data, file)
    assert obj == "hello"
    assert repr(obj).startswith("<TObjString 'hello' at 0x")
    assert obj.fTitle == "Collectable string class"

    out = []
    obj._serialize(out, False, None, numpy.uint32(0))
    assert b"".join(out)[-6:] == b"\x05hello"

    form = obj.awkward_form(None, form_context())
    assert form.parameters == {"__array__": "string"}
    assert form.offsets == "i64"
    assert form.content.parameters == {"__array__": "char"}


def test_tnamed(file):
    payload = tobject() + b"\x04name\x05title"
    data = struct.pack(">IH", kByteCount | (2 + len(payload)), 1) + payload
    obj, _ = read(uproot.models.TNamed.Model_TNamed, data, file)
    assert repr(obj).startswith("<TNamed 'name' title='title' at 0x")

    obj.members["fTitle"] = ""
    assert repr(obj).startswith("<TNamed 'name' at 0x")


@pytest.mark.parametrize(
    "header, names",
    [(False, ["fName", "fTitle"]), (True, ["@num_bytes", "@instance_version"])],
)
def test_tnamed_form(header, names):
    form = uproot.models.TNamed.Model_TNamed.awkward_form(
        None, form_context(header=header)
    )
    assert form.fields[: len(names)] == names
    assert form.fields[-2:] == ["fName", "fTitle"]
    assert form.parameters == {"__record__": "TNamed"}
    assert form.content("fName").parameters["__array__"] == "string"


def test_collections_serialize_without_header(file):
    payload = tobject() + b"\x00" + struct.pack(">i", 0)
    data = struct.pack(">IH", kByteCount | (2 + len(payload)), 5) + payload
    tlist, _ = read(uproot.models.TList.Model_TList, data, file)

    named = uproot.models.TNamed.Model_TNamed.empty()
    named._bases.append(uproot.models.TObject.Model_TObject.empty())
    named._members.update({"fName": "name", "fTitle": "title"})

    array = file.root_directory["tree"].branches[0].member("fBranches")

    for obj in tlist, named, array:
        out = []
        obj._serialize(out, False, None, numpy.uint32(0))
        assert b"".join(out) == obj.serialize()[6:]


@pytest.mark.parametrize(
    "cls, data, text",
    [
        ("TList", [1, 2], "<TList (version 5) of 2 items at 0x"),
        ("TObjArray", [1], "<TObjArray (version 3) of 1 items at 0x"),
        ("THashList", None, "<THashList (version 5) of 2 items at 0x"),
    ],
)
def test_collection_repr_with_version(cls, data, text):
    version = int(text.split("version ")[1].split(")")[0])
    obj = versioned(uproot.classes[cls], version).empty()
    if data is None:
        base = uproot.classes["TList"].empty()
        base._data = [1, 2]
        obj._bases.append(base)
    else:
        obj._data = data
    assert repr(obj).startswith(text)


@pytest.mark.parametrize(
    "cls, text",
    [
        ("TList", "<TList of 0 items at 0x"),
        ("TObjArray", "<TObjArray of 0 items at 0x"),
    ],
)
def test_collection_repr_without_version(cls, text):
    obj = uproot.classes[cls].empty()
    obj._data = []
    assert repr(obj).startswith(text)


def test_thashlist_repr_without_version():
    obj = uproot.models.THashList.Model_THashList.empty()
    obj._bases.append(uproot.classes["TList"].empty())
    obj._bases[0]._data = ["a"]
    assert repr(obj).startswith("<THashList of 1 items at 0x")
    assert obj[0] == "a"
    assert len(obj) == 1


@pytest.mark.parametrize(
    "module, name, text",
    [
        ("TArray", "Model_TArrayI", "<TArrayI (version 2) [1 2] at 0x"),
        ("TRef", "Model_TRefArray", "<TRefArray (version 2) [1 2] at 0x"),
    ],
)
def test_numeric_repr_with_version(module, name, text):
    cls = getattr(getattr(uproot.models, module), name)
    obj = versioned(cls, 2).empty()
    obj._data = numpy.array([1, 2], ">i4")
    assert repr(obj).replace(", ", " ").startswith(text)


@pytest.mark.parametrize(
    "module, name, text",
    [
        ("TString", "Model_TString", "<TString (version 3) 'hi' at 0x"),
        ("TObjString", "Model_TObjString", "<TObjString (version 3) 'hi' at 0x"),
    ],
)
def test_string_repr_with_version(module, name, text):
    cls = getattr(getattr(uproot.models, module), name)
    assert repr(versioned(cls, 3)("hi")).startswith(text)


def test_tobjarray_to_writable(file):
    original = file.root_directory["tree"].branches[0].member("fLeaves")
    writable = original.to_writable()
    assert writable is not original
    assert type(writable) is type(original)
    assert list(writable) == list(original)


def test_tref(file):
    # TObject version, fUniqueID (the reference), fBits (kNotDeleted), pidf
    data = struct.pack(">HIIH", 1, 1234, 0x02000000, 0)
    obj, cursor = read(uproot.models.TRef.Model_TRef, data, file)
    assert cursor.index == 12
    assert obj.ref == 1234
    assert repr(obj) == "<TRef 1234>"


@pytest.mark.parametrize(
    "tobject_header, names",
    [(False, []), (True, ["@pidf", "ref", "@other1", "@other2"])],
)
def test_tref_form(tobject_header, names):
    form = uproot.models.TRef.Model_TRef.awkward_form(
        None, form_context(tobject_header=tobject_header)
    )
    assert form.fields == names
    assert form.parameters == {"__record__": "TRef"}


def test_tbasket_of_a_flat_branch():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")) as f:
        branch = f["events"]["Run"]
        basket = branch.basket(0)
        assert basket.counts is None
        assert basket.block_compression_info is None
        assert basket.raw_data is not None
        assert basket.raw_data.tobytes() == basket.data.tobytes()
        assert basket.byte_offsets is None


def test_tbasket_header_only():
    # TBranch does this to find the number of entries in a basket.
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")) as f:
        branch = f["events"]["Run"]
        start = int(branch.member("fBasketSeek")[0])
        stop = start + int(branch.member("fBasketBytes")[0])
        basket = uproot.models.TBasket.Model_TBasket.read(
            f.file.source.chunk(start, stop),
            uproot.source.cursor.Cursor(start),
            {"read_basket": False, "basket_num": 0},
            f.file,
            f.file,
            branch,
        )
        assert basket.num_entries == branch.member("fEntries")
        assert basket.data is None
        assert basket.raw_data is None
        assert basket.byte_offsets is None


def test_tbasket_repr_without_number():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")) as f:
        branch = f["events"]["Run"]
        start = int(branch.member("fBasketSeek")[0])
        stop = start + int(branch.member("fBasketBytes")[0])
        basket = uproot.models.TBasket.Model_TBasket.read(
            f.file.source.chunk(start, stop),
            uproot.source.cursor.Cursor(start),
            {},
            f.file,
            f.file,
            branch,
        )
        assert basket.basket_num is None
        assert repr(basket).startswith("<TBasket (unknown) of 'Run' at 0x")


def clones_data(item_classname, items):
    payload = tobject() + b"\x00"
    payload += bytes([len(item_classname)]) + item_classname.encode()
    payload += struct.pack(">ii", len(items), 0)
    for item in items:
        if item is None:
            payload += b"\x00"
        else:
            body = tobject() + bytes([len(item)]) + item.encode()
            payload += b"\x01" + (
                struct.pack(">IH", kByteCount | (2 + len(body)), 1) + body
            )
    # ClassDef(TClonesArray, 4)
    return struct.pack(">IH", kByteCount | (2 + len(payload)), 4) + payload


@pytest.mark.parametrize(
    "item_classname, version",
    [("TObjString", "max"), ("TObjString;1", 1)],
)
def test_tclonesarray(file, item_classname, version):
    data = clones_data(item_classname, ["one", None, "three"])
    obj, cursor = read(uproot.models.TClonesArray.Model_TClonesArray, data, file)

    assert cursor.index == len(data)
    assert obj.item_classname == "TObjString"
    assert obj.item_classversion == version
    assert obj.item_class is uproot.classes["TObjString"]
    assert obj.member("fName") == ""
    assert obj.member("fSize") == 3
    # empty slots are skipped
    assert len(obj) == 2
    assert list(obj) == ["one", "three"]
    assert obj[1] == "three"


def test_tclonesarray_from_root(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filename = str(tmp_path / "clones.root")
    file = ROOT.TFile(filename, "recreate")
    clones = ROOT.TClonesArray("TObjString", 2)
    clones.SetName("clones")
    # the bypassed format isn't implemented (and is the default)
    clones.BypassStreamer(False)
    for i in range(2):
        clones.ConstructedAt(i).SetString(f"item{i}")
    clones.Write("clones", ROOT.TObject.kSingleKey)
    file.Close()

    with uproot.open(filename) as f:
        obj = f["clones"]
        assert obj.item_classname == "TObjString"
        assert obj.item_classversion == 1
        assert list(obj) == ["item0", "item1"]
