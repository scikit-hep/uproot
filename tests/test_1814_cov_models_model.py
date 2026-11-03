# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import gc
import io
import pickle
import struct
import sys
import threading
import types

import numpy
import pytest
import skhep_testdata

import uproot

kByteCount = 0x40000000


@pytest.fixture(scope="module")
def file():
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        yield f.file


def wrap(file, data):
    return uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(data, numpy.uint8)
    )


def forth_context():
    forth = threading.local()
    forth.gen = uproot._awkwardforth.ForthGenerator(None)
    return {"forth": forth, "path": ("branch",)}


def with_header(version, payload=b""):
    return struct.pack(">IH", kByteCount | (2 + len(payload)), version) + payload


def test_bootstrap_classes():
    assert uproot.model.bootstrap_classes() == {
        name: uproot.classes[name] for name in uproot.model.bootstrap_classnames
    }


def test_reset_classes_restores_every_class(reset_classes):
    assert set(uproot.model.bootstrap_classes()) == set(
        uproot.model.bootstrap_classnames
    )
    for name in "TClonesArray", "TTime", "TLeafG", "TH1F", "ROOT::RNTuple":
        assert name in uproot.classes


@pytest.mark.parametrize(
    "encoded, decoded, version",
    [
        ("Model_TH1F", "TH1F", None),
        ("Model_TH1F_v3", "TH1F", 3),
        ("Unknown_TH1F_v3", "TH1F", 3),
        ("Model_Some_3a3a_Thing_v2", "Some::Thing", 2),
        # the "v3" is part of the name, not a version
        ("Model_Some_3a3a_v3", "Some::v3", None),
    ],
)
def test_classname_decode(encoded, decoded, version):
    assert uproot.model.classname_decode(encoded) == (decoded, version)
    assert uproot.model.classname_version(encoded) == version


def test_classname_encode_round_trip():
    encoded = uproot.model.classname_encode("Some::v3")
    assert encoded == "Model_Some_3a3a_v3"
    assert uproot.model.classname_version(encoded) is None


def test_classname_not_encoded():
    with pytest.raises(ValueError, match="not an encoded classname: TH1F"):
        uproot.model.classname_decode("TH1F")


@pytest.mark.parametrize(
    "name, unknown", [("Model_TH1F", False), ("Unknown_TH1F", True)]
)
def test_classname_already_encoded(name, unknown):
    with pytest.raises(ValueError, match=f"classname is already encoded: {name}"):
        uproot.model.classname_encode(name, unknown=unknown)


def test_class_named():
    tree = uproot.classes["TTree"]
    assert uproot.model.class_named("TTree") is tree
    assert uproot.model.class_named("TTree", 20) is tree.known_versions[20]
    assert uproot.model.class_named("TNamed", 12345) is uproot.classes["TNamed"]

    with pytest.raises(ValueError, match="no class named Nope in uproot.classes"):
        uproot.model.class_named("Nope")
    with pytest.raises(
        ValueError, match="no class named TTree with version 1 in uproot.classes"
    ):
        uproot.model.class_named("TTree", 1)
    with pytest.raises(
        ValueError, match="no class named TTree in the 'custom_classes' dict"
    ):
        uproot.model.class_named("TTree", custom_classes={})


def test_model_repr():
    versionless = uproot.models.TTree.Model_ROOT_3a3a_TIOFeatures.empty()
    assert repr(versionless).startswith("<ROOT::TIOFeatures at 0x")

    versioned = uproot.models.TAtt.Model_TAttLine_v2.empty()
    assert repr(versioned).startswith("<TAttLine (version 2) at 0x")


def test_context_manager_and_close_attached():
    path = skhep_testdata.data_path("uproot-small-flat-tree.root")
    tree = uproot.open(path)["tree"]
    assert tree.closed is False
    with tree as same:
        assert same is tree
        assert tree.closed is False
    assert tree.closed is True

    tree = uproot.open(path)["tree"]
    tree.close()
    assert tree.closed is True


def test_context_manager_and_close_detached():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        hist = f["hpx"]

    assert isinstance(hist.file, uproot.reading.DetachedFile)
    assert hist.closed is None
    with hist as same:
        assert same is hist
    hist.close()
    assert hist.closed is None
    assert hist.member("fEntries") == 75000


def test_members(file):
    tree = file.root_directory["tree"]

    assert tree.instance_version == tree.class_version == 19
    assert tree.has_member("fName")
    assert not tree.has_member("fName", all=False)
    assert tree.has_member("fEntries", all=False)
    assert not tree.has_member("nope")

    assert tree.member("fName") == "tree"
    assert tree.member("fName", all=False, none_if_missing=True) is None
    assert tree.member("nope", none_if_missing=True) is None

    with pytest.raises(uproot.KeyInFileError, match="not found: 'fName'") as err:
        tree.member("fName", all=False)
    assert "'fEntries'" in str(err.value)
    assert err.value.file_path == file.file_path


def test_base_with_a_non_model():
    # A base is whatever the model's postprocess returned, not necessarily a Model.
    obj = uproot.models.TNamed.Model_TNamed.empty()
    obj._bases.append("not a model")
    assert obj.base(str) == ["not a model"]
    assert obj.base(int) == []
    assert obj.base("TObject") == []


@pytest.mark.parametrize("name", ["TList", "TBasket"])
def test_cannot_be_awkward_by_default(name):
    with pytest.raises(uproot.interpretation.objects.CannotBeAwkward, match=name):
        uproot.classes[name].awkward_form(None, {})


def test_wrong_version_is_redirected_in_forth(file):
    # The instance has a version that this class doesn't model: read goes to the
    # right class (unknown here), pushing a Forth node for the time that lasts.
    cls = uproot.models.TMatrixT.Model_TMatrixTSym_3c_double_3e__v5
    context = forth_context()
    data = with_header(6, b"\x00" * 10) + b"tail"
    cursor = uproot.source.cursor.Cursor(0)

    obj = cls.read(wrap(file, data), cursor, context, file, file.detached, None)

    assert isinstance(obj, uproot.model.UnknownClassVersion)
    assert repr(obj).startswith("<TMatrixTSym<double> with unknown version 6 at 0x")
    assert cursor.index == 16

    gen = context["forth"].gen
    assert gen._active_node is gen._top_node
    assert [x.name.split()[0] for x in gen._top_node.children] == [
        "wrong-instance-version"
    ]


def test_to_writable_of_a_non_writable_dispatch(file):
    tree = file.root_directory["tree"]
    with pytest.raises(
        NotImplementedError, match="this ROOT type is not writable: TTree"
    ):
        tree.to_writable()


def test_to_writable_of_an_unmodeled_class():
    cls = uproot.model.unknown_class("NotAnyClass")
    with pytest.raises(
        NotImplementedError, match="this ROOT type is not writable: NotAnyClass"
    ):
        cls.empty().to_writable()


def test_to_writable_from_streamer_generated_model():
    # The generated class isn't writable, but the unversioned model of the same name is.
    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        generated = f.file.streamer_named("TNamed", 1).new_class(f.file)

    obj = generated.empty()
    obj._members.update({"fName": "name", "fTitle": "title"})
    obj._bases.append(uproot.models.TObject.Model_TObject.empty())

    assert not generated.writable
    writable = obj.to_writable()
    assert type(writable) is uproot.classes["TNamed"]
    assert writable.member("fName") == "name"
    assert writable.member("fTitle") == "title"


def test_serialize_not_implemented(file):
    tree = file.root_directory["tree"]
    with pytest.raises(
        NotImplementedError,
        match=r"can't write Model_TTree_v19 instances yet \('serialize' method",
    ):
        tree.serialize()


def custom_file():
    return uproot.open(
        skhep_testdata.data_path("uproot-small-flat-tree.root"),
        custom_classes=uproot.model.bootstrap_classes(),
    )


def test_new_class_falls_back_on_uproot_classes():
    dispatch = uproot.classes["TMatrixTSym<double>"]
    with custom_file() as f:
        assert f.file.streamer_named("TMatrixTSym<double>", 5) is None
        assert dispatch.new_class(f.file, 5) is dispatch.known_versions[5]


def test_new_class_for_unknown_version_with_custom_classes():
    dispatch = uproot.classes["TMatrixTSym<double>"]
    with custom_file() as f:
        cls = dispatch.new_class(f.file, 99)
    assert cls is uproot.model.unknown_class("TMatrixTSym<double>", 99)
    assert issubclass(cls, uproot.model.UnknownClassVersion)


unknown_cases = [pytest.param(None, id="class"), pytest.param(7, id="version")]


def read_unknown(file, version, chunk):
    return uproot.model.unknown_class("NotAnyClass", version).read(
        chunk, uproot.source.cursor.Cursor(0), {"a": 1}, file, file.detached, None
    )


@pytest.mark.parametrize("version", unknown_cases)
def test_unknown_is_skipped(file, version):
    payload = struct.pack(">ii", 1, 2)
    chunk = wrap(file, with_header(version or 0, payload) + b"tail")
    cursor = uproot.source.cursor.Cursor(0)

    obj = uproot.model.unknown_class("NotAnyClass", version).read(
        chunk, cursor, {"a": 1}, file, file.detached, None
    )

    assert cursor.index == 14
    assert obj.chunk is chunk
    assert obj.context["a"] == 1
    if version is None:
        assert repr(obj).startswith("<Unknown NotAnyClass at 0x")
        assert isinstance(obj, uproot.model.UnknownClass)
    else:
        assert repr(obj).startswith("<NotAnyClass with unknown version 7 at 0x")
        assert isinstance(obj, uproot.model.UnknownClassVersion)


@pytest.mark.parametrize("version", unknown_cases)
def test_unknown_debug(file, version):
    payload = struct.pack(">ii", 1, 2)
    chunk = wrap(file, with_header(version or 0, payload))
    obj = read_unknown(file, version, chunk)

    assert obj.debug_array(skip_bytes=6, dtype=">i4").tolist() == [1, 2]
    assert obj.debug_array().tolist() == list(chunk.raw_data)
    # the array is cut to a multiple of the dtype's size
    assert len(obj.debug_array(skip_bytes=3, dtype=">i4")) == 2

    stream = io.StringIO()
    obj.debug(skip_bytes=6, limit_bytes=8, dtype=">i4", stream=stream)
    lines = stream.getvalue().splitlines()
    assert lines[1].split() == ["0", "0", "0", "1", "0", "0", "0", "2"]
    assert lines[3].split() == ["1", "2"]


@pytest.mark.parametrize("version", unknown_cases)
def test_unknown_chunk_was_deleted(file, version):
    chunk = wrap(file, with_header(version or 0))
    obj = read_unknown(file, version, chunk)
    del chunk
    gc.collect()
    with pytest.raises(RuntimeError, match="been deleted"):
        obj.chunk  # noqa: B018


@pytest.mark.parametrize(
    "version, message",
    [
        (None, "unknown class NotAnyClass that cannot be skipped because its number"),
        (7, "class NotAnyClass with unknown version 7 cannot be skipped because"),
    ],
)
def test_unknown_without_byte_count_cannot_be_skipped(file, version, message):
    # a version number with no byte count before it
    data = struct.pack(">HI", 7, 0)
    with pytest.raises(ValueError, match=message) as err:
        read_unknown(file, version, wrap(file, data))
    assert file.file_path in str(err.value)


def test_real_unknown_class_without_byte_count():
    filename = skhep_testdata.data_path("uproot-issue-418.root")
    with uproot.open(filename) as f:
        with pytest.raises(ValueError, match="unknown class StIOEvent that cannot be"):
            f["bfcTree.0000002000"]


def test_unpickled_in_a_process_without_the_dynamic_class(monkeypatch):
    # A streamer-generated class doesn't exist in a new process until it's unpickled.
    with uproot.open(skhep_testdata.data_path("uproot-issue-350.root")) as f:
        obj = f["data"]
        expected = obj.all_members

    name = type(obj).__name__
    assert type(obj).__module__ == "uproot.dynamic"
    data = pickle.dumps(obj)
    monkeypatch.delitem(vars(uproot.dynamic), name)

    restored = pickle.loads(data)

    assert type(restored) is not type(obj)
    assert type(restored).__name__ == name
    assert isinstance(restored, uproot.model.DynamicModel)
    assert type(restored).member_names == obj.member_names
    assert "_nominalBinWidth" in type(restored).member_names
    assert type(restored).base_names_versions == obj.base_names_versions
    assert restored.classname == obj.classname == "RooHist"
    assert restored.all_members.keys() == expected.keys()
    assert restored.member("fName") == expected["fName"]

    # the second time, the class already has its data
    again = pickle.loads(data)
    assert type(again) is type(restored)
    assert again.member("fName") == expected["fName"]


def test_dynamic_getattr_returns_the_same_class():
    name = "Model_NotInAnyFile_v1"
    assert name not in vars(uproot.dynamic)
    try:
        first = uproot.dynamic.__getattr__(name)
        assert issubclass(first, uproot.model.DynamicModel)
        assert uproot.dynamic.__getattr__(name) is first
        assert uproot.dynamic.Model_NotInAnyFile_v1 is first
    finally:
        vars(uproot.dynamic).pop(name, None)
    assert name not in vars(uproot.dynamic)


def test_behavior_of_tbranchelement():
    import uproot.behaviors.TBranchElement

    cls = uproot.behavior_of("TBranchElement")
    assert cls is uproot.behaviors.TBranchElement.TBranchElement
    assert issubclass(cls, uproot.behaviors.TBranch.TBranch)


def test_behavior_module_without_its_class(monkeypatch):
    name = "NotReallyABehavior"
    module = types.ModuleType(f"uproot.behaviors.{name}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        uproot.behavior.behavior_of,
        "_module_names",
        [*uproot.behavior.behavior_of._module_names, name],
    )
    assert uproot.behavior_of(name) is None
