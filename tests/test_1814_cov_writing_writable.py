# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import io
import shutil
import uuid

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.writing.writable import WritableNTuple, _regularize_input_type_to_dict


def histogram():
    return np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0])


def test_file_attributes(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path, compression=uproot.ZLIB(3)) as directory:
        file = directory.file
        directory["object"] = "text"

        assert repr(file) == (
            f"<WritableFile {path.as_posix()!r} at 0x{id(file):012x}>"
        )

        new_uuid = uuid.UUID(int=1)
        file.initial_directory_bytes = 512
        file.uuid_function = lambda: new_uuid
        assert file.options == {
            "initial_directory_bytes": 512,
            "uuid_function": file.uuid_function,
        }
        assert directory.mkdir("sub")._cascading.header.uuid == new_uuid

        assert file.is_64bit is False
        assert file.fUnits == 4
        assert file.fCompress == uproot.ZLIB(3).code

        file.compression = None
        assert file.fCompress == uproot.ZLIB(0).code
        file.compression = uproot.LZMA(2)
        assert file.compression == uproot.LZMA(2)

        with pytest.raises(TypeError, match="compression must be None or a uproot"):
            file.compression = "ZLIB"

        written = {
            name: getattr(file, name)
            for name in [
                "fSeekFree",
                "fNbytesFree",
                "nfree",
                "fSeekInfo",
                "fNbytesInfo",
                "fUnits",
                "fCompress",
            ]
        }
        assert not file.closed

    assert file.closed

    with uproot.open(path) as readable:
        for name, value in written.items():
            assert getattr(readable.file, name) == value


def test_file_as_a_context_manager(tmp_path):
    file = uproot.recreate(tmp_path / "out.root").file
    with file as entered:
        assert entered is file
        assert not file.closed
    assert file.closed


def test_directory_attributes(tmp_path):
    with uproot.recreate(tmp_path / "out.root", compression=None) as file:
        sub = file.mkdir("one/two")
        assert file.path == ()
        assert sub.path == ("one", "two")
        assert sub.object_path == "/one/two/"
        assert repr(sub) == f"<WritableDirectory '/one/two' at 0x{id(sub):012x}>"

        assert sub.compression is None
        sub.compression = uproot.ZSTD(1)
        assert file.compression == uproot.ZSTD(1)

        assert not sub.closed
    assert sub.closed


def make_nested(file):
    file["a"] = "one"
    file["a"] = "two"
    file["sub/b"] = histogram()
    file["sub/deeper/c"] = "three"
    file["d"] = histogram()


def test_listing_names(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)

        everything = [
            "a;1",
            "a;2",
            "sub;1",
            "sub/b;1",
            "sub/deeper;1",
            "sub/deeper/c;1",
            "d;1",
        ]
        assert list(file) == everything
        assert list(file._ipython_key_completions_()) == everything
        assert file.keys(recursive=False) == ["a;1", "a;2", "sub;1", "d;1"]
        # without cycle numbers, each name once
        assert file.keys(cycle=False, recursive=False) == ["a", "sub", "d"]

        # as when reading, both the name in the subdirectory and the path must match
        assert file.keys(filter_name="*b") == ["sub;1", "sub/b;1"]
        assert file.keys(filter_name="*c", cycle=False) == ["sub/deeper/c"]
        assert file.keys(filter_name="b") == []
        assert file.keys(filter_classname="TH1D") == ["sub/b;1", "d;1"]
        written = [file.keys(cycle=cycle) for cycle in (True, False)]

    with uproot.open(tmp_path / "out.root") as file:
        assert [file.keys(cycle=cycle) for cycle in (True, False)] == written


def test_listing_classnames(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)

        assert file.classnames(recursive=False, cycle=False) == {
            "a": "TObjString",
            "sub": "TDirectory",
            "d": "TH1D",
        }
        assert file.classnames(filter_name="*b", cycle=True) == {
            "sub;1": "TDirectory",
            "sub/b;1": "TH1D",
        }
        assert file.classnames(filter_name="b") == {}
        assert file.classnames(filter_classname="TH1D", cycle=False) == {
            "sub/b": "TH1D",
            "d": "TH1D",
        }


def test_values_and_items(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)

        strings = file.values(filter_classname="TObjString")
        assert strings == ["one", "two", "three"]

        assert file.values(filter_name="nothing") == []

        # each name once, as its highest cycle
        assert file.keys(filter_classname="TObjString", cycle=False) == [
            "a",
            "sub/deeper/c",
        ]
        assert file.items(filter_classname="TObjString", cycle=False) == [
            ("a", "two"),
            ("sub/deeper/c", "three"),
        ]
        assert file.items(filter_classname="TObjString", recursive=False) == [
            ("a;1", "one"),
            ("a;2", "two"),
        ]


def test_indexing_paths(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)
        tree = file.mktree("sub/tree", {"x": "float64"})

        assert file["sub/"].path == ("sub",)
        assert file["/sub//deeper/c"] == "three"
        assert file["sub/tree"] is tree
        assert file["sub/deeper"]["c;1"] == "three"

        with pytest.raises(uproot.KeyInFileError, match="indexed by TBranch name"):
            file["sub/tree/x"]
        with pytest.raises(uproot.KeyInFileError, match="indexed by TBranch name"):
            file["sub/tree:x"]
        with pytest.raises(uproot.KeyInFileError, match="a is not a TDirectory"):
            file["a/x"]


def test_semicolon_in_a_name(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["with;semicolon"] = "text"
        file["other"] = "text"
        assert file["with;semicolon"] == "text"
        del file["with;semicolon"]
        assert file.keys() == ["other;1"]


def test_closed_files(tmp_path):
    file = uproot.recreate(tmp_path / "out.root")
    file["object"] = "text"
    tree = file.mktree("tree", {"x": "float64"})
    file.close()

    with pytest.raises(ValueError, match="cannot get data from a closed file"):
        file["object"]
    with pytest.raises(ValueError, match="cannot write data to a closed file"):
        file["object"] = "text"
    with pytest.raises(ValueError, match="cannot delete data from a closed file"):
        del file["object"]
    with pytest.raises(ValueError, match="cannot create a TDirectory in a closed"):
        file.mkdir("sub")
    with pytest.raises(ValueError, match="cannot create a TTree in a closed file"):
        file.mktree("t", {"x": "float64"})
    with pytest.raises(ValueError, match="cannot create a RNTuple in a closed file"):
        file.mkrntuple("t", {"x": "float64"})
    with pytest.raises(ValueError, match="cannot modify a TTree in a closed file"):
        tree.add_branches({"y": np.array([])})


def test_mkdir_over_an_object(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        file["object"] = "text"
        with pytest.raises(TypeError, match="a TObjString already has that name"):
            file.mkdir("object/sub")


def test_mkrntuple_invalid(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(TypeError, match="Input must be a type specification.* int"):
            file.mkrntuple("ntuple", 5)


@pytest.mark.parametrize("kind", ["ttree", "rntuple"])
def test_update_a_file_object_cannot_load_trees(tmp_path, kind):
    buffer = io.BytesIO()
    file = uproot.recreate(buffer)
    if kind == "ttree":
        file.mktree("t", {"x": "float64"})
        message = "does not support accessing existing TTrees"
    else:
        file.mkrntuple("t", {"x": "float64"})
        message = "does not support accessing existing RNTuples"

    with pytest.raises(TypeError, match=message):
        uproot.update(buffer)["t"]


def test_copy_from_invalid_sources(tmp_path):
    source = tmp_path / "source.root"
    with uproot.recreate(source) as file:
        file["object"] = "text"
        file.mktree("tree", {"x": "float64"})
        file.mkrntuple("ntuple", {"x": "float64"})

    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(NotImplementedError, match="from a WritableDirectory"):
            file.copy_from(file)
        with pytest.raises(TypeError, match="'source' must be a TDirectory"):
            file.copy_from(str(source))

        with uproot.open(source) as readable:
            with pytest.raises(ValueError, match="no objects found with names"):
                file.copy_from(readable, filter_name="nothing")
            assert (
                file.copy_from(readable, filter_name="nothing", require_matches=False)
                is None
            )
            with pytest.raises(NotImplementedError, match="cannot copy TTree"):
                file.copy_from(readable, filter_name="tree")
            with pytest.raises(NotImplementedError, match="cannot copy ROOT::RNTuple"):
                file.copy_from(readable, filter_name="ntuple")

        assert file.keys() == []


def test_copy_from_directories(tmp_path):
    source = tmp_path / "source.root"
    with uproot.recreate(source) as file:
        file["top"] = "one"
        file["sub/first"] = "two"
        file["sub/second"] = histogram()
        file.mkdir("empty")

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        with uproot.open(source) as readable:
            file.copy_from(readable)

    with uproot.open(path) as file:
        assert sorted(file.keys(cycle=False)) == [
            "empty",
            "sub",
            "sub/first",
            "sub/second",
            "top",
        ]
        assert file["sub/first"] == "two"

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    assert str(root_file.Get("sub/first")) == "two"
    assert root_file.Get("sub/second").GetBinContent(2) == 2
    assert root_file.GetDirectory("empty").GetListOfKeys().GetSize() == 0
    root_file.Close()


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (({"a": "one", "b": "two"},), {}),
        (([("a", "one")],), {"b": "two"}),
        ((), {"a": "one", "b": "two"}),
    ],
    ids=["mapping", "pairs-and-keywords", "keywords"],
)
def test_update_arguments(tmp_path, args, kwargs):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.update(*args, **kwargs)

    with uproot.open(path) as file:
        assert file["a"] == "one"
        assert file["b"] == "two"


def test_loading_subdirectory_again_finds_trees_made_elsewhere(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.mkdir("sub")

        # another handle doesn't share the first one's subdirectories
        other = file.file.root_directory
        tree = other.mktree("sub/tree", {"x": "float64"})
        tree.extend({"x": np.array([1.5])})
        assert file["sub"]._cascading is tree._cascading.directory

        file["sub/after"] = "text"
        tree.extend({"x": np.array([2.5])})

    with uproot.open(path) as file:
        assert file["sub/after"] == "text"
        assert file["sub/tree/x"].array().tolist() == [1.5, 2.5]


def test_tree_attributes(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("sub/tree", {"x": "float64", "y": "var * int32"})
        assert tree.path == ("sub", "tree")
        assert tree.object_path == "/sub/tree/"
        assert tree.file_path == path.as_posix()
        assert tree.file is file.file
        assert repr(tree) == f"<WritableTree '/sub/tree' at 0x{id(tree):012x}>"
        assert tree.num_baskets == 0
        assert not tree.closed

        branch = tree["x"]
        assert repr(branch) == (
            f"<WritableBranch 'x' in '/sub/tree' at 0x{id(branch):012x}>"
        )
        assert branch.type == "float64"
        assert branch.compression == uproot.ZLIB(1)
        with pytest.raises(TypeError, match="compression must be None or a uproot"):
            branch.compression = "ZLIB"

        with pytest.raises(uproot.KeyInFileError, match="no such branch in writable"):
            tree["nothing"]

        tree.close()
        assert tree.closed


def test_tree_as_a_context_manager(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("tree", {"x": "float64"})
        with tree as entered:
            assert entered is tree
        assert tree.closed


def test_tree_compression(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path, compression=None) as file:
        tree = file.mktree("tree", {"rec": {"a": "int32"}, "x": "float64"})
        assert tree.compression is None

        tree["x"].compression = uproot.ZLIB(5)
        assert tree.compression == {"rec_a": None, "x": uproot.ZLIB(5)}

        tree.compression = uproot.LZMA(1)
        assert tree.compression == uproot.LZMA(1)

        tree.compression = {"rec_a": uproot.ZLIB(2), "x": None}
        assert tree["rec_a"].compression == uproot.ZLIB(2)
        assert tree["x"].compression is None

        for value in ["ZLIB", {"x": None}, {"x": None, "rec_a": None, "z": None}]:
            with pytest.raises(TypeError, match="or a mapping of branch names"):
                tree.compression = value

        tree.extend({"rec": {"a": np.ones(1000, np.int32)}, "x": np.ones(1000)})

    with uproot.open(path) as file:
        rec_a, x = file["tree/rec_a"], file["tree/x"]
        assert rec_a.compressed_bytes < rec_a.uncompressed_bytes
        assert x.compressed_bytes == x.uncompressed_bytes
        assert rec_a.array().tolist() == [1] * 1000


def test_add_branches_problems(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("tree", {"x": "float64"})
        tree.extend({"x": np.array([1.0, 2.0])})

        with pytest.raises(ValueError, match="branch 'x' already exists"):
            tree.add_branches({"x": np.array([1.0, 2.0])})

        with pytest.raises(TypeError, match="branch 'y' has object dtype"):
            tree.add_branches({"y": np.array([None, 1.0], dtype=object)})


def test_show(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("sub/tree", {"x": "float64"})
        tree.extend({"x": np.array([1.0])})

        stream = io.StringIO()
        tree.show(stream=stream, name_width=4, typename_width=10)
        assert (
            stream.getvalue().splitlines()[-1].startswith("x    | double     | AsDtype")
        )


def make_ntuple(file, name="sub/ntuple"):
    return file.mkrntuple(name, {"x": "float64", "rec": {"a": "int32"}})


def test_ntuple_attributes(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path, compression=uproot.ZLIB(2)) as file:
        ntuple = make_ntuple(file)
        assert ntuple.path == ("sub", "ntuple")
        assert ntuple.object_path == "/sub/ntuple/"
        assert ntuple.file_path == path.as_posix()
        assert ntuple.file is file.file
        assert repr(ntuple) == f"<WritableNTuple '/sub/ntuple' at 0x{id(ntuple):012x}>"
        assert ntuple.compression == uproot.ZLIB(2)
        assert not ntuple.closed

        with pytest.raises(NotImplementedError, match="set compression on the file"):
            ntuple.compression = uproot.ZLIB(3)

        ntuple.close()
        assert ntuple.closed


def test_ntuple_as_a_context_manager(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        ntuple = make_ntuple(file)
        with ntuple as entered:
            assert entered is ntuple
        assert ntuple.closed


def test_ntuple_add_fields_problems(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        ntuple = file.mkrntuple("ntuple", {"x": "float64", "a": {"b": {"c": "int32"}}})

        with pytest.raises(TypeError, match="only supports simple numeric types"):
            ntuple.add_fields({"y": "var * int32"})

        with pytest.raises(ValueError, match="Field 'a.nothing' not found"):
            ntuple.add_fields({"a.nothing.y": "int32"})

        with pytest.raises(ValueError, match="Field 'nothing' not found"):
            ntuple.add_fields({"nothing.b.y": "int32"})


def test_ntuple_handle_to_a_deleted_key(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        ntuple = make_ntuple(file, "ntuple")

        # this handle isn't known to the file, so it can't be told about the deletion
        unregistered = WritableNTuple(ntuple.path, ntuple.file, ntuple._cascading)
        del file["ntuple"]

        with pytest.raises(uproot.KeyInFileError, match="ntuple"):
            unregistered.add_fields({"y": "int32"})


def test_ntuple_with_columns_that_uproot_does_not_write(tmp_path):
    name = "test_multiple_representations_rntuple_v1-0-0-0.root"
    path = shutil.copy(skhep_testdata.data_path(name), tmp_path / name)

    with uproot.update(path) as file:
        ntuple = file["ntuple"]
        message = "existing RNTuple has 2 columns but reconstructed header has 1"
        with pytest.raises(ValueError, match=message):
            ntuple.add_fields({"y": "int32"})
        with pytest.raises(ValueError, match=message):
            ntuple.extend({"y": np.array([1])})


def test_mktree_with_a_dict_of_multidimensional_arrays(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "out.root"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    with uproot.recreate(path) as file:
        file.mktree("tree", {"rec": {"a": data}})

    with uproot.open(path) as file:
        assert file["tree/rec_a"].array().tolist() == data.tolist()

    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("tree")
    tree.GetEntry(1)
    assert list(tree.rec_a) == [3, 4]
    root_file.Close()


@pytest.mark.parametrize(
    "data",
    [
        {"rec": {"a": [[1, 2], [3]]}},
        {"x": [object()]},
    ],
    ids=["record", "branch"],
)
def test_mktree_with_unrecognizable_data(tmp_path, data):
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(TypeError, match="unrecognizable array type .* associated"):
            file.mktree("tree", data)


def test_mktree_with_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.mktree("series", {"x": pandas.Series([1.5, 2.5])})
        # this index can't be a branch
        file.mktree(
            "dataframe", pandas.DataFrame({"x": [1.5, 2.5]}, index=["one", "two"])
        )

    with uproot.open(path) as file:
        assert file["series/x"].array().tolist() == [1.5, 2.5]
        assert file["dataframe"].keys() == ["x"]
        assert file["dataframe/x"].array().tolist() == [1.5, 2.5]


def test_only_awkward_arrays_are_wrapped_as_one_group_of_branches():
    record = ak.Record({"x": 1.5})
    assert _regularize_input_type_to_dict(record) is record

    array = ak.Array([{"x": 1.5}])
    assert _regularize_input_type_to_dict(array) == {"": array}


def test_extending_a_tree_written_by_root(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "out.root"
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("tree", "from ROOT")
    x = np.zeros(1, dtype=np.float32)
    tree.Branch("x", x, "x/F")
    for value in [1.5, 2.5]:
        x[0] = value
        tree.Fill()
    tree.Write()
    root_file.Close()

    with uproot.update(path) as file:
        tree = file["tree"]
        assert tree._cascading._branch_data[0]["counter"] is None
        tree.extend({"x": np.array([3.5], dtype=np.float32)})

    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("tree")
    assert [entry.x for entry in tree] == [1.5, 2.5, 3.5]
    root_file.Close()


def test_counter_leaf_named_differently_from_its_branch(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "var * int32"})

    # fLeafCount of branch "x" is a leaf named "nx", in a branch named "nx": the
    # name of the branch, the name of its leaf, and the title of its leaf
    data = path.read_bytes()
    assert data.count(b"\x02nx") == 3
    leaf_name = data.index(b"\x02nx", data.index(b"\x02nx") + 1)
    path.write_bytes(data[:leaf_name] + b"\x02mx" + data[leaf_name + 3 :])

    with uproot.update(path) as file:
        branch_data = file["tree"]._cascading._branch_data
        assert [x["fName"] for x in branch_data] == ["nx", "x"]
        # the counter is found through the leaf itself, not through its name
        assert branch_data[1]["counter"] is branch_data[0]
        file["tree"].extend({"x": ak.Array([[1, 2], [3]])})

    with uproot.open(path) as file:
        assert file["tree"]["x"].array().tolist() == [[1, 2], [3]]


def test_reading_a_class_that_needs_the_streamers_of_the_file(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "out.root"
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    vector = ROOT.TVector2(1.5, 2.5)
    root_file.WriteObject(vector, "vector")
    root_file.Close()

    with uproot.update(path) as file:
        vector = file["vector"]
        assert vector.classname == "TVector2"
        assert vector.member("fX") == 1.5
        assert vector.member("fY") == 2.5


def test_jagged_branch_counted_by_another_branch_type(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    ROOT.gInterpreter.Declare("struct Cov1814CountHolder { int n; };")

    path = tmp_path / "out.root"
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("tree", "from ROOT")
    holder = ROOT.Cov1814CountHolder()
    tree.Branch("holder", holder)
    # the counter of this TBranch is a TLeafElement in a TBranchElement
    x = np.zeros(3, dtype=np.float32)
    tree.Branch("x", x, "x[n]/F")
    for n in [1, 2, 3]:
        holder.n = n
        tree.Fill()
    tree.Write()
    root_file.Close()

    with uproot.update(path) as file:
        tree = file["tree"]
        assert [bd["fName"] for bd in tree._cascading._branch_data] == ["x"]
        assert tree._cascading._branch_data[0]["counter"] is None
        assert tree._cascading._rewrite_blockers == (
            "jagged branches with a counter in another branch type",
        )
        # the TBranchElement already stops both, before the blocker is checked
        with pytest.raises(NotImplementedError, match="TBranchElement"):
            tree.add_branches({"y": np.arange(3)})
        with pytest.raises(NotImplementedError, match="TBranchElement"):
            tree.extend({"x": ak.Array([[1.5]])})
