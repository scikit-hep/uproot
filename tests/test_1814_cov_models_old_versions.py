# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Old versions of TTree, TBranch, and TLeaf: real samples written by old ROOT versions
where they exist, and real TBranches rewritten byte by byte for the serializations
that no sample has.
"""

from __future__ import annotations

import struct

import numpy
import pytest
import skhep_testdata

import uproot

kByteCount = 0x40000000

samples = [
    ("5.23.02", 16, 11),
    ("5.24.00", 16, 11),
    ("5.25.02", 17, 12),
    ("5.26.00", 18, 12),
    ("5.30.00", 19, 12),
    ("6.08.04", 19, 12),
    ("6.14.00", 20, 13),
    ("6.20.04", 20, 13),
]


def sample_path(root_version):
    return skhep_testdata.data_path(f"uproot-sample-{root_version}-uncompressed.root")


@pytest.mark.parametrize("root_version, tree_version, branch_version", samples)
@pytest.mark.parametrize("minimal", [True, False])
def test_real_samples(root_version, tree_version, branch_version, minimal):
    with uproot.open(sample_path(root_version), minimal_ttree_metadata=minimal) as f:
        tree = f["sample"]
        branch = tree["i4"]

        assert tree.class_version == tree_version
        assert branch.class_version == branch_version
        assert isinstance(tree, uproot.behaviors.TTree.TTree)

        # member_names describes what the model read
        assert tree.member_names == list(tree.members)
        assert branch.member_names == list(branch.members)
        assert tree.has_member("fIndexValues") != minimal
        assert branch.has_member("fBaskets") != minimal
        assert branch.has_member("fFileName") != minimal

        assert branch.array(library="np").tolist() == list(range(-15, 15))

        if not minimal:
            assert branch.member("fFileName") == ""
            assert len(branch.member("fBaskets")) == 6
            assert list(tree.member("fIndexValues")) == []
            assert list(tree.member("fIndex")) == []
            assert tree.member("fTreeIndex") is None
            assert tree.member("fFriends") is None
            assert tree.member("fBranchRef") is None


@pytest.mark.parametrize("root_version, tree_version, branch_version", samples[:1])
def test_tree_sizes(root_version, tree_version, branch_version):
    with uproot.open(sample_path(root_version)) as f:
        tree = f["sample"]
        assert tree.compressed_bytes == 40497
        assert tree.uncompressed_bytes == 40497
        assert tree.compression_ratio == 1.0


def test_compression_ratio():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu-zlib.root")) as f:
        tree = f["events"]
        assert tree.compressed_bytes == tree.member("fZipBytes") == 172789
        assert tree.uncompressed_bytes == tree.member("fTotBytes")
        assert tree.uncompressed_bytes > 172789
        assert tree.compression_ratio == tree.uncompressed_bytes / 172789


def test_aliases():
    with uproot.open(skhep_testdata.data_path("uproot-issue-308.root")) as f:
        aliases = f["Events"].aliases
        assert aliases["t0"] == "float_genParticles_t0_HLT.obj"
        assert aliases["Tau"] == "l1tTauBXVector_caloStage2Digis_Tau_RECO.obj"
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
        )


def test_aliases_none():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        assert f["events"].aliases == {}


def test_ttree_num_entries_model():
    filename = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(
        {filename: "events"},
        custom_classes={"TTree": uproot.models.TTree.Model_TTree_NumEntries},
    ) as tree:
        assert tree.member_names == list(tree.members) == ["fEntries"]
        assert tree.member("fEntries") == (2304,)


def test_tbranchref():
    # Event.root's "T" has a TRef branch, which requires a TBranchRef.
    filename = skhep_testdata.data_path("uproot-Event.root")
    with uproot.open(filename, minimal_ttree_metadata=False) as f:
        tree = f["T"]
        assert tree.class_version == 19
        assert tree.member("fBranchRef").name == "TRefTable"


def test_tleafg_from_root(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filename = str(tmp_path / "tleafg.root")
    file = ROOT.TFile(filename, "recreate")
    tree = ROOT.TTree("tree", "tree")
    x = numpy.zeros(1, dtype=numpy.int64)
    tree.Branch("x", x, "x/G")
    for i in range(3):
        x[0] = i * 2**40
        tree.Fill()
    file.Write()
    file.Close()

    with uproot.open(filename) as f:
        branch = f["tree"]["x"]
        assert isinstance(
            branch.member("fLeaves")[0], uproot.models.TLeaf.Model_TLeafG_v1
        )
        assert branch.array(library="np").tolist() == [0, 2**40, 2**41]


def test_tleafg(tmp_path):
    # ROOT's "/G" leaves (Long_t) are like "/L" (Long64_t) on disk.
    with open(sample_path("6.20.04"), "rb") as file:
        data = file.read()

    filename = tmp_path / "tleafg.root"
    filename.write_bytes(data.replace(b"TLeafL", b"TLeafG"))

    with uproot.open(filename) as f:
        branch = f["sample"]["i8"]
        leaf = branch.member("fLeaves")[0]
        assert isinstance(leaf, uproot.models.TLeaf.Model_TLeafG_v1)
        assert (leaf.member("fMinimum"), leaf.member("fMaximum")) == (0, 0)
        assert branch.array(library="np").tolist() == list(range(-15, 15))


class Rewritable:
    """
    The serialized form of a TTree, from which we can read the whole TTree (after
    removing its TBranches) or its first TBranch (because the others refer to classes
    defined in the bytes before them).
    """

    def __init__(self, filename, tree_name, minimal=False):
        self.file = uproot.open(filename, minimal_ttree_metadata=minimal)
        key = self.file.key(tree_name)
        chunk, cursor = key.get_uncompressed_chunk_cursor()
        self.source = chunk.source
        self.tree = self.file[tree_name]
        self.branch = self.tree.branches[0]
        self.origin = cursor.origin
        self.first = chunk.start
        self.data = bytearray(chunk.get(chunk.start, chunk.stop, cursor, {}).tobytes())
        self.start = self.branch.cursor.index - self.first

    def u32(self, where):
        return struct.unpack_from(">I", self.data, where)[0]

    def object_end(self, where):
        return where + (self.u32(where) & ~kByteCount) + 4

    def layout(self, branch=None):
        branch = self.branch if branch is None else branch
        fbaskets = branch._cursor_baskets.index - self.first
        bump1 = self.object_end(fbaskets)
        bump2 = bump1 + 1 + 4 * branch.member("fMaxBaskets")
        bump3 = bump2 + 1 + 8 * branch.member("fMaxBaskets")
        return fbaskets, bump1, bump2, bump3

    def remove(self, start, stop):
        del self.data[start:stop]
        self.add_to_size(self.start, start - stop)

    def add_to_size(self, where, change):
        size = (self.u32(where) & ~kByteCount) + change
        struct.pack_into(">I", self.data, where, size | kByteCount)

    def read(self, cls, context=None):
        chunk = uproot.source.chunk.Chunk.wrap(
            self.source,
            numpy.frombuffer(bytes(self.data), numpy.uint8),
            start=self.first,
        )
        cursor = uproot.source.cursor.Cursor(
            self.start + self.first, origin=self.origin
        )
        file = self.file.file
        parent = self.tree.parent if cls.__name__ == "Model_TTree" else self.tree
        return cls.read(
            chunk, cursor, {} if context is None else context, file, file, parent
        )

    def remove_branches(self):
        self.start = self.tree.cursor.index - self.first
        fbranches = self.tree.member("fBranches").cursor.index - self.first
        stop = self.object_end(self.tree.member("fLeaves").cursor.index - self.first)
        empty_tobjarray = struct.pack(
            ">IHHIIBii", kByteCount | 21, 3, 1, 0, 0x03000000, 0, 0, 0
        )
        change = 2 * len(empty_tobjarray) - (stop - fbranches)
        self.data[fbranches:stop] = 2 * empty_tobjarray
        self.add_to_size(self.start, change)


def downgrade_tbranch_to_v10(rewritable, tbranch):
    # Version 10 is version 11 without fFirstEntry, which is 44 bytes into the
    # fields after TAttFill. Everything before that moves forward instead of
    # everything after it moving back so that references to positions stay valid.
    data = rewritable.data
    last_base = tbranch.bases[1]
    fields = last_base.cursor.index - rewritable.first + last_base.num_bytes
    moved = data[rewritable.start : fields + 44]
    data[rewritable.start : fields + 52] = bytes(8) + moved
    rewritable.start += 8

    tbranch_start = tbranch.cursor.index - rewritable.first + 8
    rewritable.add_to_size(tbranch_start, -8)
    struct.pack_into(">H", data, tbranch_start + 4, 10)

    if tbranch_start != rewritable.start:
        # TBranchElement version 8 has a TBranch version 10 as its base.
        rewritable.add_to_size(rewritable.start, -8)
        struct.pack_into(">H", data, rewritable.start + 4, 8)


branch_cases = [
    pytest.param("5.23.02", 11, False, id="v11"),
    pytest.param("5.25.02", 12, False, id="v12"),
    pytest.param("6.14.00", 13, False, id="v13"),
    pytest.param("5.23.02", 10, True, id="v10"),
]


def rewritable_branch(root_version, downgrade, minimal):
    out = Rewritable(sample_path(root_version), "sample", minimal)
    if downgrade:
        downgrade_tbranch_to_v10(out, out.branch)
    return out


def assert_same_baskets(actual, expected, version):
    assert actual.class_version == version
    assert actual.name == expected.name
    assert actual.member("fEntries") == expected.member("fEntries")
    for name in "fBasketBytes", "fBasketEntry", "fBasketSeek":
        assert actual.member(name).dtype == expected.member(name).dtype
        assert actual.member(name).tolist() == expected.member(name).tolist()


@pytest.mark.parametrize("root_version, version, downgrade", branch_cases)
@pytest.mark.parametrize("minimal", [True, False])
def test_tbranch_as_it_is(root_version, version, downgrade, minimal):
    rewritable = rewritable_branch(root_version, downgrade, minimal)
    branch = rewritable.read(uproot.models.TBranch.Model_TBranch)
    assert_same_baskets(branch, rewritable.branch, version)
    assert branch.has_member("fFileName") != minimal
    assert branch.member_names == list(branch.members)


@pytest.mark.parametrize("root_version, version, downgrade", branch_cases)
@pytest.mark.parametrize("minimal", [True, False])
def test_tbranch_without_speedbumps(root_version, version, downgrade, minimal):
    rewritable = rewritable_branch(root_version, downgrade, minimal)
    _, bump1, bump2, bump3 = rewritable.layout()
    for bump in bump3, bump2, bump1:
        rewritable.remove(bump, bump + 1)

    branch = rewritable.read(uproot.models.TBranch.Model_TBranch, {"speedbump": False})
    assert_same_baskets(branch, rewritable.branch, version)


@pytest.mark.parametrize("root_version, version, downgrade", branch_cases)
@pytest.mark.parametrize("minimal", [True, False])
def test_tbranch_64bit_seek_flag(root_version, version, downgrade, minimal):
    rewritable = rewritable_branch(root_version, downgrade, minimal)
    _, _, _, bump3 = rewritable.layout()
    assert rewritable.data[bump3] == 1
    rewritable.data[bump3] = 2

    branch = rewritable.read(uproot.models.TBranch.Model_TBranch)
    assert_same_baskets(branch, rewritable.branch, version)


@pytest.mark.parametrize("root_version, version, downgrade", branch_cases)
def test_tbranch_fbaskets_without_byte_count(root_version, version, downgrade):
    # Without the byte count, the minimal reader can't skip over it: it has to read it.
    rewritable = rewritable_branch(root_version, downgrade, True)
    fbaskets, _, _, _ = rewritable.layout()
    rewritable.remove(fbaskets, fbaskets + 4)

    branch = rewritable.read(uproot.models.TBranch.Model_TBranch)
    assert_same_baskets(branch, rewritable.branch, version)
    assert not branch.has_member("fBaskets")


tree_cases = [
    pytest.param("5.30.00", uproot.models.TTree._ttree19_format1, 19, id="v19"),
    pytest.param("6.14.00", uproot.models.TTree._ttree20_format1, 20, id="v20"),
]


def test_tree_without_branches():
    rewritable = Rewritable(sample_path("6.14.00"), "sample", True)
    rewritable.remove_branches()

    tree = rewritable.read(uproot.models.TTree.Model_TTree)
    assert len(tree) == 0
    assert repr(tree).startswith("<TTree 'sample' at 0x")


@pytest.mark.parametrize("root_version, fields, version", tree_cases)
def test_ttree_without_speedbumps(root_version, fields, version):
    rewritable = Rewritable(sample_path(root_version), "sample", True)
    rewritable.remove_branches()

    tree = rewritable.tree
    assert tree.member("fNClusterRange") == 0
    last_base = tree.bases[-1]
    bump = last_base.cursor.index - rewritable.first + last_base.num_bytes + fields.size
    rewritable.remove(bump, bump + 2)

    result = rewritable.read(uproot.models.TTree.Model_TTree, {"speedbump": False})
    assert result.class_version == version
    assert len(result) == 0
    assert result.num_entries == tree.num_entries
    assert result.member("fEstimate") == tree.member("fEstimate")


@pytest.mark.parametrize("minimal", [True, False])
def test_tbranchelement_v8(minimal):
    filename = skhep_testdata.data_path("uproot-issue-208.root")
    rewritable = Rewritable(filename, "header", minimal)
    expected = rewritable.branch
    assert type(expected).__name__ == "Model_TBranchElement_v9"
    downgrade_tbranch_to_v10(rewritable, expected.bases[0])

    branch = rewritable.read(uproot.models.TBranch.Model_TBranchElement)
    assert_same_baskets(branch, expected, 8)
    assert branch.bases[0].class_version == 10
    for name in branch.member_names:
        assert branch.has_member(name)
    for name in "fClassName", "fCheckSum", "fID", "fType", "fStreamerType":
        assert branch.member(name) == expected.member(name)
    assert branch.member("fBranchCount") is None
    assert branch.member("fBranchCount2") is None
