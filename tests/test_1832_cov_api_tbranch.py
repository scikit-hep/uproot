# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import array
import io
import struct

import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.interpretation.grouped import AsGrouped
from uproot.interpretation.identify import UnknownInterpretation

ak = pytest.importorskip("awkward")


@pytest.fixture
def tree_path(tmp_path):
    """``t`` with the 32-bit ``x`` (0 to 9, in 2 TBaskets of 4 and 6 entries) and the long-named
    ``a_rather_long_branch_name`` (twice ``x``, as doubles)."""
    path = str(tmp_path / "tree.root")
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"x": "int32", "a_rather_long_branch_name": "float64"})
        for start, stop in ((0, 4), (4, 10)):
            x = np.arange(start, stop, dtype=np.int32)
            tree.extend({"x": x, "a_rather_long_branch_name": x * 2.0})
    return path


@pytest.fixture
def branch(tree_path):
    with uproot.open(tree_path) as file:
        yield file["t"]["x"]


def test_show_of_a_tree(tree_path):
    stream = io.StringIO()
    with uproot.open(tree_path) as file:
        file["t"].show(
            name_width=8, typename_width=7, interpretation_width=20, stream=stream
        )
    assert stream.getvalue().split("\n") == [
        "name     | typenam | interpretation      ",
        "---------+---------+---------------------",
        "x        | int32_t | AsDtype('>i4')",
        "a_rat... | double  | AsDtype('>f8')",
        "",
    ]


def test_show_of_a_branch_starts_with_the_branch(branch):
    stream = io.StringIO()
    branch.show(name_width=4, typename_width=8, interpretation_width=15, stream=stream)
    assert stream.getvalue().split("\n") == [
        "name | typename | interpretation ",
        "-----+----------+----------------",
        "x    | int32_t  | AsDtype('>i4') ",
        "",
    ]


@pytest.mark.parametrize(
    "width", ["name_width", "typename_width", "interpretation_width"]
)
def test_show_widths_must_fit_an_ellipsis(tree_path, width):
    with (
        uproot.open(tree_path) as file,
        pytest.raises(ValueError, match=f"'{width}' must be at least 3"),
    ):
        file["t"].show(**{width: 2})


def test_mapping_access(tree_path):
    with uproot.open(tree_path) as file:
        tree = file["t"]
        assert tree[0].name == "x"
        assert tree[1].name == "a_rather_long_branch_name"
        assert tree[np.int64(0)] is tree["x"]
        with pytest.raises(
            TypeError, match=r"where must be an integer or a string, not 1\.5"
        ):
            tree[1.5]
        assert [name for name, _ in tree.items()] == [
            "x",
            "a_rather_long_branch_name",
        ]
        assert tree.items()[0][1] is tree["x"]
        assert list(tree._ipython_key_completions_()) == tree.keys()


def test_filter_branch_argument(tree_path):
    with uproot.open(tree_path) as file:
        tree = file["t"]
        assert tree.keys(filter_branch=None) == tree.keys()
        with pytest.raises(
            TypeError, match="filter_branch must be None or a function: TBranch -> bool"
        ):
            tree.keys(filter_branch="x")


def test_tree_outside_of_its_open_file_has_no_source(tree_path):
    with uproot.open(tree_path) as file:
        assert file["t"].source is file.file.source
        assert file["t"]["x"].source is file.file.source

        # a tree read as part of another object is on a detached file
        key = file.key("t")
        chunk, cursor = key.get_uncompressed_chunk_cursor()
        detached = file.file.class_named("TTree").read(
            chunk,
            cursor,
            {"breadcrumbs": (), "TKey": key},
            file.file,
            file.file.detached,
            None,
        )
        assert detached.name == "t"
        assert detached.source is None


def test_repr_of_branches():
    path = skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    with uproot.open(path) as file:
        tree = file["tree"]
        parent, leaf = tree["evt"], tree["evt/I32"]
        assert repr(leaf).startswith("<TBranchElement 'I32' at 0x")
        assert repr(parent).startswith(
            f"<TBranchElement 'evt' ({len(parent)} subbranches) at 0x"
        )


def test_array_protocol(branch):
    assert np.asarray(branch).tolist() == list(range(10))
    out = np.array(branch, dtype=np.float32)
    assert out.dtype == np.float32
    assert out.tolist() == list(range(10))


JETS = "patJets_slimmedJets__PAT."


@pytest.fixture(scope="module")
def cms_events():
    """A CMS file in which the branches of some classes have no interpretation."""
    with uproot.open(skhep_testdata.data_path("uproot-issue-308.root")) as file:
        yield file["Events"]


def _named(name):
    return lambda branch: branch.name == name


def test_branch_without_interpretation(cms_events):
    jets = cms_events[JETS][JETS + "obj"]
    unknown = jets[JETS + "obj.genJetRef_.refVector_.product_"]
    assert isinstance(unknown.interpretation, UnknownInterpretation)
    assert unknown.typename == "unknown"


def test_branch_without_interpretation_is_skipped_by_a_filter(cms_events):
    name = JETS + "obj.genJetRef_.refVector_.product_"
    assert cms_events.keys(filter_branch=_named(name)) == [f"{JETS}/{JETS}obj/{name}"]
    assert cms_events.arrays(filter_branch=_named(name), library="np") == {}


def test_group_with_an_uninterpretable_member_is_skipped_by_a_filter(cms_events):
    name = JETS + "obj"
    assert isinstance(cms_events[JETS][name].interpretation, AsGrouped)
    assert cms_events.arrays(filter_branch=_named(name), library="np") == {}


def test_group_with_an_uninterpretable_nested_member_cannot_be_read(cms_events):
    with pytest.raises(UnknownInterpretation, match="none of the rules matched"):
        cms_events.arrays(filter_branch=_named(JETS), entry_stop=1, library="np")


@pytest.fixture(scope="module")
def nested_groups():
    """``branch`` groups ``TObject`` (which groups ``fUniqueID`` and ``fBits``, all zero) and
    ``value`` (0 to 9)."""
    with uproot.open(skhep_testdata.data_path("uproot-issue-1502.root")) as file:
        yield file["tree"]


def test_group_inside_a_selected_group_is_read_as_part_of_it(nested_groups):
    out = nested_groups.arrays(
        filter_branch=lambda branch: isinstance(branch.interpretation, AsGrouped),
        entry_stop=2,
    )
    assert out.tolist() == [
        {"branch": {"TObject": {"fUniqueID": 0, "fBits": 0}, "value": i}}
        for i in (0, 1)
    ]


def test_group_inside_a_group_that_is_not_selected(nested_groups):
    out = nested_groups.arrays(filter_branch=_named("TObject"), entry_stop=2)
    assert out.tolist() == [{"TObject": {"fUniqueID": 0, "fBits": 0}}] * 2


@pytest.mark.parametrize(
    "expressions",
    [["branch", "branch/TObject"], ["branch/TObject", "branch"]],
)
def test_group_and_the_group_inside_it_as_expressions(nested_groups, expressions):
    out = nested_groups.arrays(expressions, entry_stop=2)
    assert out.fields == expressions
    assert out["branch"].value.tolist() == [0, 1]
    assert out["branch/TObject"].fBits.tolist() == [0, 0]


def test_arrays_of_all_the_branches_gives_the_leaves(nested_groups):
    out = nested_groups.arrays(entry_stop=2, library="np")
    assert set(out) == {"fUniqueID", "fBits", "value"}
    assert out["value"].tolist() == [0, 1]


def test_object_interpretation_that_cannot_be_awkward(branch):
    pointer = uproot.AsObjects(uproot.containers.AsPointer(), branch)
    with pytest.raises(
        ValueError,
        match="(?s)cannot produce Awkward Arrays for .* arbitrary pointer.*"
        'instead, try library="np"',
    ):
        branch.array(pointer, library="ak")


def test_compression_ratio(tree_path):
    path = skhep_testdata.data_path("uproot-HZZ-zlib.root")
    with uproot.open(path) as file:
        branch = file["events"]["Jet_Px"]
        assert (branch.uncompressed_bytes, branch.compressed_bytes) == (20859, 14608)
        assert branch.compression_ratio == pytest.approx(20859 / 14608)
    with uproot.open(tree_path) as file:
        assert file["t"]["x"].compression_ratio == 1.0


@pytest.mark.parametrize(
    "method",
    [
        "basket",
        "basket_chunk_cursor",
        "basket_compressed_bytes",
        "basket_uncompressed_bytes",
        "basket_key",
        "basket_entry_start_stop",
    ],
)
@pytest.mark.parametrize("basket_num", [-1, 2])
def test_basket_numbers_out_of_range(branch, method, basket_num):
    with pytest.raises(
        IndexError, match=f"branch 'x' has 2 baskets; cannot get .*{basket_num}"
    ):
        getattr(branch, method)(basket_num)


@pytest.fixture
def jagged_branch(tmp_path):
    path = str(tmp_path / "jagged.root")
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"j": "var * int32"})
        tree.extend({"j": ak.Array([[1], [], [5, 6]])})
    with uproot.open(path) as file:
        yield file["t"]["j"]


def test_debug_array(jagged_branch):
    # entry 2 is the big-endian 32-bit 5, 6
    assert jagged_branch.debug_array(2).tolist() == [0, 0, 0, 5, 0, 0, 0, 6]
    assert jagged_branch.debug_array(2, skip_bytes=2).tolist() == [0, 5, 0, 0, 0, 6]
    assert jagged_branch.debug_array(2, dtype=">i4").tolist() == [5, 6]
    # truncated to a whole number of items
    assert jagged_branch.debug_array(2, skip_bytes=1, dtype=">i4").tolist() == [
        0x00000500
    ]


def test_debug_array_needs_entry_offsets(branch):
    with pytest.raises(
        uproot.deserialization.DeserializationError,
        match="missing offsets",
    ):
        branch.debug_array(3)


@pytest.mark.parametrize(
    ("kwargs", "numbers"),
    [
        ({}, "  0   0   0   5   0   0   0   6"),
        ({"skip_bytes": 4}, "  0   0   0   6"),
        ({"limit_bytes": 2}, "  0   0"),
    ],
)
def test_debug(jagged_branch, kwargs, numbers):
    stream = io.StringIO()
    jagged_branch.debug(2, stream=stream, **kwargs)
    assert stream.getvalue().split("\n")[1].rstrip() == numbers


def test_debug_interprets_a_dtype(jagged_branch):
    stream = io.StringIO()
    jagged_branch.debug(2, dtype=">i4", stream=stream)
    assert [float(x) for x in stream.getvalue().split("\n")[3].split()] == [5, 6]


def _replaced(raw, old, new, fmt, start):
    """``raw`` with the first ``old`` after ``start`` replaced by ``new``, both packed by ``fmt``."""
    old, new = struct.pack(fmt, *old), struct.pack(fmt, *new)
    where = raw.index(old, start)
    return raw[:where] + new + raw[where + len(new) :]


def test_entries_that_do_not_add_up(tree_path, tmp_path):
    with uproot.open(tree_path) as file:
        branch = file["t"]["x"]
        assert branch.num_entries == 10
        start = file.key("t").fSeekKey
        max_baskets = branch.member("fMaxBaskets")
    with open(tree_path, "rb") as stream:
        raw = stream.read()
    # fMaxBaskets, fSplitLevel and fEntries, in the first of the TBranches
    corrupted = _replaced(
        raw, (max_baskets, 0, 10), (max_baskets, 0, 11), ">iiq", start
    )
    path = tmp_path / "corrupted.root"
    path.write_bytes(corrupted)

    with uproot.open(path) as file:
        branch = file["t"]["x"]
        assert branch.num_entries == 11
        with pytest.raises(
            ValueError,
            match=r"entries in normal baskets \(10\) plus embedded baskets \(0\) "
            r"don't add up to expected number of entries \(11\)",
        ):
            branch.entry_offsets


@pytest.mark.parametrize(
    ("entry_start", "entry_stop", "expected"),
    [
        (5, 2, []),
        (-3, -8, []),
        (20, 30, []),
        (-100, 2, [0, 1]),
    ],
)
def test_entry_ranges_are_clipped(branch, entry_start, entry_stop, expected):
    out = branch.array(entry_start=entry_start, entry_stop=entry_stop, library="np")
    assert out.tolist() == expected


def test_streamer_of_an_object_branch():
    path = skhep_testdata.data_path("uproot-Event.root")
    with uproot.open(path) as file:
        tree = file["T"]
        streamers = file.file.streamers

        event = tree["event"]
        assert event.streamer is streamers["Event"][max(streamers["Event"])]
        assert event.streamer is event.streamer

        # no member of Track has this name, but fTriggerBits, a TBits, has fNbits
        nbits = tree["event/fTracks/fTracks.fTriggerBits.fNbits"]
        assert nbits.streamer.name == "fNbits"
        assert nbits.streamer.typename == "unsigned int"

        # TBits has no member named TObject, even though it inherits from it
        assert tree["event/fTriggerBits/fTriggerBits.TObject"].streamer is None


def test_streamer_of_a_member_missing_from_its_parents_class(tmp_path):
    """The TTree is moved to the end of the file, uncompressed, with one TBranch renamed."""
    path = skhep_testdata.data_path("uproot-Event.root")
    with uproot.open(path) as file:
        key = file.key("T")
        seek, key_size, size = key.fSeekKey, key.fKeylen, key.fNbytes
        chunk, _ = key.get_uncompressed_chunk_cursor()
        data = bytes(chunk.raw_data)
    with open(path, "rb") as stream:
        raw = stream.read()

    old_name, new_name = b"fTracks.fTriggerBits.fNbits", b"fTracks.fTriggerBits.fNbads"
    assert new_name not in data
    data = data.replace(old_name, new_name, 1)

    header = raw[seek : seek + key_size]
    # fNbytes at the start of the header and fSeekKey after the 4 + 2 + 4 + 4 + 2 + 2 bytes
    assert struct.unpack(">ihiIhhi", header[:22])[0] == size
    assert struct.unpack(">ihiIhhi", header[:22])[6] == seek
    moved = struct.pack(">i", key_size + len(data)) + header[4:18]
    moved += struct.pack(">i", len(raw)) + header[22:]
    # the same bytes are in the directory's list of keys
    assert raw.count(header) == 2
    corrupted = tmp_path / "renamed.root"
    corrupted.write_bytes(raw.replace(header, moved) + moved + data)

    with uproot.open(corrupted) as file:
        tracks = file["T"]["event/fTracks"]
        # fTriggerBits is a member of Track, but TBits has no fNbads
        assert tracks["fTracks.fTriggerBits.fNbads"].streamer is None
        assert tracks["fTracks.fTriggerBits.fAllBits"].streamer.name == "fAllBits"


def test_streamer_of_a_tclonesarray_branch():
    path = skhep_testdata.data_path("uproot-delphes-pr442.root")
    with uproot.open(path) as file:
        branch = file["Delphes"]["CaloJet02"]
        assert branch.streamer.name == "Jet"
        assert branch._streamer_isTClonesArray


def test_streamer_of_a_class_not_in_the_file():
    path = skhep_testdata.data_path("uproot-FCCDelphesOutput.root")
    with uproot.open(path) as file:
        assert file["events"]["bTags"].streamer is None


def test_streamer_of_a_parent_not_in_the_file(cms_events):
    name = "uint_bunchSpacingProducer__PAT."
    assert cms_events[name][name + "obj"].streamer is None


def test_plain_branch_has_no_streamer():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as file:
        assert file["events"]["M"].streamer is None


def test_virtual_arrays_with_full_paths_keep_grouped_children(nested_groups):
    out = nested_groups._virtual_arrays(entry_stop=2)
    assert "branch/value" in out.fields
    assert out["branch/value"].tolist() == [0, 1]


def test_more_normal_baskets_than_basket_entries(tmp_path):
    """ROOT can read a ``fBasketEntry`` with no entry after the last TBasket: its size comes
    from the TBasket's header. Here, fMaxBaskets = 3 and the third TBasket is that of ``other``,
    as uproot always leaves one free."""
    path = str(tmp_path / "full.root")
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"x": "int32"}, initial_basket_capacity=3)
        tree.extend({"x": np.arange(0, 4, dtype=np.int32)})
        tree.extend({"x": np.arange(4, 10, dtype=np.int32)})
        other = file.mktree("other", {"x": "int32"})
        other.extend({"x": np.arange(100, 106, dtype=np.int32)})

    with uproot.open(path) as file:
        branch = file["t"]["x"]
        assert branch.member("fMaxBaskets") == 3
        assert branch.member("fWriteBasket") == 2
        seeks = branch.member("fBasketSeek").tolist()
        sizes = branch.member("fBasketBytes").tolist()
        extra_seek = int(file["other"]["x"].member("fBasketSeek")[0])
        extra_size = int(file["other"]["x"].member("fBasketBytes")[0])
        start = file.key("t").fSeekKey

    with open(path, "rb") as stream:
        raw = stream.read()

    raw = _replaced(raw, (*seeks[:2], 0), (*seeks[:2], extra_seek), ">3q", start)
    raw = _replaced(raw, (*sizes[:2], 0), (*sizes[:2], extra_size), ">3i", start)
    # fWriteBasket and fEntryNumber
    raw = _replaced(raw, (2, 10), (3, 16), ">iq", start)
    # fMaxBaskets, fSplitLevel and fEntries
    raw = _replaced(raw, (3, 0, 10), (3, 0, 16), ">iiq", start)
    # the TTree's fEntries comes before all of the above
    raw = _replaced(raw, (10,), (16,), ">q", start)

    corrupted = tmp_path / "corrupted.root"
    corrupted.write_bytes(raw)
    with uproot.open(corrupted) as file:
        branch = file["t"]["x"]
        assert branch.num_baskets == 3
        assert branch.embedded_baskets == []
        assert branch.entry_offsets == [0, 4, 10, 16]
        expected = [*range(10), *range(100, 106)]
        assert branch.array(library="np").tolist() == expected


@pytest.fixture(scope="module", params=[True, False], ids=["minimal", "all metadata"])
def minimal_ttree_metadata(request):
    return request.param


def _fill(ROOT, tree, stop, start=0):
    value = array.array("i", [0])
    tree.SetBranchAddress("x", value)
    for i in range(start, stop):
        value[0] = i
        tree.Fill()


def _new_tree(ROOT):
    tree = ROOT.TTree("t", "t")
    tree.SetAutoFlush(0)
    value = array.array("i", [0])
    tree.Branch("x", value, "x/I").SetBasketSize(4096)
    return tree


@pytest.fixture(scope="module")
def autosaved(tmp_path_factory):
    """As left by a ROOT process that died: the last TBasket is still in the TBranch metadata."""
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path_factory.mktemp("embedded") / "autosaved.root")
    file = ROOT.TFile(path, "RECREATE")
    tree = _new_tree(ROOT)
    _fill(ROOT, tree, 3100)
    tree.AutoSave("SaveSelf")
    file.Close()
    return path


@pytest.fixture(scope="module")
def memory_resident(tmp_path_factory):
    """A TTree that was never attached to a file has all of its TBaskets in the metadata."""
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path_factory.mktemp("embedded") / "memory.root")
    file = ROOT.TFile(path, "RECREATE")
    tree = _new_tree(ROOT)
    tree.SetDirectory(ROOT.nullptr)
    _fill(ROOT, tree, 5000)
    file.WriteObject(tree, "t")
    file.Close()
    return path


def test_embedded_basket_after_normal_baskets(autosaved, minimal_ttree_metadata):
    with uproot.open(autosaved, minimal_ttree_metadata=minimal_ttree_metadata) as file:
        branch = file["t"]["x"]
        normal = branch.num_baskets - 1
        assert normal > 0
        assert len(branch.embedded_baskets) == 1
        embedded = branch.embedded_baskets[0]

        offsets = branch.entry_offsets
        assert offsets[-1] == 3100
        assert branch.basket(normal) is embedded
        assert embedded.basket_num == normal
        assert embedded.num_entries == offsets[-1] - offsets[-2]
        assert branch.basket_entry_start_stop(normal) == (offsets[-2], 3100)
        assert branch.basket_compressed_bytes(normal) == embedded.compressed_bytes
        assert branch.basket_uncompressed_bytes(normal) == embedded.uncompressed_bytes

        with pytest.raises(IndexError, match="only normal baskets have cursors"):
            branch.basket_chunk_cursor(normal)
        with pytest.raises(
            ValueError, match="is an embedded basket, which has no TKey"
        ):
            branch.basket_key(normal)

        assert branch.array(library="np").tolist() == list(range(3100))
        # across the boundary between normal and embedded baskets
        middle = offsets[-2]
        assert branch.array(
            entry_start=middle - 2, entry_stop=middle + 2, library="np"
        ).tolist() == list(range(middle - 2, middle + 2))

        # all of the baskets count for a step in bytes
        total = sum(branch.basket_uncompressed_bytes(i) for i in range(normal + 1))
        assert file["t"].num_entries_for(f"{total} B", ["x"]) == 3100


def test_several_embedded_baskets(memory_resident, minimal_ttree_metadata):
    with uproot.open(
        memory_resident, minimal_ttree_metadata=minimal_ttree_metadata
    ) as file:
        branch = file["t"]["x"]
        assert branch.num_baskets == len(branch.embedded_baskets) > 1
        assert [b.basket_num for b in branch.embedded_baskets] == list(
            range(branch.num_baskets)
        )

        offsets = branch.entry_offsets
        assert offsets[0] == 0
        assert offsets[-1] == 5000
        assert [
            branch.basket_entry_start_stop(i) for i in range(branch.num_baskets)
        ] == list(zip(offsets[:-1], offsets[1:], strict=True))

        assert branch.array(library="np").tolist() == list(range(5000))
        steps = list(file["t"].iterate(["x"], step_size=1500, library="np"))
        assert [len(step["x"]) for step in steps] == [1500, 1500, 1500, 500]


def test_subbranches_with_the_same_name(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "duplicates.root")

    file = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("t", "t")
    values = [array.array("i", [0]) for _ in range(3)]
    parent = tree.Branch("p", values[0], "p/I")
    ignore_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kError  # ROOT warns about the second "c"
    try:
        first = tree.Branch("c", values[1], "c/I")
        second = tree.Branch("c", values[2], "c/I")
    finally:
        ROOT.gErrorIgnoreLevel = ignore_level
    for i in range(3):
        values[0][0], values[1][0], values[2][0] = i, i * 10, i * 100
        tree.Fill()
    for branch in (first, second):
        tree.GetListOfBranches().Remove(branch)
        parent.GetListOfBranches().Add(branch)
    tree.Write()
    file.Close()

    with uproot.open(path) as file:
        parent = file["t"]["p"]
        assert [branch.name for branch in parent.branches] == ["c", "c"]
        # a name finds the first; the second is reachable by its index
        assert parent["c"] is parent[0]
        assert parent["c"].array(library="np").tolist() == [0, 10, 20]
        assert parent[1].array(library="np").tolist() == [0, 100, 200]
