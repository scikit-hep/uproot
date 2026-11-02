# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of deserialization through uproot.interpretation.objects and the read
methods of uproot.containers.

Where a model has both a Python reader and an AwkwardForth reader, the test
reads the same bytes with both and requires identical results. The raw bytes
are put in the entries of a real file by writing them as a jagged array of
uint8 and reading that TBranch with the interpretation under test.
"""

from __future__ import annotations

import pickle
import struct
import threading
import types

import numpy
import pytest
import skhep_testdata

import awkward
import uproot
from uproot.containers import (
    AsArray,
    AsBitSet,
    AsList,
    AsMap,
    AsPointer,
    AsRVec,
    AsSet,
    AsString,
    AsVector,
    STLMap,
    STLVector,
)
from uproot.interpretation.identify import interpretation_of
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.objects import (
    AsObjects,
    AsStridedObjects,
    CannotBeAwkward,
    ObjectArray,
    StridedObjectArray,
)
from uproot.interpretation.strings import AsStrings
from uproot.models.TObject import Model_TObject

i4 = numpy.dtype(">i4")
f8 = numpy.dtype(">f8")
kByteCountMask = 0x40000000
kNewClassTag = 0xFFFFFFFF


def ints(*values):
    return struct.pack(f">{len(values)}i", *values)


def size(n):
    return struct.pack(">I", n)


def header(payload, version=9):
    """The 6-byte (num_bytes, version) header that precedes objects."""
    return struct.pack(">IH", kByteCountMask | (len(payload) + 2), version) + payload


def memberwise(payload, version=9):
    """A header with the memberwise bit set; 6 more bytes are not interpreted."""
    return header(b"\x00" * 6 + payload, version | 0x4000)


def string(text):
    if len(text) < 255:
        return bytes([len(text)]) + text
    else:
        return b"\xff" + size(len(text)) + text


def as_uint8_jagged(entries):
    return awkward.unflatten(
        numpy.frombuffer(b"".join(entries), dtype=numpy.uint8),
        [len(x) for x in entries],
    )


@pytest.fixture
def raw_branch(tmp_path):
    """
    Makes a real TBranch with the given bytes in each entry: every argument is a
    list of entries that goes into a separate TBasket.
    """
    opened = []

    def make(*baskets):
        path = tmp_path / f"raw{len(opened)}.root"
        with uproot.recreate(path) as file:
            tree = file.mktree("tree", {"x": "var * uint8"})
            for entries in baskets:
                tree.extend({"x": as_uint8_jagged(entries)})
        opened.append(uproot.open(path))
        return opened[-1]["tree"]["x"]

    yield make

    for file in opened:
        file.close()


def as_python(item):
    """Deserialized Python objects as what ak.Array.tolist() would show."""
    if isinstance(item, STLMap):
        return [(as_python(k), as_python(v)) for k, v in item.items()]
    elif isinstance(item, (str, bytes)):
        return item
    elif hasattr(item, "__iter__"):
        return [as_python(x) for x in item]
    elif isinstance(item, numpy.generic):
        return item.item()
    else:
        return item


def read_both_ways(branch, model):
    """
    Reads with the Python reader and with the AwkwardForth reader, checking that
    the second is really AwkwardForth and that the results are the same.
    """
    python = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )

    interpretation = AsObjects(model, branch)
    forth = branch.array(interpretation=interpretation, library="ak", array_cache=None)
    assert interpretation._complete_forth_code is not None
    assert interpretation._forth is True

    assert as_python(python) == forth.tolist()
    return python, forth


@pytest.mark.parametrize(
    "model, entries, expected",
    [
        (
            AsVector(False, i4),
            [size(0), size(2) + ints(1, 2), size(1) + ints(-3)],
            [[], [1, 2], [-3]],
        ),
        (
            AsVector(True, i4),
            [header(size(2) + ints(1, 2)), header(size(0)), header(size(1) + ints(9))],
            [[1, 2], [], [9]],
        ),
        (
            AsRVec(True, f8),
            [header(size(1) + struct.pack(">d", 1.5)), header(size(0))],
            [[1.5], []],
        ),
        (
            AsList(False, i4),
            [size(2) + ints(7, 8), size(0)],
            [[7, 8], []],
        ),
        (
            AsSet(False, i4),
            [size(2) + ints(1, 2), size(1) + ints(5)],
            [[1, 2], [5]],
        ),
        (
            AsBitSet(False, 3),
            [size(3) + b"\x01\x00\x01", size(3) + b"\x00\x00\x01"],
            [[True, False, True], [False, False, True]],
        ),
        (
            AsVector(False, AsString(False)),
            [size(2) + string(b"ab") + string(b"c"), size(0)],
            [["ab", "c"], []],
        ),
        (
            AsVector(False, AsString(False)),
            [size(1) + string(b"x" * 300), size(1) + string(b"y")],
            [["x" * 300], ["y"]],
        ),
        (
            AsVector(False, AsString(True)),
            [size(2) + header(string(b"ab")) + header(string(b"")), size(0)],
            [["ab", ""], []],
        ),
        (
            AsVector(False, AsString(False, length_bytes="4")),
            [size(1) + size(3) + b"abc", size(2) + size(0) + size(1) + b"d"],
            [["abc"], ["", "d"]],
        ),
        (
            AsVector(False, AsVector(False, i4)),
            [size(2) + size(1) + ints(1) + size(0), size(1) + size(2) + ints(2, 3)],
            [[[1], []], [[2, 3]]],
        ),
        (
            AsVector(True, AsVector(True, i4)),
            [
                header(size(1) + header(size(1) + ints(1))),
                header(size(2) + header(size(0)) + header(size(1) + ints(2))),
            ],
            [[[1]], [[], [2]]],
        ),
        (
            AsMap(True, i4, f8),
            [
                memberwise(
                    size(2) + ints(1, 2) + struct.pack(">dd", 1.5, 2.5),
                ),
                memberwise(size(1) + ints(3) + struct.pack(">d", 3.5)),
            ],
            [[(1, 1.5), (2, 2.5)], [(3, 3.5)]],
        ),
        (
            AsMap(True, AsString(False), i4),
            [
                memberwise(size(2) + string(b"a") + string(b"bc") + ints(1, 2)),
                memberwise(size(1) + string(b"d") + ints(3)),
            ],
            [[("a", 1), ("bc", 2)], [("d", 3)]],
        ),
        (
            AsMap(True, i4, AsString(False)),
            [
                memberwise(size(2) + ints(1, 2) + string(b"a") + string(b"bc")),
                memberwise(size(1) + ints(3) + string(b"d")),
            ],
            [[(1, "a"), (2, "bc")], [(3, "d")]],
        ),
        (
            AsMap(True, AsString(True), AsVector(True, i4)),
            [
                memberwise(
                    size(2)
                    + b"\x00" * 6
                    + string(b"a")
                    + string(b"b")
                    + b"\x00" * 6
                    + size(1)
                    + ints(1)
                    + size(2)
                    + ints(2, 3)
                ),
            ],
            [[("a", [1]), ("b", [2, 3])]],
        ),
        (
            AsArray(True, False, AsString(False)),
            [header(string(b"a") + string(b"bc")), header(string(b"d"))],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, False, AsString(False)),
            [string(b"a") + string(b"bc"), string(b"d")],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, True, AsString(False)),
            [b"\x00" + string(b"a") + string(b"bc"), b"\x00" + string(b"d")],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, False, AsString(False), (2,)),
            [
                string(b"a") + string(b"b") + string(b"c") + string(b"d"),
                string(b"e") + string(b"f"),
            ],
            [[["a", "b"], ["c", "d"]], [["e", "f"]]],
        ),
        (
            AsArray(True, False, AsString(False), (2,)),
            [
                header(string(b"a") + string(b"b") + string(b"c") + string(b"d")),
                header(string(b"e") + string(b"f")),
            ],
            [[["a", "b"], ["c", "d"]], [["e", "f"]]],
        ),
    ],
    ids=lambda x: x.cache_key if hasattr(x, "cache_key") else "",
)
def test_python_and_forth_agree(raw_branch, model, entries, expected):
    python, forth = read_both_ways(raw_branch(entries), model)
    assert forth.tolist() == expected
    assert isinstance(python, numpy.ndarray)
    assert python.dtype == numpy.dtype(object)


@pytest.mark.parametrize(
    "branch_name",
    [
        "vector_int32",
        "vector_string",
        "vector_vector_int32",
        "vector_vector_string",
        "vector_set_int32",
        "vector_set_string",
        "set_int32",
        "set_string",
        "map_int32_int16",
        "map_int32_vector_int16",
        "map_int32_vector_string",
        "map_int32_set_int16",
        "map_int32_set_string",
        "map_string_int16",
        "map_string_vector_int16",
        "map_string_vector_string",
        "map_string_set_int16",
        "map_string_set_string",
        "map_int32_vector_vector_int16",
        "map_int32_vector_set_int16",
        "map_string_string",
        "map_string_tstring",
        "vector_tstring",
        "vector_vector_tstring",
    ],
)
def test_python_and_forth_agree_on_real_stl_containers(branch_name):
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree"][branch_name]
        interpretation = interpretation_of(branch, {}, simplify=False)
        assert isinstance(interpretation, AsObjects)

        read_both_ways(branch, interpretation.model)


@pytest.mark.parametrize(
    "model, entries, expected",
    [
        # Forth is not attempted for non-memberwise maps
        (
            AsMap(False, i4, i4),
            [size(2) + ints(2, 20, 1, 10), size(0), size(1) + ints(5, 50)],
            [{1: 10, 2: 20}, {}, {5: 50}],
        ),
        (
            AsMap(True, i4, i4),
            [header(size(2) + ints(2, 20, 1, 10)), header(size(0))],
            [{1: 10, 2: 20}, {}],
        ),
        # strings of non-memberwise maps have no headers, whatever the AsString says
        (
            AsMap(True, AsString(True), AsString(True)),
            [
                header(
                    size(2) + string(b"b") + string(b"B") + string(b"a") + string(b"A")
                )
            ],
            [{"a": "A", "b": "B"}],
        ),
        (
            AsMap(False, i4, AsString(False)),
            [size(2) + ints(2) + string(b"b") + ints(1) + string(b"a")],
            [{1: "a", 2: "b"}],
        ),
        # NumPy dtypes without AwkwardForth symbols
        (
            AsVector(False, numpy.dtype("<i4")),
            [size(2) + struct.pack("<ii", 1, 2), size(0)],
            [[1, 2], []],
        ),
        (
            AsMap(True, numpy.dtype("<i4"), i4),
            [memberwise(size(1) + struct.pack("<i", 1) + ints(2))],
            [{1: 2}],
        ),
        # arrays of fixed-width numbers should have been simplified
        (
            AsArray(False, False, i4),
            [ints(1, 2), ints(3)],
            [[1, 2], [3]],
        ),
        (
            AsArray(False, True, i4),
            [b"\x00" + ints(1, 2), b"\x00" + ints(3)],
            [[1, 2], [3]],
        ),
        (
            AsArray(True, False, i4),
            [header(ints(1, 2)), header(ints(3))],
            [[1, 2], [3]],
        ),
    ],
    ids=lambda x: x.cache_key if hasattr(x, "cache_key") else "",
)
def test_models_that_cannot_be_forth(raw_branch, model, entries, expected):
    branch = raw_branch(entries)

    python = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )
    assert [x.tolist() for x in python] == expected

    interpretation = AsObjects(model, branch)
    forth = branch.array(interpretation=interpretation, library="ak", array_cache=None)
    assert interpretation._forth is False
    assert interpretation._complete_forth_code is None
    if isinstance(model, AsMap):
        assert [dict(x) for x in forth.tolist()] == expected
    else:
        assert forth.tolist() == expected


def test_map_read_empty_has_no_keys(raw_branch):
    branch = raw_branch([size(0)])
    result = branch.array(
        interpretation=AsObjects(AsMap(False, i4, i4), branch), library="np"
    )
    assert isinstance(result[0], STLMap)
    assert len(result[0]) == 0
    assert result[0].tolist() == {}


def test_array_with_header_does_not_read_past_itself(raw_branch):
    # two arrays in the same entry: each one has to stop where its header says
    entry = header(ints(1, 2)) + header(ints(3))
    model = AsVector(False, AsArray(True, False, i4))
    branch = raw_branch([size(2) + entry])

    result = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )
    assert [x.tolist() for x in result[0]] == [[1, 2], [3]]


def test_header_check_of_array_of_objects(raw_branch):
    # the header claims that there are 2 bytes in the array, but the string has 4
    entry = struct.pack(">IH", kByteCountMask | 4, 9) + string(b"abc")
    branch = raw_branch([entry])

    with pytest.raises(uproot.deserialization.DeserializationError) as excinfo:
        branch.array(
            interpretation=AsObjects(AsArray(True, False, AsString(False)), branch),
            library="np",
        )
    assert excinfo.value.message == (
        "expected 8 bytes but cursor moved by 10 bytes (through std::string[])"
    )


def test_memberwise_array_is_not_implemented(raw_branch):
    branch = raw_branch([memberwise(ints(1))])
    with pytest.raises(NotImplementedError) as excinfo:
        branch.array(
            interpretation=AsObjects(AsArray(True, False, i4), branch), library="np"
        )
    assert str(excinfo.value) == (
        f"memberwise serialization of AsArray\nin file {branch.file.file_path}"
    )


def test_memberwise_vector_of_nothing_with_streamers():
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as file:
        cls = file.file.class_named(
            [x for x in file.file.streamers if x.startswith("pair<")][0]
        )
        assert issubclass(cls, uproot.model.DispatchByVersion)

        data = numpy.frombuffer(memberwise(size(0)), dtype=numpy.uint8)
        chunk = uproot.source.chunk.Chunk.wrap(file.file.source, data)
        out = AsVector(True, cls).read(
            chunk,
            uproot.source.cursor.Cursor(0),
            {"breadcrumbs": ()},
            file.file,
            file.file.detached,
            None,
        )
    assert isinstance(out, STLVector)
    assert len(out) == 0


def uint8_entries(n, start=0):
    return [bytes([start + i]) * 4 for i in range(n)]


def vector_of_vectors(*lists):
    out = size(len(lists))
    for values in lists:
        out += size(len(values)) + ints(*values)
    return out


nested_model = AsVector(False, AsVector(False, i4))


def mixed_baskets(raw_branch):
    # In the first basket, the type of the innermost vector is unknown, so it
    # takes Python; the second basket reveals it, so that it takes Forth.
    return raw_branch(
        [vector_of_vectors(), vector_of_vectors()],
        [vector_of_vectors([1, 2]), vector_of_vectors([], [3])],
    )


@pytest.mark.parametrize("entry_start, entry_stop", [(0, 4), (1, 3), (0, 1), (2, 4)])
def test_baskets_of_python_and_forth_in_awkward(raw_branch, entry_start, entry_stop):
    branch = mixed_baskets(raw_branch)
    interpretation = AsObjects(nested_model, branch)

    result = branch.array(
        interpretation=interpretation,
        library="ak",
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    # the Forth code is only discovered in the second basket
    assert (interpretation._complete_forth_code is not None) is (entry_stop > 2)
    expected = [[], [], [[1, 2]], [[], [3]]]
    assert result.tolist() == expected[entry_start:entry_stop]


def test_baskets_of_python_and_forth_in_pandas(raw_branch):
    pytest.importorskip("pandas")
    pytest.importorskip("awkward_pandas")

    branch = mixed_baskets(raw_branch)
    interpretation = AsObjects(nested_model, branch)

    result = branch.array(interpretation=interpretation, library="pd")

    assert interpretation._complete_forth_code is not None
    assert [x.tolist() for x in result] == [[], [], [[1, 2]], [[], [3]]]


def test_final_array_of_awkward_baskets_in_numpy(raw_branch):
    branch = raw_branch(uint8_entries(2))
    interpretation = AsObjects(AsVector(False, i4), branch)

    result = interpretation.final_array(
        {0: numpy.array([1, 2, 3]), 1: awkward.Array([4, 5])},
        1,
        4,
        [0, 3, 5],
        uproot.interpretation.library._libraries["np"],
        branch,
        {},
    )
    assert result.tolist() == [2, 3, 4]


@pytest.mark.parametrize("library", ["np", "ak"])
def test_final_array_without_baskets(raw_branch, library):
    branch = raw_branch([size(1) + ints(1)])
    result = branch.array(
        interpretation=AsObjects(AsVector(False, AsString(False)), branch),
        library=library,
        entry_start=1,
        entry_stop=1,
    )
    assert result.tolist() == []


class ForthIsGivenUpOnElsewhere(AsObjects):
    # same as another thread giving up on Forth just before this one gets the lock
    def hook_before_basket_array(self, *args, **kwargs):
        self.hook_calls = getattr(self, "hook_calls", 0) + 1
        if self.hook_calls == 2:
            self._forth = False


def test_forth_is_abandoned_by_another_thread(raw_branch):
    branch = raw_branch([size(2) + ints(1, 2)])
    interpretation = ForthIsGivenUpOnElsewhere(AsVector(False, i4), branch)

    result = branch.array(interpretation=interpretation, library="ak")

    assert interpretation._forth is False
    assert interpretation._complete_forth_code is None
    assert result.tolist() == [[1, 2]]


class OtherThreadFinishes:
    """
    A lock whose waiter finds that the holder has already finished generating
    the Forth code.
    """

    def __init__(self, interpretation, code, form):
        self.interpretation = interpretation
        self.code = code
        self.form = form

    def __enter__(self):
        self.interpretation._complete_forth_code = self.code
        self.interpretation._form = self.form

    def __exit__(self, *args):
        pass


def test_forth_is_completed_by_another_thread(raw_branch):
    branch = raw_branch([size(2) + ints(1, 2), size(0)])

    finished = AsObjects(AsVector(False, i4), branch)
    expected = branch.array(interpretation=finished, library="ak", array_cache=None)
    assert expected.tolist() == [[1, 2], []]

    interpretation = AsObjects(AsVector(False, i4), branch)
    interpretation._forth_lock = OtherThreadFinishes(
        interpretation, finished._complete_forth_code, finished._form
    )
    result = branch.array(interpretation=interpretation, library="ak", array_cache=None)

    assert result.tolist() == [[1, 2], []]


def test_invalid_forth_code_is_reported(raw_branch):
    branch = raw_branch([size(1) + ints(1)])
    interpretation = AsObjects(AsVector(False, i4), branch)
    interpretation._complete_forth_code = "input stream\nnot-a-forth-word\n"
    interpretation._form = {"class": "NumpyArray", "primitive": "int32"}

    with pytest.raises(
        ValueError,
        match=r"(?s)Forth code generated for this data type:.*not-a-forth-word",
    ):
        branch.array(interpretation=interpretation, library="ak")


@pytest.mark.parametrize(
    "form, expected",
    [
        ("NULL", True),
        ({"class": "NumpyArray"}, False),
        ({"content": "NULL"}, True),
        ({"content": {"content": {"class": "NumpyArray"}}}, False),
        ({"contents": []}, False),
        ({"contents": [{"class": "NumpyArray"}, "NULL"]}, True),
        ({"contents": {"x": {"class": "NumpyArray"}}}, False),
        ({"contents": {"x": {"class": "NumpyArray"}, "y": "NULL"}}, True),
        ({"contents": {"x": {"content": "NULL"}}}, True),
    ],
)
def test_any_null(form, expected):
    assert AsObjects(AsVector(False, i4))._any_NULL(form) is expected


def test_debug_forth(raw_branch):
    branch = raw_branch([size(1) + ints(1)])
    interpretation = AsObjects(AsVector(False, i4), branch)

    generator = uproot._awkwardforth.ForthGenerator(interpretation)
    context = {"forth": types.SimpleNamespace(gen=generator), "path": ()}
    data = numpy.frombuffer(size(1) + ints(7), dtype=numpy.uint8)
    chunk = uproot.source.chunk.Chunk.wrap(branch.file.source, data)
    read = interpretation.model.read(
        chunk,
        uproot.source.cursor.Cursor(0),
        context,
        branch.file,
        branch.file.detached,
        branch,
    )
    assert read.tolist() == [7]

    with pytest.raises(
        Exception,
        match=r'(?s)^EXPECTED FORM:\n.*"content": "int32".*DISCOVERED FORM:\n.*"primitive": "int32".*FORTH CODE:\ninput stream\n.*stream !I-> stack',
    ):
        generator._debug_forth()


def object_array(branch, *entries, cursor_offset=0):
    return ObjectArray(
        AsVector(False, i4),
        branch,
        dict(branch.context),
        numpy.cumsum([0] + [len(x) for x in entries]),
        numpy.frombuffer(b"".join(entries), dtype=numpy.uint8),
        cursor_offset,
    )


def test_object_array(raw_branch):
    branch = raw_branch([size(0)])
    entries = [size(1) + ints(1), size(0), size(2) + ints(2, 3)]
    array = object_array(branch, *entries, cursor_offset=7)

    assert array.model == AsVector(False, i4)
    assert array.branch is branch
    assert array.context["in_TBranch"] is True
    assert array.byte_offsets.tolist() == [0, 8, 12, 24]
    assert array.byte_content.tobytes() == b"".join(entries)
    assert array.cursor_offset == 7
    assert repr(array).startswith("ObjectArray(AsVector(False, dtype('>i4')), <TBranch")

    assert len(array) == 3
    assert array[0].tolist() == [1]
    assert array[2].tolist() == [2, 3]
    assert [x.tolist() for x in array] == [[1], [], [2, 3]]
    assert [x.tolist() for x in array.to_numpy()] == [[1], [], [2, 3]]


@pytest.mark.parametrize(
    "where, expected",
    [
        (slice(None), [[1], [], [2, 3]]),
        (slice(1, None), [[], [2, 3]]),
        (slice(0, 2), [[1], []]),
        (slice(1, 2), [[]]),
        (slice(-1, None), [[2, 3]]),
        (slice(None, -1), [[1], []]),
        (slice(2, 1), []),
        (slice(5, 7), []),
    ],
)
def test_object_array_slices(raw_branch, where, expected):
    branch = raw_branch([size(0)])
    array = object_array(
        branch, size(1) + ints(1), size(0), size(2) + ints(2, 3), cursor_offset=7
    )
    sliced = array[where]

    assert isinstance(sliced, ObjectArray)
    assert len(sliced) == len(expected)
    assert [x.tolist() for x in sliced] == expected


@pytest.mark.parametrize("where", [slice(None, None, 2), "name", [0, 1]])
def test_object_array_unsupported_selections(raw_branch, where):
    branch = raw_branch([size(0)])
    array = object_array(branch, size(0), size(0))

    with pytest.raises(NotImplementedError) as excinfo:
        array[where]
    assert str(excinfo.value) == repr(where)


def test_object_array_iteration_finds_references(raw_branch):
    # Each entry is a vector of two pointers to the same TObjString: the second
    # one is a reference to the position of the first in the whole buffer.
    branch = raw_branch([size(0)])
    kMapOffset = 2

    def entry(position, text):
        tobject = struct.pack(">HII", 1, 0, 0x03000000)
        instance = header(tobject + string(text), version=1)
        new_object = struct.pack(">II", 0, kNewClassTag) + b"TObjString\x00" + instance
        new_object = (
            struct.pack(">I", kByteCountMask | (len(new_object) - 4)) + new_object[4:]
        )
        reference = struct.pack(">I", position + 4 + kMapOffset)
        return size(2) + new_object + reference

    first = entry(0, b"first")
    second = entry(len(first), b"second")

    array = ObjectArray(
        AsVector(False, AsPointer()),
        branch,
        dict(branch.context),
        numpy.array([0, len(first), len(first) + len(second)]),
        numpy.frombuffer(first + second, dtype=numpy.uint8),
        0,
    )

    by_index = [array[i] for i in range(2)]
    by_iteration = list(array)
    assert [[str(x) for x in vector] for vector in by_index] == [
        ["first", "first"],
        ["second", "second"],
    ]
    assert [[str(x) for x in vector] for vector in by_iteration] == [
        ["first", "first"],
        ["second", "second"],
    ]


def test_cannot_be_awkward_because():
    assert CannotBeAwkward("some reason").because == "some reason"


def test_as_objects_model_and_repr():
    model = AsVector(False, i4)
    interpretation = AsObjects(model)

    assert interpretation.model is model
    assert repr(interpretation) == "AsObjects(AsVector(False, dtype('>i4')))"
    assert interpretation.numpy_dtype == numpy.dtype(object)
    assert interpretation.typename == "std::vector<int32_t>"
    assert interpretation.cache_key == "AsObjects(AsVector(False,Bi4))"


def test_as_objects_with_class_model():
    interpretation = AsObjects(Model_TObject)

    assert repr(interpretation) == "AsObjects(Model_TObject)"
    assert interpretation.typename == "TObject"
    assert interpretation.cache_key == "AsObjects(Model_TObject)"


def test_as_objects_eq_of_arrays():
    assert AsObjects(AsArray(True, False, i4)) == AsObjects(AsArray(True, False, i4))
    assert AsObjects(AsArray(True, False, i4)) != AsObjects(AsArray(False, False, i4))


def test_simplify_without_branch():
    simple = AsObjects(Model_TObject).simplify()
    assert isinstance(simple, AsObjects)

    # no file means no streamers to find out if the items are fixed-width
    array = AsArray(False, False, Model_TObject)
    simple = AsObjects(array).simplify()
    assert isinstance(simple, AsObjects)
    assert simple.model == array

    jagged = AsObjects(AsArray(False, True, i4)).simplify()
    assert isinstance(jagged, AsJagged)
    assert jagged.header_bytes == 1
    assert jagged.original == AsArray(False, True, i4)

    jagged = AsObjects(AsVector(True, i4)).simplify()
    assert jagged.header_bytes == 10

    strings = AsObjects(AsString(True)).simplify()
    assert isinstance(strings, AsStrings)
    assert strings.header_bytes == 6


class Model_NotAnAsDtype(uproot.model.Model):
    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        return uproot.interpretation.numerical.AsFloat16(0, 1, 12)


def test_simplify_arrays_of_unshapeable_content(raw_branch):
    branch = raw_branch([size(0)])
    interpretation = AsObjects(AsArray(False, False, Model_NotAnAsDtype, (2,)), branch)
    with pytest.raises(RuntimeError, match=r"^I shouldn't be here$"):
        interpretation.simplify()


def strided_interpretation():
    return AsStridedObjects(
        Model_TObject, [("a", numpy.dtype(">i4")), ("b", numpy.dtype(">f8"))]
    )


def test_strided_object_array():
    data = numpy.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=[("a", "i4"), ("b", "f8")])
    array = StridedObjectArray(strided_interpretation(), data)

    assert array.array is data
    assert array.shape == (3,)
    assert len(array) == 3
    assert repr(array).startswith("StridedObjectArray(AsStridedObjects(Model_TObject)")

    item = array[1]
    assert isinstance(item, Model_TObject)
    assert item.member("a") == 2
    assert item.member("b") == 2.5

    part = array[1:]
    assert isinstance(part, StridedObjectArray)
    assert [x.member("a") for x in part] == [2, 3]

    assert [x.member("b") for x in array] == [1.5, 2.5, 3.5]
    assert [(i, x.member("a")) for i, x in array.ndenumerate()] == [
        ((0,), 1),
        ((1,), 2),
        ((2,), 3),
    ]


def test_strided_object_array_of_two_dimensions():
    data = numpy.array(
        [[(1, 1.5), (2, 2.5)], [(3, 3.5), (4, 4.5)]],
        dtype=[("a", "i4"), ("b", "f8")],
    )
    interpretation = strided_interpretation()
    interpretation.reshape((2,))
    array = StridedObjectArray(interpretation, data)

    assert array.shape == (2, 2)
    rows = list(array)
    assert all(isinstance(row, StridedObjectArray) for row in rows)
    assert [x.member("a") for x in rows[1]] == [3, 4]


def test_as_strided_objects_properties():
    interpretation = strided_interpretation()

    assert repr(interpretation) == "AsStridedObjects(Model_TObject)"
    assert interpretation.model is Model_TObject
    assert interpretation.original is None
    assert interpretation.numpy_dtype == numpy.dtype(object)
    assert interpretation.typename == "TObject"
    assert interpretation.cache_key == "AsStridedObjects(Model_TObject)"

    interpretation.reshape((2, 3))
    assert repr(interpretation) == "AsStridedObjects(Model_TObject, (2, 3))"


def test_strided_objects_from_a_file():
    path = skhep_testdata.data_path("uproot-issue-407.root")
    with uproot.open(path) as file:
        branch = file["tree/branch"]
        interpretation = branch.interpretation

        assert isinstance(interpretation, AsStridedObjects)
        assert interpretation.original is not None
        assert interpretation.model == branch.file.class_named("TDatime", 2)

        everything = branch.array(library="np")
        assert len(everything) == 2
        assert all(isinstance(x, interpretation.model) for x in everything)

        for library in "np", "ak":
            nothing = branch.array(library=library, entry_start=1, entry_stop=1)
            assert len(nothing) == 0


def test_strided_objects_without_public_members_keep_their_length():
    interpretation = AsStridedObjects(Model_TObject, [("@padding", numpy.dtype(">i4"))])
    data = numpy.zeros(3, dtype=[("@padding", "i4")])

    out = uproot.interpretation.library._strided_to_awkward(
        awkward, "", interpretation, data
    )
    assert isinstance(out, awkward.contents.RecordArray)
    assert out.fields == []
    assert len(out) == 3


def test_strided_objects_awkward_form_with_inner_shape():
    path = skhep_testdata.data_path("uproot-issue-407.root")
    with uproot.open(path) as file:
        branch = file["tree/branch"]
        flat = branch.interpretation
        interpretation = AsStridedObjects(flat.model, flat.members, flat.original)
        interpretation.reshape((2, 3))

        record = flat.awkward_form(branch.file)
        assert record.parameters == {"__record__": "TDatime"}
        assert interpretation.awkward_form(branch.file) == awkward.forms.RegularForm(
            awkward.forms.RegularForm(record, 3), 2
        )


def test_strided_object_array_needs_numpy():
    with pytest.raises(
        NotImplementedError,
        match=r"^non-numpy temporary arrays require changes in library interpretations$",
    ):
        StridedObjectArray(None, [1, 2, 3])


def test_threading_lock_is_recreated_by_pickling():
    interpretation = AsObjects(AsVector(False, i4))

    copy = pickle.loads(pickle.dumps(interpretation))

    assert copy == interpretation
    assert isinstance(copy._forth_lock, type(threading.Lock()))
