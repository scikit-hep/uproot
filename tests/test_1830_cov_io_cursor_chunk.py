# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import datetime
import io
import queue
import struct

import numpy
import pytest

import uproot
from uproot.deserialization import DeserializationError
from uproot.source.chunk import Chunk, Resource, Source
from uproot.source.cursor import Cursor


def chunk_of(data, start=0):
    source = Source()
    source._file_path = "test.root"
    return Chunk.wrap(source, numpy.frombuffer(data, dtype=numpy.uint8), start)


def test_resource_has_no_file_path_by_default():
    assert Resource().file_path is None


def test_source_counters():
    with uproot.source.object.ObjectSource(
        io.BytesIO(bytes(range(100))), use_threads=False
    ) as source:
        assert source.num_requests == 0

        assert source.chunk(10, 20).raw_data.tolist() == list(range(10, 20))
        notifications = queue.Queue()
        chunks = source.chunks([(0, 5), (50, 53)], notifications)
        assert [x.raw_data.tolist() for x in chunks] == [
            list(range(5)),
            [50, 51, 52],
        ]

        assert source.num_requests == 2
        assert source.num_requested_chunks == 3
        assert source.num_requested_bytes == 18
        assert notifications.qsize() == 2


def test_source_without_executor_is_closed():
    assert Source().closed is True


def test_source_closed_follows_its_executor():
    source = Source()
    source._executor = uproot.source.futures.TrivialExecutor()
    assert source.closed is False


@pytest.mark.parametrize(
    "path, shown", [("short", "'short'"), ("a/long/file/path.root", "'.../path.root'")]
)
def test_multithreaded_source_repr(path, shown):
    source = uproot.source.chunk.MultithreadedSource()
    source._file_path = path
    source._executor = uproot.source.futures.ThreadPoolExecutor(2)
    try:
        assert repr(source).startswith(
            f"<MultithreadedSource {shown} (2 workers) at 0x"
        )
        assert source.num_workers == 2
        assert source.executor is source._executor
    finally:
        source._executor.shutdown()


def test_chunk_properties_and_repr():
    chunk = chunk_of(b"abcd", start=10)
    assert repr(chunk) == "<Chunk 10-14>"
    assert chunk.start == 10
    assert chunk.stop == 14
    assert chunk.is_memmap is False
    assert isinstance(chunk.future, uproot.source.futures.TrivialFuture)


def test_chunk_contains_integers_and_cursors():
    chunk = chunk_of(b"abcd", start=10)
    assert (10, 14) in chunk
    assert (9, 14) not in chunk
    assert (Cursor(11), Cursor(13)) in chunk
    assert (Cursor(11), Cursor(15)) not in chunk


def test_chunk_wait_insist_must_be_bool_or_int():
    with pytest.raises(TypeError, match="insist must be a bool or an int, not 'yes'"):
        chunk_of(b"abcd").wait(insist="yes")


def test_chunk_wait_insist_int():
    chunk = chunk_of(b"abcd", start=10)
    chunk.wait(insist=14)
    with pytest.raises(OSError, match="expected Chunk of length 4"):
        chunk.wait(insist=15)


def test_chunk_wait_insist_false():
    chunk = chunk_of(b"abcd")
    chunk._stop = 100
    chunk.wait(insist=False)
    with pytest.raises(OSError, match="expected Chunk of length 100"):
        chunk.wait()


def test_detach_memmap(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcd")
    options = {"num_fallback_workers": 1, "use_threads": False}
    with uproot.source.file.MemmapSource(str(path), **options) as source:
        unread = source.chunk(0, 4)
        read = source.chunk(1, 3)
        read.wait()
        assert unread.is_memmap is True
        detached = [unread.detach_memmap(), read.detach_memmap()]

    assert [x.is_memmap for x in detached] == [False, False]
    assert detached[0].raw_data.tobytes() == b"abcd"
    assert detached[1].raw_data.tobytes() == b"bc"
    assert detached[0].detach_memmap() is detached[0]


def test_performance_counters():
    with uproot.source.object.ObjectSource(
        io.BytesIO(bytes(range(100))), use_threads=False
    ) as source:
        source.chunk(10, 20).wait()
        counters = source.performance_counters
    assert counters.asdict() == {
        "num_requested_bytes": 10,
        "num_requests": 1,
        "num_requested_chunks": 1,
    }


def test_chunk_get_outside_of_range():
    chunk = chunk_of(b"abcd", start=10)
    with pytest.raises(
        DeserializationError,
        match="attempting to get bytes 8:12\noutside expected range 10:14",
    ):
        chunk.get(8, 12, Cursor(8), {})


def test_chunk_remainder():
    chunk = chunk_of(b"abcd", start=10)
    assert chunk.remainder(12, Cursor(12), {}).tobytes() == b"cd"
    with pytest.raises(
        DeserializationError,
        match="attempting to get bytes after 8\noutside expected range 10:14",
    ):
        chunk.remainder(8, Cursor(8), {})


@pytest.mark.parametrize(
    "cursor, expected",
    [
        (Cursor(5), "Cursor(5)"),
        (Cursor(5, origin=-3), "Cursor(5, origin=-3)"),
        (Cursor(5, refs={}), "Cursor(5)"),
        (Cursor(5, refs={1: "a", 2: "b"}), "Cursor(5, 2 refs: 1, 2)"),
        (Cursor(5, refs={1: 0, 2: 0, 3: 0, 4: 0}), "Cursor(5, 4 refs: 1, 2, 3...)"),
    ],
)
def test_cursor_repr(cursor, expected):
    assert repr(cursor) == expected


def test_cursor_copy_links_or_copies_refs():
    cursor = Cursor(5, origin=1, refs={1: "a"})

    linked = cursor.copy()
    assert linked.refs is cursor.refs

    copied = cursor.copy(link_refs=False)
    assert copied.refs == cursor.refs
    assert copied.refs is not cursor.refs
    assert copied.index == 5
    assert copied.origin == 1


def test_cursor_skip_after():
    cursor = Cursor(0)
    obj = type("Obj", (), {"cursor": Cursor(10), "num_bytes": 5})()
    cursor.skip_after(obj)
    assert cursor.index == 15


@pytest.mark.parametrize(
    "members",
    [
        {},
        {"cursor": "not a cursor", "num_bytes": 5},
        {"cursor": Cursor(10), "num_bytes": None},
    ],
)
def test_cursor_skip_after_needs_cursor_and_num_bytes(members):
    with pytest.raises(TypeError, match="can only be used on an object with a"):
        Cursor(0).skip_after(type("Obj", (), members)())


def test_cursor_skip_over_object_without_num_bytes():
    # the top bit of a 4-byte number marks a num_bytes; this one doesn't have it
    cursor = Cursor(0)
    assert cursor.skip_over(chunk_of(b"\x00\x00\x00\x0a\x00\x01"), {}) is False
    assert cursor.index == 0


def test_cursor_skip_over_object_with_num_bytes():
    cursor = Cursor(0)
    data = struct.pack(">IH", 0x40000000 | 6, 1) + b"\x00" * 10
    assert cursor.skip_over(chunk_of(data), {}) is True
    assert cursor.index == 10


@pytest.mark.parametrize(
    "exponent, mantissa, num_bits, expected",
    [
        (127, 0, 12, 1.0),
        (128, 0, 12, 2.0),
        (127, 1 << 11, 12, 1.5),  # top bit of the 12-bit mantissa
        (127, 1 << 13, 12, -1.0),  # the sign is one bit above the mantissa
        (128, (1 << 13) | (1 << 11), 12, -3.0),
    ],
)
def test_cursor_float16(exponent, mantissa, num_bits, expected):
    chunk = chunk_of(struct.pack(">BH", exponent, mantissa))

    cursor = Cursor(0)
    assert cursor.float16(chunk, num_bits, {}, move=False) == expected
    assert cursor.index == 0

    assert cursor.float16(chunk, num_bits, {}) == expected
    assert cursor.index == 3


def test_cursor_double32():
    chunk = chunk_of(struct.pack(">f", 1.5))
    cursor = Cursor(0)
    assert cursor.double32(chunk, {}, move=False) == 1.5
    assert cursor.index == 0
    assert cursor.double32(chunk, {}) == 1.5
    assert cursor.index == 4


def test_cursor_byte_bytes_without_moving():
    chunk = chunk_of(b"abcd")
    cursor = Cursor(1)
    assert cursor.byte(chunk, {}, move=False).tobytes() == b"b"
    assert cursor.bytes(chunk, 2, {}, move=False).tobytes() == b"bc"
    assert cursor.index == 1


def test_cursor_strings_without_moving():
    chunk = chunk_of(b"\x03abc\x00")
    cursor = Cursor(0)
    assert cursor.bytestring(chunk, {}, move=False) == b"abc"
    assert cursor.string(chunk, {}, move=False) == "abc"
    assert cursor.index == 0

    cursor = Cursor(1)
    assert cursor.bytestring_with_length(chunk, {}, 3, move=False) == b"abc"
    assert cursor.string_with_length(chunk, {}, 3, move=False) == "abc"
    assert cursor.classname(chunk, {}, move=False) == "abc"
    assert cursor.index == 1


def test_cursor_array_without_moving():
    chunk = chunk_of(struct.pack(">2i", 1, 2))
    cursor = Cursor(0)
    out = cursor.array(chunk, 2, numpy.dtype(">i4"), {}, move=False)
    assert out.tolist() == [1, 2]
    assert cursor.index == 0


def test_cursor_long_string():
    text = b"x" * 300
    chunk = chunk_of(b"\xff" + struct.pack(">i", len(text)) + text)
    cursor = Cursor(0)
    assert cursor.bytestring(chunk, {}) == text
    assert cursor.index == 305


def test_cursor_classname_needs_a_terminator():
    chunk = chunk_of(b"abc", start=4)
    with pytest.raises(
        OSError, match="no terminator .null byte. in Chunk 4:7\nof file path test.root"
    ):
        Cursor(4).classname(chunk, {})


def test_cursor_rntuple_string():
    chunk = chunk_of(struct.pack("<I", 3) + b"abc")

    cursor = Cursor(0)
    assert cursor.rntuple_string(chunk, {}, move=False) == "abc"
    assert cursor.index == 0

    assert cursor.rntuple_string(chunk, {}) == "abc"
    assert cursor.index == 7


def test_cursor_rntuple_datetime():
    timestamp = 1700000000
    chunk = chunk_of(struct.pack("<Q", timestamp))
    cursor = Cursor(0)
    expected = datetime.datetime.fromtimestamp(timestamp)
    assert cursor.rntuple_datetime(chunk, {}, move=False) == expected
    assert cursor.index == 0
    assert cursor.rntuple_datetime(chunk, {}) == expected
    assert cursor.index == 8


def debug_output(data, **kwargs):
    stream = io.StringIO()
    Cursor(0).debug(chunk_of(data), stream=stream, **kwargs)
    return stream.getvalue()


def test_cursor_debug_bytes():
    assert debug_output(b"ab\x00\xff", limit_bytes=3) == (
        "--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-\n"
        " 97  98   0\n"
        "  a   b ---\n"
    )


def test_cursor_debug_numbers_given_a_dtype_object():
    assert debug_output(struct.pack(">f", 1.5), dtype=numpy.dtype(">f4")) == (
        "--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-\n"
        " 63 192   0   0\n"
        "  ? --- --- ---\n"
        "            1.5\n"
    )


def test_cursor_debug_numbers_given_a_dtype_name_and_offset():
    data = b"\x7b\x7b\x7b" + struct.pack(">f", 1.5) + b"\x7b"
    assert debug_output(data, dtype=">f4", offset=3) == (
        "--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-\n"
        "123 123 123  63 192   0   0 123\n"
        "  {   {   {   ? --- --- ---   {\n"
        "                        1.5\n"
    )


def test_cursor_debug_line_with_nothing_to_interpret():
    # 5 numbers fill the first line, and the leftover byte on the second can't be one
    data = struct.pack(">5i", 1, 2, 3, 4, 5) + b"\x7b"
    lines = debug_output(data, dtype=">i4").split("\n")
    assert lines[3].split() == ["1", "2", "3", "4", "5"]
    assert lines[4:] == [
        "        " + "--+-" * 20,
        "        123",
        "          {",
        "",
        "",
    ]
