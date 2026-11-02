# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import lzma
import zlib

import cramjam
import numpy
import pytest
import skhep_testdata
import xxhash

import uproot
import uproot.compression
from uproot.compression import LZ4, LZMA, ZLIB, ZSTD, Compression

DATA = b"uproot compression test " * 200


class Unregistered(Compression):
    """A Compression type that is not in ``algorithm_codes``."""

    name = "UNREGISTERED"

    def __init__(self, level):
        self._level = level


def source():
    out = uproot.source.chunk.Source()
    out._file_path = "test.root"
    return out


def decompress(block, uncompressed_bytes):
    chunk = uproot.source.chunk.Chunk.wrap(
        source(), numpy.frombuffer(block, dtype=numpy.uint8)
    )
    cursor = uproot.source.cursor.Cursor(0)
    out = uproot.compression.decompress(
        chunk, cursor, {}, len(block), uncompressed_bytes
    )
    return out.raw_data.tobytes()


def header(algo, method, compressed_bytes, uncompressed_bytes):
    return (
        algo
        + method
        + compressed_bytes.to_bytes(3, "little")
        + uncompressed_bytes.to_bytes(3, "little")
    )


def test_from_code():
    assert uproot.compression.Compression.from_code(101) == ZLIB(1)
    assert uproot.compression.Compression.from_code(207) == LZMA(7)
    assert uproot.compression.Compression.from_code(404) == LZ4(4)
    assert uproot.compression.Compression.from_code(505) == ZSTD(5)
    assert uproot.compression.Compression.from_code(100) is None
    assert uproot.compression.Compression.from_code(5) is None
    assert ZSTD(5).code == 505
    assert ZSTD(5).code_pair == (5, 5)


def test_from_code_unrecognized():
    with pytest.raises(ValueError, match="unrecognized compression algorithm code: 9"):
        uproot.compression.Compression.from_code(905)


def test_unregistered_compression_type():
    compression = Unregistered(3)
    assert compression.level == 3
    assert repr(compression) == "Unregistered(3)"
    with pytest.raises(ValueError, match="unrecognized compression type: .*Unreg"):
        compression.code_pair  # noqa: B018 (the property raises)


def test_equality():
    assert ZLIB(1) == ZLIB(1)
    assert ZLIB(1) != ZLIB(2)
    assert ZLIB(1) != LZMA(1)
    assert ZLIB(1) != "ZLIB"


@pytest.mark.parametrize("cls, maximum", [(ZLIB, 9), (LZMA, 9), (LZ4, 12), (ZSTD, 22)])
def test_level_validation(cls, maximum):
    assert cls(maximum).level == maximum
    assert cls(numpy.int64(1)).level == 1
    assert type(cls(numpy.int64(1)).level) is int

    with pytest.raises(TypeError, match="Compression level must be an integer"):
        cls(1.5)
    with pytest.raises(ValueError, match=f"must be between 0 and {maximum} "):
        cls(maximum + 1)
    with pytest.raises(ValueError, match=f"must be between 0 and {maximum} "):
        cls(-1)


@pytest.mark.parametrize(
    "decompressor, message",
    [
        (uproot.compression._decompress_ZLIB, "zlib decompression requires"),
        (uproot.compression._decompress_LZ4, "lz4 block decompression requires"),
        (uproot.compression._decompress_ZSTD, "zstd block decompression requires"),
        (uproot.compression._decompress_LZMA, "lzma decompression requires"),
    ],
)
def test_decompress_requires_uncompressed_bytes(decompressor, message):
    with pytest.raises(ValueError, match=message):
        decompressor.decompress(b"")


@pytest.mark.parametrize("library, module", [("isal", "isal"), ("deflate", "deflate")])
def test_zlib_alternative_libraries(library, module, monkeypatch):
    pytest.importorskip(module)
    monkeypatch.setattr(uproot.compression._DecompressZLIB, "library", library)

    # what the library compresses, the standard library decompresses
    assert zlib.decompress(ZLIB(6).compress(DATA)) == DATA

    # and the other way around
    assert ZLIB(6).decompress(zlib.compress(DATA), len(DATA)) == DATA

    with pytest.raises(ValueError, match=f'ZLIB.library="{library}"'):
        ZLIB(0).compress(DATA)


@pytest.mark.parametrize("library, module", [("isal", "isal"), ("deflate", "deflate")])
def test_zlib_alternative_libraries_read_root_file(library, module, monkeypatch):
    pytest.importorskip(module)
    path = skhep_testdata.data_path("uproot-Zmumu-zlib.root")
    expected = uproot.open(path)["events/px1"].array(library="np")
    assert len(expected) == 2304
    assert expected[:3].tolist() == pytest.approx(
        [-41.1952876442, 35.1180497674, 35.1180497674]
    )

    monkeypatch.setattr(uproot.compression._DecompressZLIB, "library", library)
    actual = uproot.open(path)["events/px1"].array(library="np")
    assert numpy.array_equal(actual, expected)


def test_zlib_unrecognized_library(monkeypatch):
    monkeypatch.setattr(uproot.compression._DecompressZLIB, "library", "nonsense")
    with pytest.raises(ValueError, match="unrecognized ZLIB.library: 'nonsense'"):
        ZLIB(1).compress(DATA)
    with pytest.raises(ValueError, match="unrecognized ZLIB.library: 'nonsense'"):
        ZLIB(1).decompress(zlib.compress(DATA), len(DATA))


def test_lzma_without_cramjam_support(monkeypatch):
    # older cramjam has neither cramjam.xz nor cramjam.experimental.lzma
    monkeypatch.delattr(cramjam, "xz", raising=False)
    monkeypatch.delattr(cramjam, "experimental", raising=False)

    compressed = LZMA(3).compress(DATA)
    assert lzma.decompress(compressed) == DATA

    # the standard library doesn't need to be told the size
    assert LZMA(3).decompress(lzma.compress(DATA)) == DATA


@pytest.mark.parametrize("compression", [ZLIB(4), LZMA(4), LZ4(4), ZSTD(4)])
def test_round_trip(compression):
    block = uproot.compression.compress(DATA, compression)
    assert len(block) < len(DATA)
    assert block[:2] == compression._2byte
    assert decompress(block, len(DATA)) == DATA


@pytest.mark.parametrize(
    "filename",
    [
        "uproot-Zmumu-zlib.root",
        "uproot-Zmumu-lzma.root",
        "uproot-Zmumu-lz4.root",
        "uproot-Zmumu-zstd.root",
    ],
)
def test_read_files_written_by_root(filename):
    with uproot.open(skhep_testdata.data_path(filename)) as file:
        px1 = file["events/px1"].array(library="np")
    assert px1[:3].tolist() == pytest.approx(
        [-41.1952876442, 35.1180497674, 35.1180497674]
    )


def test_read_several_blocks():
    half = len(DATA) // 2
    first = uproot.compression.compress(DATA[:half], ZLIB(1))
    second = uproot.compression.compress(DATA[half:], ZSTD(1))
    assert decompress(first + second, len(DATA)) == DATA


def test_lz4_wrong_checksum():
    block = bytearray(uproot.compression.compress(DATA, LZ4(1)))
    block[9] ^= 0xFF
    expected = int.from_bytes(block[9:17], "big")
    computed = xxhash.xxh64(bytes(block[17:])).intdigest()
    with pytest.raises(
        ValueError,
        match=f"computed checksum {computed} didn't match expected checksum "
        f"{expected}\nin file test.root",
    ):
        decompress(bytes(block), len(DATA))


def test_old_compression_algorithm_is_unsupported():
    with pytest.raises(
        ValueError, match="unsupported compression algorithm: b'CS'.*20 years"
    ):
        decompress(header(b"CS", b"\x08", 1, 10) + b"\x00", 10)


def test_unrecognized_compression_algorithm():
    with pytest.raises(
        ValueError, match="unrecognized compression algorithm: b'XX'\nin file test.root"
    ):
        decompress(header(b"XX", b"\x08", 1, 10) + b"\x00", 10)


def test_block_is_shorter_than_its_header_says():
    compressed = zlib.compress(DATA)
    block = header(b"ZL", b"\x08", len(compressed), len(DATA) + 1) + compressed
    with pytest.raises(
        ValueError,
        match=rf"decompressed to {len(DATA)} bytes, but the block header expects "
        rf"{len(DATA) + 1} bytes.\nin file test.root",
    ):
        decompress(block, len(DATA) + 1)
