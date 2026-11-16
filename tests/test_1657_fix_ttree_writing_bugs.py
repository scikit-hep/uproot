# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.compression

awkward = pytest.importorskip("awkward")


def test_sliced_string_branch_roundtrip(tmp_path):
    # A sliced string array has offsets[0] != 0; the string-branch path of
    # Tree.extend must normalize this or the basket is unreadable.
    newfile = os.path.join(tmp_path, "sliced_string.root")

    big = "x" * 300  # also exercise the >= 255 length-extension path
    cases = [
        awkward.Array(["aaa", "bb", "cccc", "dd", "eeeee"])[2:],
        awkward.Array(["aaa", big, "bb", "cccc", "dd"])[2:],
        awkward.Array([big, "aaa", "bb", "cccc"])[1:],
        awkward.Array(["aaa", "bb", "cccc", "dd", "eeeee"]),  # unsliced still ok
    ]

    for i, arr in enumerate(cases):
        fn = os.path.join(tmp_path, f"sliced_string_{i}.root")
        with uproot.recreate(fn) as f:
            tree = f.mktree("t", {"s": "string"})
            tree.extend({"s": arr})
        with uproot.open(fn) as f:
            assert f["t/s"].array().tolist() == arr.tolist()


def test_string_branch_flen_per_branch(tmp_path):
    # TLeafC fLen must be tracked per-branch and accumulated as a max across
    # extends, not shared at the tree level or overwritten each extend.
    newfile = os.path.join(tmp_path, "flen.root")

    a1 = awkward.Array(["x" * 24, "yy", "z"])  # max length 24
    b1 = awkward.Array(["aaa", "b", "cc"])  # max length 3
    a2 = awkward.Array(["p", "q", "r"])
    b2 = awkward.Array(["x" * 40, "y", "z"])  # bigger on second extend

    with uproot.recreate(newfile) as f:
        tree = f.mktree("t", {"s1": "string", "s2": "string"})
        tree.extend({"s1": a1, "s2": b1})
        tree.extend({"s1": a2, "s2": b2})

    with uproot.open(newfile) as f:
        # fLen includes the 1-byte length prefix
        assert f["t/s1"].member("fLeaves")[0].member("fLen") == 25
        assert f["t/s2"].member("fLeaves")[0].member("fLen") == 41
        assert f["t/s1"].array().tolist() == a1.tolist() + a2.tolist()
        assert f["t/s2"].array().tolist() == b1.tolist() + b2.tolist()


def test_recarray_to_dict_nested_struct(tmp_path):
    from uproot.writing._cascadetree import recarray_to_dict

    dt = np.dtype([("a", np.int32), ("nested", [("x", np.float64), ("y", np.int16)])])
    arr = np.zeros(5, dtype=dt)
    arr["a"] = np.arange(5)
    arr["nested"]["x"] = np.arange(5) * 1.5
    arr["nested"]["y"] = np.arange(5)

    out = recarray_to_dict(arr)
    assert set(out) == {"a", "nested.x", "nested.y"}
    assert out["a"].tolist() == [0, 1, 2, 3, 4]
    assert out["nested.x"].tolist() == [0.0, 1.5, 3.0, 4.5, 6.0]
    assert out["nested.y"].tolist() == [0, 1, 2, 3, 4]


def test_tgraph_default_y_range_uses_y():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([100.0, 200.0, 150.0, 175.0])
    g = uproot.as_TGraph({"x": x, "y": y})

    span = 200.0 - 100.0
    assert g.member("fMinimum") == pytest.approx(100.0 - 0.1 * span)
    assert g.member("fMaximum") == pytest.approx(200.0 + 0.1 * span)


def test_histogram_regular_edge_detection(tmp_path):
    # The numpy.linspace regularity check must still collapse regular edges to
    # an empty fXbins and keep irregular edges.
    newfile = os.path.join(tmp_path, "hist.root")

    entries = np.ones(10)
    regular = np.linspace(0, 10, 11)
    irregular = np.array([0, 1, 2, 4, 8, 16, 17, 18, 19, 20, 21], dtype=float)

    with uproot.recreate(newfile) as f:
        f["hreg"] = (entries, regular)
        f["hirr"] = (entries, irregular)

    with uproot.open(newfile) as f:
        assert len(f["hreg"].member("fXaxis").member("fXbins")) == 0
        assert len(f["hirr"].member("fXaxis").member("fXbins")) == len(irregular)


def test_compress_incompressible_block_no_corruption():
    # An incompressible block can compress to more than 2**24-1 bytes; the
    # 3-byte header field must not be silently truncated. Mixing a compressible
    # block with an incompressible one used to produce a buffer that was
    # smaller than the input (so it was kept) but had a corrupt block header.
    # The fix falls back to storing the data uncompressed in that case.
    from uproot.compression import (
        ZLIB,
        LZ4,
        ZSTD,
        compress,
        decompress,
        _3BYTE_MAX,
    )
    from uproot.source.chunk import Chunk
    from uproot.source.cursor import Cursor

    rng = np.random.default_rng(12345)
    block1 = np.zeros(_3BYTE_MAX, dtype=np.uint8).tobytes()
    block2 = rng.integers(0, 256, size=_3BYTE_MAX, dtype=np.uint8).tobytes()
    data = block1 + block2

    for comp in [ZLIB(1), LZ4(1), ZSTD(1)]:
        out = compress(data, comp)
        # The fix returns the input unchanged (stored uncompressed) instead of
        # emitting a buffer with a truncated 3-byte block header.
        assert out is data

    # Any compressed buffer that compress() *does* return must read back.
    compressible = (np.arange(_3BYTE_MAX + 1000, dtype=np.uint8) % 4).tobytes()
    for comp in [ZLIB(1), LZ4(1), ZSTD(1)]:
        out = compress(compressible, comp)
        chunk = Chunk.wrap(None, np.frombuffer(out, dtype=np.uint8))
        result = decompress(chunk, Cursor(0), {}, len(out), len(compressible))
        assert np.asarray(result.raw_data).tobytes() == compressible


def test_write_incompressible_branch_roundtrip(tmp_path):
    # End-to-end: writing ~17 MB of incompressible random bytes (more than one
    # compression block) into a compressed TTree branch must read back exactly.
    newfile = os.path.join(tmp_path, "incompressible.root")

    rng = np.random.default_rng(20240610)
    values = rng.integers(
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max,
        size=17_000_000 // 8,
        dtype=np.int64,
    )

    with uproot.recreate(newfile, compression=uproot.ZLIB(1)) as f:
        f.mktree("t", {"x": np.int64})
        f["t"].extend({"x": values})

    with uproot.open(newfile) as f:
        np.testing.assert_array_equal(f["t/x"].array(library="np"), values)


def test_compress_still_compresses_compressible_data():
    from uproot.compression import ZLIB, LZ4, ZSTD, compress, decompress
    from uproot.source.chunk import Chunk
    from uproot.source.cursor import Cursor

    data = (np.arange(1_000_000, dtype=np.int64) % 7).tobytes()

    for comp in [ZLIB(1), LZ4(1), ZSTD(1)]:
        out = compress(data, comp)
        assert len(out) < len(data)
        chunk = Chunk.wrap(None, np.frombuffer(out, dtype=np.uint8))
        result = decompress(chunk, Cursor(0), {}, len(out), len(data))
        assert np.asarray(result.raw_data).tobytes() == data
