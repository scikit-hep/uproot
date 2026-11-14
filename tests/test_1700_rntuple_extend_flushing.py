# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""RNTuple extension flushing (issue #1688).

Blobs are no longer flushed one at a time, but everything an RNTuple's anchor
points to must still reach storage before the anchor is overwritten in place,
and the old footer must stay intact until then.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import uproot
import uproot.sink.file


class SinkLog(list):
    on_flush = None


def free_segments_classname(path):
    """Class name of the key that the file header's fSeekFree points to."""
    with uproot.open(path) as f:
        seek_free = f.file.fSeekFree
    with open(path, "rb") as raw:
        raw.seek(seek_free)
        data = raw.read(64)
    (version,) = struct.unpack(">h", data[4:6])
    # fNbytes, fVersion, fObjlen, fDatime, fKeylen, fCycle, then fSeekKey and
    # fSeekPdir, which are 64-bit in big keys
    position = 18 + (16 if version > 1000 else 8)
    return data[position + 1 : position + 1 + data[position]]


@pytest.fixture
def sink_log(monkeypatch):
    log = SinkLog()
    original_write = uproot.sink.file.FileSink.write
    original_flush = uproot.sink.file.FileSink.flush

    def write(self, location, serialization):
        log.append(("write", location, location + memoryview(serialization).nbytes))
        return original_write(self, location, serialization)

    def flush(self):
        log.append(("flush",))
        out = original_flush(self)
        if log.on_flush is not None:
            log.on_flush()
        return out

    monkeypatch.setattr(uproot.sink.file.FileSink, "write", write)
    monkeypatch.setattr(uproot.sink.file.FileSink, "flush", flush)
    return log


def checked_commit(path, name, sink_log, operation):
    """
    Runs ``operation`` (which rewrites the RNTuple ``name`` in ``path``) and
    checks that it was crash-safe. Returns the number of flushes it took.
    """
    with uproot.open(path) as f:
        anchor_location = f.key(name).data_cursor.index
        anchor = f[name]
        old_footer = (
            anchor.member("fSeekFooter"),
            anchor.member("fSeekFooter") + anchor.member("fNBytesFooter"),
        )

    def check_free_segments():
        # every flush leaves the file header pointing to the FreeSegments
        # record, not to a blob that was allocated over its old location
        assert free_segments_classname(path) == b"TFile"

    sink_log.clear()
    sink_log.on_flush = check_free_segments
    try:
        operation()
    finally:
        sink_log.on_flush = None

    anchor_write = max(
        i
        for i, event in enumerate(sink_log)
        if event[0] == "write" and event[1] == anchor_location
    )
    # everything the new anchor points to is flushed before it is overwritten
    assert sink_log[anchor_write - 1] == ("flush",)
    # and nothing written before then lands on the footer the old anchor points to
    for event in sink_log[:anchor_write]:
        if event[0] == "write":
            assert event[2] <= old_footer[0] or event[1] >= old_footer[1]
    # and the commit ends flushed
    assert sink_log[-1] == ("flush",)

    return sum(1 for event in sink_log if event[0] == "flush")


@pytest.mark.parametrize("compression", [None, uproot.ZLIB(1)])
def test_extend(tmp_path, sink_log, compression):
    path = str(tmp_path / "file.root")
    widths = {"narrow": 1, "wide": 16}

    with uproot.recreate(path, compression=compression) as f:
        for name, width in widths.items():
            f.mkrntuple(name, {f"c{i}": np.dtype("int64") for i in range(width)})

        for step in range(10):
            flushes = {}
            for name, width in widths.items():
                chunk = {
                    f"c{i}": np.arange(10 * step, 10 * (step + 1), dtype=np.int64)
                    for i in range(width)
                }
                flushes[name] = checked_commit(
                    path, name, sink_log, lambda n=name, c=chunk: f[n].extend(c)
                )
                # readable between extensions, not only after close
                with uproot.open(path) as check:
                    assert check[name].num_entries == 10 * (step + 1)
            # the number of flushes does not grow with the number of columns
            assert flushes["narrow"] == flushes["wide"]

    with uproot.open(path) as f:
        for name, width in widths.items():
            arrays = f[name].arrays()
            for i in range(width):
                assert arrays[f"c{i}"].tolist() == list(range(100))


def test_add_fields_and_extend_uncompressed(tmp_path, sink_log):
    path = str(tmp_path / "file.root")

    with uproot.recreate(path, compression=None) as f:
        f.mkrntuple("nt", {"x": np.dtype("int64")})
        f["nt"].extend({"x": np.arange(10, dtype=np.int64)})

    with uproot.update(path) as f:
        checked_commit(
            path, "nt", sink_log, lambda: f["nt"].add_fields({"y": np.int32})
        )
        checked_commit(
            path,
            "nt",
            sink_log,
            lambda: f["nt"].extend(
                {
                    "x": np.arange(10, 20, dtype=np.int64),
                    "y": np.arange(10, 20, dtype=np.int32),
                }
            ),
        )

    with uproot.open(path) as f:
        arrays = f["nt"].arrays()
        assert arrays["x"].tolist() == list(range(20))
        assert arrays["y"].tolist() == [0] * 10 + list(range(10, 20))
