# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import types

import numpy as np

import uproot


def test_empty_map():
    # a std::map<int, double> with no entries, not memberwise: byte count,
    # version, and a size of 0
    data = np.frombuffer(b"\x40\x00\x00\x06\x00\x09\x00\x00\x00\x00", np.uint8)
    interpretation = uproot.containers.AsMap(True, np.dtype(">i4"), np.dtype(">f8"))
    out = interpretation.read(
        uproot.source.chunk.Chunk.wrap(None, data),
        uproot.source.cursor.Cursor(0),
        {},
        types.SimpleNamespace(file_path="memory"),
        None,
        None,
    )
    assert len(out) == 0
    assert out.keys().dtype == np.dtype(">i4")
    assert out.values().dtype == np.dtype(">f8")
