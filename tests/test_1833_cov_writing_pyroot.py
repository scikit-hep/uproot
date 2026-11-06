# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import struct

import numpy as np
import pytest

import uproot
from uproot.pyroot import _GetStreamersOnce


def test_converted_object_has_a_detached_file():
    ROOT = pytest.importorskip("ROOT")

    hist = ROOT.TH1D("test_1833_detached", "the title", 3, 0.0, 3.0)
    hist.Fill(1.5)

    model = uproot.from_pyroot(hist)
    assert model.values().tolist() == [0, 1, 0]

    # looking at the streamers of the object collects them for all detached files
    assert (
        _GetStreamersOnce(hist).streamers["TH1D"]
        is model.file.streamers["TH1D"]
        is _GetStreamersOnce._streamers["TH1D"]
    )
    assert hist.IsA().GetClassVersion() in model.file.streamers["TH1D"]
    assert model.file.custom_classes is _GetStreamersOnce._custom_classes


def test_deserialization_errors_do_not_name_a_file():
    ROOT = pytest.importorskip("ROOT")

    stand_in = _GetStreamersOnce(ROOT.TObject())
    assert stand_in.file_path is None

    # a reference to a class that hasn't been seen before
    data = np.frombuffer(struct.pack(">II", 0x40000004, 0x80000005), np.uint8)
    chunk = uproot.source.chunk.Chunk.wrap(None, data)
    with pytest.raises(uproot.deserialization.DeserializationError) as err:
        uproot.deserialization.read_object_any(
            chunk, uproot.source.cursor.Cursor(0), {}, stand_in, None, None
        )
    assert "invalid class-tag reference: 5" in str(err.value)
    assert "in file" not in err.value.message
