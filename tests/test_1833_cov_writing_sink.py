# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle

import pytest

import uproot.sink.file


def test_file_object_must_be_writable(tmp_path):
    path = tmp_path / "readonly.bin"
    path.write_bytes(b"data")
    with open(path, "rb") as file:
        with pytest.raises(TypeError, match="supports reading and writing"):
            uproot.sink.file.FileSink(file)
    assert path.read_bytes() == b"data"


def test_pickle_reopens_the_same_path(tmp_path):
    path = tmp_path / "pickled.bin"
    sink = uproot.sink.file.FileSink(str(path))
    sink.write(0, b"hello")
    sink.flush()

    clone = pickle.loads(pickle.dumps(sink))
    sink.close()

    # the open file object does not travel with the pickle
    assert clone._file is None
    assert clone.file_path == sink.file_path
    assert not clone.closed
    assert clone.read(0, 5) == b"hello"
    clone.write(5, b" world")
    clone.close()
    assert path.read_bytes() == b"hello world"


@pytest.mark.parametrize(
    "insist, message",
    [
        (True, r"could not read 10 bytes from the file at position 1\n\nin path: "),
        (8, r"could not read 8 bytes from the file at position 1\n\nin path: "),
    ],
)
def test_read_insists_on_the_length(tmp_path, insist, message):
    with uproot.sink.file.FileSink(str(tmp_path / "short.bin")) as sink:
        sink.write(0, b"abcde")
        sink.flush()
        with pytest.raises(OSError, match=message):
            sink.read(1, 10, insist=insist)


@pytest.mark.parametrize("insist", [False, 4])
def test_read_accepts_a_short_result(tmp_path, insist):
    with uproot.sink.file.FileSink(str(tmp_path / "short.bin")) as sink:
        sink.write(0, b"abcde")
        sink.flush()
        assert sink.read(1, 10, insist=insist) == b"bcde"
