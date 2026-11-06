# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import asyncio
import pathlib
import pickle
import queue
import zipfile

import fsspec
import pytest

import uproot
import uproot.source.fsspec
from uproot.source.coalesce import CoalesceConfig

DATA = bytes(range(256))
FSSpecSource = uproot.source.fsspec.FSSpecSource


@pytest.fixture
def path(tmp_path):
    out = tmp_path / "data.bin"
    out.write_bytes(DATA)
    return str(out)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("/local/file.root", "/local/file.root"),
        ("root://server//file.root", "root://server//file.root"),
        ("s3://bucket/file.root", "s3://bucket/file.root"),
        ("https://server/file.root", "https://server/file.root"),
        (
            "https://server/remote.php/dav/public-files/abc/file.root",
            "simplecache::https://server/remote.php/dav/public-files/abc/file.root",
        ),
    ],
)
def test_maybe_wrap_remote_url(url, expected):
    assert uproot.source.fsspec._maybe_wrap_remote_url(url) == expected


def test_properties(path):
    with FSSpecSource(path) as source:
        assert source.num_bytes == len(DATA)
        assert source.async_impl is False
        assert source.closed is False
    assert source.closed is True


def test_repr_of_short_path():
    memory = fsspec.filesystem("memory")
    memory.pipe("/a", DATA)
    try:
        with FSSpecSource("memory://a") as source:
            assert repr(source) == f"<FSSpecSource '/a' at 0x{id(source):012x}>"
    finally:
        memory.rm("/a")


def test_repr_of_long_path(path):
    with FSSpecSource(path) as source:
        assert repr(source) == (
            # fsspec keeps local paths with forward slashes, also on Windows
            f"<FSSpecSource '...{pathlib.Path(path).as_posix()[-10:]}' "
            f"at 0x{id(source):012x}>"
        )


def test_pickled_source_still_reads(path):
    with FSSpecSource(path) as source:
        with pickle.loads(pickle.dumps(source)) as clone:
            assert clone.file_path == source.file_path
            assert clone.chunk(1, 4).raw_data.tobytes() == DATA[1:4]


def test_coalesce_config_of_one_call_overrides_the_sources(path, monkeypatch):
    separate = CoalesceConfig(max_range_gap=0, min_first_request_bytes=0)
    together = CoalesceConfig(max_range_gap=100)
    ranges = [(0, 4), (10, 14)]
    expected = [DATA[0:4], DATA[10:14]]
    requests = []

    with FSSpecSource(path, coalesce_config=separate) as source:
        original = source._fs.cat_ranges

        def cat_ranges(paths, starts, ends):
            requests.append(list(zip(starts, ends, strict=True)))
            return original(paths, starts, ends)

        monkeypatch.setattr(source._fs, "cat_ranges", cat_ranges)

        chunks = source.chunks(ranges, queue.Queue())
        assert [x.raw_data.tobytes() for x in chunks] == expected
        assert sorted(requests) == [[(0, 4)], [(10, 14)]]

        requests.clear()
        chunks = source.chunks(ranges, queue.Queue(), coalesce_config=together)
        assert [x.raw_data.tobytes() for x in chunks] == expected
        assert requests == [[(0, 14)]]


def test_file_inside_a_zip_archive_is_closed_with_its_filesystem(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("data.bin", DATA)

    with FSSpecSource(f"zip://data.bin::{archive}") as source:
        assert source.chunk(1, 4).raw_data.tobytes() == DATA[1:4]
        underlying = source._fs.fo

    assert source.closed is True
    assert underlying.closed is True


def test_closing_ignores_errors_of_the_workarounds(path, monkeypatch):
    calls = []

    class Closing:
        def close(self):
            calls.append("close")
            raise ValueError("this filesystem can't close")

        def __exit__(self, *args):
            calls.append("__exit__")
            raise ValueError("this file can't exit")

    source = FSSpecSource(path)
    monkeypatch.setattr(source._fs, "close", Closing().close, raising=False)
    monkeypatch.setattr(source._fs, "of", Closing(), raising=False)
    source.close()
    assert source.closed is True
    # both workarounds were tried, and their errors didn't stop the closing
    assert sorted(calls) == ["__exit__", "close"]


def test_closed_source_refuses_to_read(path):
    source = FSSpecSource(path)
    source.close()
    with pytest.raises(OSError, match="file .*data.bin' is closed"):
        source.chunk(0, 4)
    with pytest.raises(OSError, match="file .*data.bin' is closed"):
        source.chunks([(0, 4)], queue.Queue())


def test_chunks_in_wasm_run_synchronously(path, monkeypatch):
    def no_threads(*args, **kwargs):
        raise AssertionError("threads can't be spawned in pyodide")

    monkeypatch.setattr(uproot._util, "wasm", True)
    monkeypatch.setattr(asyncio, "to_thread", no_threads)

    notifications = queue.Queue()
    with FSSpecSource(path) as source:
        chunks = source.chunks([(0, 4), (200, 204)], notifications)
        assert notifications.qsize() == 2
        assert [x.raw_data.tobytes() for x in chunks] == [DATA[:4], DATA[200:204]]


def test_cat_ranges_that_is_not_callable(path, monkeypatch):
    with FSSpecSource(path) as source:
        monkeypatch.setattr(source._fs, "cat_ranges", None)
        chunks = source.chunks([(0, 4)], queue.Queue())
        with pytest.raises(TypeError, match="blocking_func must be callable"):
            chunks[0].wait()


def test_loop_executor_only_takes_coroutines():
    executor = uproot.source.fsspec.FSSpecLoopExecutor()

    async def coroutine():
        return 123

    assert executor.submit(coroutine()).result(timeout=60) == 123

    with pytest.raises(TypeError, match="loop executor can only submit coroutines"):
        executor.submit(lambda: 123)
