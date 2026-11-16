# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import warnings

import pytest
import skhep_testdata

import uproot
import uproot._graphed
import uproot._util
import uproot.source.file
import uproot.source.fsspec

ZMUMU = "uproot-Zmumu.root"
MEMMAP = uproot.source.file.MemmapSource


class MyMemmap(uproot.source.file.MemmapSource):
    pass


class MyFSSpec(uproot.source.fsspec.FSSpecSource):
    pass


def _open_file_like(path, **options):
    with open(path, "rb") as f:
        uproot.open(f, **options)


def _dask(path, **options):
    pytest.importorskip("dask_awkward")
    uproot.dask(path + ":events", **options)


@pytest.mark.parametrize(
    "call",
    [
        lambda p, **o: uproot.open(p, handler=MEMMAP, **o),
        lambda p, **o: uproot.open(p, handler=MyMemmap, **o),
        _open_file_like,
        lambda p, **o: next(uproot.iterate(p + ":events", handler=MEMMAP, **o)),
        lambda p, **o: uproot.concatenate(p + ":events", handler=MEMMAP, **o),
        lambda p, **o: _dask(p, handler=MEMMAP, **o),
        lambda p, **o: _dask(p, handler=MEMMAP, open_files=False, **o),
        lambda p, **o: uproot.graphed(p + ":events", handler=MEMMAP, **o),
        lambda p, **o: uproot._graphed.graphed_partitions(
            p + ":events", handler=MEMMAP, **o
        ),
    ],
    ids=[
        "open",
        "source_subclass",
        "file_like",
        "iterate",
        "concatenate",
        "dask",
        "dask_no_open",
        "graphed",
        "graphed_partitions",
    ],
)
def test_unknown_option_raises_for_uproot_source(call):
    with pytest.raises(TypeError, match="bogus"):
        call(skhep_testdata.data_path(ZMUMU), bogus=1)


def test_valid_options_accepted_for_uproot_source():
    path = skhep_testdata.data_path(ZMUMU)
    with uproot.open(path, handler=MEMMAP, num_workers=2) as f:
        assert f["events"].num_entries == 2304


def test_non_source_handler_is_rejected():
    def handler(file_path, **options):
        return MEMMAP(file_path, **options)

    path = skhep_testdata.data_path(ZMUMU)
    with pytest.raises(TypeError, match="not a class object inheriting from Source"):
        uproot.open(path, handler=handler, bogus=1)


@pytest.mark.parametrize(
    "call",
    [
        lambda p, **o: uproot.open(p, **o).close(),
        lambda p, **o: uproot.open(p, handler=MyFSSpec, **o).close(),
        lambda p, **o: next(uproot.iterate(p + ":events", **o)),
        lambda p, **o: uproot.concatenate(p + ":events", library="np", **o),
        _dask,
        lambda p, **o: _dask(p, open_files=False, **o),
    ],
    ids=["open", "fsspec_subclass", "iterate", "concatenate", "dask", "dask_no_open"],
)
def test_typo_warns_for_fsspec_source(call):
    path = skhep_testdata.data_path(ZMUMU)
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'") as record:
        call(path, num_wrokers=2)
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("num_wrokers", "num_workers"),
        ("num_worker", "num_workers"),
        ("numworkers", "num_workers"),
        ("handlr", "handler"),
        ("timout", "timeout"),
        ("begin_chunksize", "begin_chunk_size"),
        ("max_workers", None),
        ("num_threads", None),
        ("use_threading", None),
        ("max_header_bytes", None),
        ("headers", None),
        ("anon", None),
        ("key", None),
        ("token", None),
        ("client_kwargs", None),
        ("block_size", None),
        ("auto_mkdir", None),
    ],
)
def test_fsspec_option_suggestions(key, expected):
    if expected is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            uproot._util._check_unknown_options({key: 1})
    else:
        with pytest.warns(UserWarning, match=f"did you mean.*{expected!r}"):
            uproot._util._check_unknown_options({key: 1})


def test_timeout():
    path = skhep_testdata.data_path(ZMUMU)
    for timeout, expected in [(5, {"timeout": 5}), (None, {})]:
        source = uproot.source.fsspec.FSSpecSource(path, timeout=timeout)
        assert source._fsspec_options == expected
        source.close()

    assert uproot._util.fsspec_storage_options(
        {"timeout": 5, "num_workers": 2, "anon": True}
    ) == {"timeout": 5, "anon": True}

    # nothing listens here; the failure must be a connection error, not a TypeError
    with pytest.raises(OSError):  # noqa: PT011
        uproot.open("http://127.0.0.1:1/f.root", handler=uproot.HTTPSource, timeout=1)


@pytest.mark.parametrize(
    ("handler", "storage_options"),
    [(MEMMAP, {"headers": {"X": "y"}}), (None, {"num_wrokers": 1})],
    ids=["ignored_for_non_fsspec", "merged_for_default"],
)
def test_path_object_storage_options(handler, storage_options):
    path = skhep_testdata.data_path(ZMUMU)

    class StubPath:
        def __str__(self):
            return path

    StubPath.storage_options = storage_options

    if handler is None:
        with pytest.warns(UserWarning, match="did you mean.*'num_workers'"):
            uproot.open(StubPath()).close()
    else:
        with uproot.open(StubPath(), handler=handler) as f:
            assert f["events"].num_entries == 2304
