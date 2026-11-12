# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
uproot.create, uproot.recreate and uproot.update document that extra keyword
arguments are passed to fsspec as storage_options, but any of them raised TypeError.
uproot.dask_write also did not pass its storage_options to the files it writes.
"""

from __future__ import annotations

import os

import awkward as ak
import fsspec.core
import pytest

import uproot


@pytest.fixture
def url_to_fs_kwargs(monkeypatch):
    """Record the keyword arguments of every fsspec.core.url_to_fs call."""
    seen = []
    url_to_fs = fsspec.core.url_to_fs

    def recording_url_to_fs(url, **kwargs):
        seen.append(kwargs)
        return url_to_fs(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", recording_url_to_fs)
    return seen


@pytest.fixture
def make_path(tmp_path, url_to_fs_kwargs):
    """Path of a file that exists only for uproot.update, which needs one."""

    def make(function):
        path = os.path.join(tmp_path, "file.root")
        if function is uproot.update:
            with uproot.recreate(path) as file:
                file["x"] = "hello"
            url_to_fs_kwargs.clear()
        return path

    return make


@pytest.mark.parametrize(
    ("function", "uproot_options"),
    [
        (uproot.create, {"compression": None}),
        (uproot.recreate, {"initial_streamers_bytes": 512}),
        (uproot.update, {"initial_directory_bytes": 512}),
    ],
)
def test_storage_options_are_forwarded(
    make_path, url_to_fs_kwargs, function, uproot_options
):
    path = make_path(function)
    with function(path, auto_mkdir=True, **uproot_options) as file:
        file["y"] = "world"

    # the storage option reaches the filesystem, the uproot option does not
    assert url_to_fs_kwargs
    assert all(kwargs == {"auto_mkdir": True} for kwargs in url_to_fs_kwargs)
    with uproot.open(path) as file:
        assert "y;1" in file.keys()


@pytest.mark.parametrize(
    ("function", "options", "error"),
    [
        # a misspelled uproot option is not handed to the filesystem,
        # which may silently ignore it
        (uproot.create, {"compresion": None}, "did you mean 'compression'"),
        (uproot.recreate, {"compresion": None}, "did you mean 'compression'"),
        (uproot.update, {"uuid_func": None}, "did you mean 'uuid_function'"),
        # compression is an uproot option, but not one uproot.update takes
        (uproot.update, {"compression": None}, r"uproot\.update: 'compression'"),
        # real fsspec options that only loosely resemble compression
        (uproot.create, {"session": None}, None),
        (uproot.recreate, {"compresslevel": None}, None),
        (uproot.update, {"session": None}, None),
    ],
)
def test_leftover_options(make_path, url_to_fs_kwargs, function, options, error):
    path = make_path(function)
    if error is None:
        function(path, **options).close()
        assert url_to_fs_kwargs
        assert all(kwargs == options for kwargs in url_to_fs_kwargs)
    else:
        existed = os.path.exists(path)
        with pytest.raises(TypeError, match=error):
            function(path, **options)
        assert url_to_fs_kwargs == []
        assert os.path.exists(path) == existed


def test_dask_write_forwards_storage_options(tmp_path, monkeypatch):
    dask_awkward = pytest.importorskip("dask_awkward")

    seen = []
    recreate = uproot.recreate

    def recording_recreate(file_path, **options):
        seen.append(options)
        return recreate(file_path, **options)

    monkeypatch.setattr(uproot, "recreate", recording_recreate)

    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=2)
    out = uproot.dask_write(
        dask_arr,
        str(tmp_path),
        prefix="data",
        storage_options={"auto_mkdir": True},
        compute=False,
    )
    # in this process, so that the monkeypatch is visible to the writing tasks
    out.compute(scheduler="sync")

    assert len(seen) == 2
    assert all(options.get("auto_mkdir") is True for options in seen)
    with uproot.open(os.path.join(tmp_path, "data-part0.root")) as file:
        assert file["tree"].num_entries == 1
