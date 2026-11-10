# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed``'s refused keywords and the public ``uproot.graphed_partitions``."""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")

import fsspec  # noqa: E402
import graphed  # noqa: E402
from graphed.awkward import AwkwardBackend  # noqa: E402

REMOTE = "memory://not-there.root:events"
GLOB = "/no/such/dir/*.root:events"


def _session():
    return graphed.Session(AwkwardBackend())


def _make_tree(path):
    with uproot.recreate(path) as f:
        f.mktree("events", {"x": "f8"})
        f["events"].extend({"x": np.arange(10.0)})
    return path


@pytest.fixture
def local(tmp_path):
    return _make_tree(os.path.join(tmp_path, "a.root")) + ":events"


@pytest.fixture
def remote(local, tmp_path):
    """A copy of ``local`` that only fsspec's memory filesystem holds."""
    path = f"/{tmp_path.name}.root"
    with open(local.rpartition(":")[0], "rb") as f:
        fsspec.filesystem("memory").pipe(path, f.read())
    return f"memory://{path}:events"


@pytest.fixture
def form(local):
    with uproot.open(local) as tree:
        return tree.arrays(library="ak").layout.form


@pytest.mark.parametrize("name", ["entry_start", "entry_stop"])
def test_an_entry_range_is_refused_with_the_route_that_gives_one(local, name):
    for files in (local, REMOTE):
        with pytest.raises(
            TypeError, match=f"got {name}=.*'steps'.*without align_baskets"
        ):
            uproot.graphed(files, **{name: 5})


@pytest.mark.parametrize(
    "name, route",
    [
        ("entry_start", "files of uproot.graphed"),
        ("entry_stop", "files of uproot.graphed"),
        ("allow_read_errors_with_report", "run_resumable"),
    ],
)
def test_graphed_partitions_refuses_a_reader_keyword_before_touching_files(
    local, monkeypatch, name, route
):
    def touched(*args, **kwargs):
        raise AssertionError("resolved files")

    monkeypatch.setattr(uproot._util, "regularize_files", touched)
    for files in (local, REMOTE, f"simplecache::{REMOTE}"):
        with pytest.raises(
            TypeError, match=f"graphed_partitions.*got {name}=.*{route}"
        ):
            uproot.graphed_partitions(files, open_files=False, **{name: 5})


@pytest.mark.parametrize("read", [uproot.graphed, uproot.graphed_partitions])
def test_an_unknown_option_is_refused_when_every_file_is_local(local, read):
    with uproot.open(local) as tree:
        for files in (local, [local, local], f"file://{local}", tree):
            with pytest.raises(TypeError, match="for local files: 'anon'"):
                read(files, anon=True)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows names cannot hold ':'")
def test_a_local_path_with_a_double_colon_is_local(tmp_path):
    os.mkdir(tmp_path / "run::1")
    path = _make_tree(os.path.join(tmp_path, "run::1", "f.root")) + ":events"
    assert len(uproot.graphed_partitions(path)) == 1
    with pytest.raises(TypeError, match="for local files: 'anon'"):
        uproot.graphed_partitions(path, anon=True)


def test_a_misspelled_option_on_a_local_file_names_the_option_it_resembles(local):
    with pytest.raises(
        TypeError, match=r"'num_wrokers' \(did you mean 'num_workers'\?\)"
    ):
        uproot.graphed(local, num_wrokers=2, session=_session())


@pytest.mark.parametrize(
    "option",
    [
        "object_cache",
        "array_cache",
        "decompression_executor",
        "interpretation_executor",
    ],
)
def test_an_uproot_open_keyword_is_not_an_unknown_option(local, remote, option):
    for files in (local, remote):
        uproot.graphed(files, session=_session(), **{option: None})
        assert len(uproot.graphed_partitions(files, **{option: None})) == 1


def test_a_tree_from_a_file_object_without_iteration_is_all_local(local):
    class Reader(io.BytesIO):
        __iter__ = None

    with open(local.rpartition(":")[0], "rb") as f:
        reader = Reader(f.read())
    with uproot.open(reader) as tree_file:
        parts = uproot.graphed_partitions(tree_file["events"])
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 10)]


def test_a_storage_option_passes_with_a_chain_over_a_local_file(local, tmp_path):
    cache = str(tmp_path / "cache")
    session = _session()
    array = uproot.graphed(
        f"simplecache::{local}", session=session, cache_storage=cache
    )
    assert session.materialize(array.x).tolist()[:2] == [0.0, 1.0]
    assert os.listdir(cache)


def test_a_storage_option_passes_with_a_remote_file(local, form):
    session = _session()
    uproot.graphed([local, REMOTE], open_files=False, anon=True, session=session)
    uproot.graphed(
        REMOTE, known_base_form=form, anon=True, name="remote", session=session
    )
    parts = uproot.graphed_partitions([local, REMOTE], open_files=False, anon=True)
    assert len(parts) == 2


def test_a_misspelled_option_only_warns_with_a_remote_file(local):
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'") as record:
        uproot.graphed_partitions([local, REMOTE], open_files=False, num_wrokers=1)
    assert record[0].filename == __file__


def test_graphed_partitions_is_public_where_the_step_size_refusal_points(local):
    with pytest.raises(TypeError, match=r"uproot\.graphed_partitions\(step_size"):
        uproot.graphed(local, step_size=5)
    parts = uproot.graphed_partitions(local, step_size=5)
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 5), (5, 10)]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: uproot.graphed(GLOB, library="np"), NotImplementedError, "library"),
        (lambda: uproot.graphed(GLOB, session=object()), TypeError, "session="),
        (
            lambda: uproot.graphed_partitions(GLOB, step_size=1, steps_per_file=2),
            TypeError,
            "mutually exclusive",
        ),
        (
            lambda: uproot.graphed(GLOB, handler=uproot.MemmapSource, bogus=1),
            TypeError,
            "MemmapSource: 'bogus'",
        ),
        (
            lambda: uproot.graphed_partitions(
                GLOB, handler=uproot.MemmapSource, bogus=1
            ),
            TypeError,
            "MemmapSource: 'bogus'",
        ),
    ],
)
def test_an_argument_refusal_comes_before_reading_the_glob(call, error, match):
    with pytest.raises(error, match=match):
        call()


@pytest.mark.parametrize("read", [uproot.graphed, uproot.graphed_partitions])
def test_without_graphed_the_install_hint_comes_before_reading_the_glob(
    monkeypatch, read
):
    def missing():
        raise ModuleNotFoundError("install graphed")

    monkeypatch.setattr(uproot.extras, "graphed", missing)
    with pytest.raises(ModuleNotFoundError, match="install graphed"):
        read(GLOB)
