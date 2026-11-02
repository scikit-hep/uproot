# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed``'s refused keywords and the public ``uproot.graphed_partitions``."""

from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402

REMOTE = "memory://not-there.root:events"


def _make_tree(path):
    with uproot.recreate(path) as f:
        f.mktree("events", {"x": "f8"})
        f["events"].extend({"x": np.arange(10.0)})
    return path


@pytest.fixture
def local(tmp_path):
    return _make_tree(os.path.join(tmp_path, "a.root")) + ":events"


@pytest.fixture
def form(local):
    with uproot.open(local) as tree:
        return tree.arrays(library="ak").layout.form


@pytest.mark.parametrize("name", ["entry_start", "entry_stop"])
def test_an_entry_range_is_refused_with_the_route_that_gives_one(local, name):
    for files in (local, REMOTE):
        with pytest.raises(TypeError, match=f"got {name}=.*'steps'"):
            uproot.graphed(files, **{name: 5})

    path, _, tree = local.rpartition(":")
    steps = uproot.graphed(
        {path: {"object_path": tree, "steps": [[2, 5]]}},
        session=graphed.new_session(),
    )
    assert graphed.materialize(steps.x).tolist() == [2.0, 3.0, 4.0]


@pytest.mark.parametrize("read", [uproot.graphed, uproot.graphed_partitions])
@pytest.mark.parametrize("option", ["foo_bar", "anon", "num_wrokers"])
def test_an_unknown_option_is_refused_when_every_file_is_local(
    tmp_path, local, read, option
):
    path = local.rpartition(":")[0]
    second = _make_tree(os.path.join(tmp_path, "b.root")) + ":events"
    with uproot.open(local) as tree:
        for files in (
            local,
            [local, second],
            {path: "events"},
            os.path.join(tmp_path, "*.root:events"),
            f"file://{local}",
            tree,
        ):
            with pytest.raises(TypeError, match=option):
                read(files, **{option: 1})


@pytest.mark.parametrize(
    "option", ["object_cache", "decompression_executor", "interpretation_executor"]
)
def test_a_file_keyword_is_not_an_unknown_option(local, option):
    uproot.graphed(local, session=graphed.new_session(), **{option: None})
    assert len(uproot.graphed_partitions(local, **{option: None})) == 1


def test_a_tree_from_a_file_object_without_iteration_is_all_local(local):
    class Reader:
        def __init__(self, data):
            self.raw = io.BytesIO(data)

        def read(self, *args):
            return self.raw.read(*args)

        def seek(self, *args):
            return self.raw.seek(*args)

        def tell(self):
            return self.raw.tell()

    with open(local.rpartition(":")[0], "rb") as f:
        data = f.read()
    with uproot.open(Reader(data)) as tree_file:
        parts = uproot.graphed_partitions(tree_file["events"])
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 10)]


def test_a_storage_option_passes_with_a_chain_over_a_local_file(local, tmp_path):
    cache = str(tmp_path / "cache")
    array = uproot.graphed(
        f"simplecache::{local}",
        session=graphed.new_session(),
        cache_storage=cache,
    )
    assert graphed.materialize(array.x).tolist()[:2] == [0.0, 1.0]
    assert os.listdir(cache)


def test_a_storage_option_passes_with_a_remote_file(local, form):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = graphed.new_session()
        uproot.graphed([local, REMOTE], open_files=False, anon=True, session=session)
        uproot.graphed(REMOTE, known_base_form=form, anon=True, name="remote")
        parts = uproot.graphed_partitions([local, REMOTE], open_files=False, anon=True)
    assert len(parts) == 2


def test_a_misspelled_option_only_warns_with_a_remote_file(local):
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'") as record:
        uproot.graphed_partitions([local, REMOTE], open_files=False, num_wrokers=1)
    assert record[0].filename == __file__


def test_graphed_partitions_is_public_where_the_step_size_refusal_points(local):
    with pytest.raises(TypeError) as refusal:
        uproot.graphed(local, step_size=5)
    assert "uproot.graphed_partitions(step_size=...)" in str(refusal.value)
    assert "uproot._graphed" not in str(refusal.value)

    parts = uproot.graphed_partitions(local, step_size=5)
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 5), (5, 10)]
