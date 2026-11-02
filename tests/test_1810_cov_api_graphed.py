# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")
graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")

import graphed.aggregate  # noqa: E402
from graphed.awkward import AwkwardBackend  # noqa: E402


def _make_tree(path, n=8):
    with uproot.recreate(path) as file:
        file.mktree("events", {"x": "f8"})
        file["events"].extend({"x": np.arange(n, dtype="f8")})
    return path + ":events"


@pytest.mark.parametrize("backend", [None, AwkwardBackend])
def test_write_with_a_graphed_without_worker_backend(tmp_path, monkeypatch, backend):
    """``graphed`` 0.0.9 has no ``aggregate.worker_backend``: the workers rebuild the session's
    backend from its type, unless given one."""
    monkeypatch.delattr(graphed.aggregate, "worker_backend", raising=False)
    source = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    paths = uproot.graphed_write(
        source,
        os.path.join(tmp_path, "out"),
        steps_per_file=2,
        executor="thread",
        backend=backend,
    )
    assert len(paths) == 2
    back = ak.concatenate([uproot.open(path)["tree"].arrays() for path in paths])
    assert back.x.tolist() == list(range(8))


@pytest.mark.skipif(
    not hasattr(graphed, "current_session"),
    reason="graphed has no one-binding-per-name rule",
)
def test_tree_whose_name_reads_as_a_cycle(tmp_path):
    """The tree in hand has its key as the only spelling: ``a;3`` would open cycle 3 of ``a``."""
    path = os.path.join(tmp_path, "cycle.root")
    with uproot.recreate(path) as file:
        file.mktree("a;3", {"x": "i4"}).extend({"x": np.arange(3, dtype="i4")})

    with uproot.open(path, object_cache=None) as file:
        tree = file.values()[0]
        assert uproot._graphed._spellings(tree, None, tree).paths == {"a;3;1"}

        session = graphed.Session(AwkwardBackend())
        first = uproot.graphed(tree, session=session, name="events")
        second = uproot.graphed(tree, session=session, name="events")
        assert first.node_id == second.node_id
