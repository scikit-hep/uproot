# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed`` opens only the first file that has the ``TTree`` (``open_files=True`` opens
every file), and ``uproot.root_write`` writes ROOT parts beside a ``graphed.aggregate_plan``'s
reductions from the same read."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")

import graphed  # noqa: E402
import graphed.write  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402


def _make_tree(path, n=10, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return path


def _sum_plan(out, **kwargs):
    return graphed.aggregate_plan(
        out,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        **kwargs,
    )


def _run(plan):
    return SequentialRunner().run(plan).value


def _source_of(array):
    _nid, src = uproot._graphed._uproot_source(array, "test")
    return src


def _opens(monkeypatch):
    opened = []
    real_open = uproot.reading.ReadOnlyFile.__init__

    def counting(self, *args, **kwargs):
        opened.append(args[0])
        real_open(self, *args, **kwargs)

    monkeypatch.setattr(uproot.reading.ReadOnlyFile, "__init__", counting)
    return opened


def _reads(monkeypatch, array):
    reads = []
    source = _source_of(array)
    real = source.read_range

    def counting(tree, keys, start, stop):
        reads.append((tree.file.file_path, start, stop))
        return real(tree, keys, start, stop)

    monkeypatch.setattr(source, "read_range", counting)
    return reads


def _keys(array):
    return [(f, o) for f, o, _ in _source_of(array)._file_tree]


# ---- lazy open -----------------------------------------------------------------------------
def test_only_the_first_file_with_the_tree_is_opened(tmp_path, monkeypatch):
    nott = _make_tree(os.path.join(tmp_path, "nott.root"), name="other")
    a, b, c = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "abc")
    files = [f + ":events" for f in (nott, a, b, c)]
    opened = _opens(monkeypatch)
    lazy = uproot.graphed(files, allow_missing=True)
    assert opened == [nott, a]
    assert [p.uri for p in _source_of(lazy).partitions(1)] == [a, b, c]
    opened.clear()
    eager = uproot.graphed(files, allow_missing=True, open_files=True, name="eager")
    assert opened == [nott, a, b, c]
    assert [p.uri for p in _source_of(eager).partitions(1)] == [a, b, c]
    opened.clear()
    assert _run(_sum_plan(lazy.x)) == 135.0
    assert sorted(opened) == [a, b, c]  # each file is opened when it is read
    with pytest.raises(ValueError, match="no TTrees found in\n\n    {'.*nott.root'"):
        uproot.graphed(files[:1], allow_missing=True)
    with pytest.raises(TypeError, match="known_base_form= opens no file"):
        uproot.graphed(
            files[1:],
            known_base_form=lazy.session.form(lazy).tt.layout.form,
            open_files=True,
        )


def test_a_branch_missing_from_a_later_file_fails_its_read(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = os.path.join(tmp_path, "b.root")
    with uproot.recreate(b) as f:
        f.mktree("events", {"x": "f8"})
        f["events"].extend({"x": np.arange(3, dtype="f8")})
    lazy = uproot.graphed([a + ":events", b + ":events"])
    assert lazy.session.form(lazy).describe() == "## * {x: float64, y: float64}"
    assert _run(_sum_plan(lazy.x)) == 48.0
    with pytest.raises(uproot.KeyInFileError, match="y"):
        _run(_sum_plan(lazy.y))
    eager = uproot.graphed(
        [a + ":events", b + ":events"], open_files=True, name="eager"
    )
    assert eager.session.form(eager).describe() == "## * {x: float64}"


def test_unopened_files_resolve_a_leaf_branch_and_a_duplicate_like_the_opened_one(
    tmp_path,
):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    leaf = uproot.graphed([a + ":events/x", b + ":events/x"])
    assert _keys(leaf) == [(a, "/events;1"), (b, "events")]
    assert _run(_sum_plan(leaf.x)) == 90.0
    assert _keys(uproot.graphed([a, b, a], name="dup")) == [
        (a, "/events;1"),
        (b, None),
        (a, "/events;1"),
    ]


@pytest.mark.parametrize("open_files", [False, True])
@pytest.mark.parametrize(
    "spellings",
    [("events:x", "events:x"), ("events/x", "events:x"), ("events:x", "events/x")],
    ids=["colon-twice", "slash-then-colon", "colon-then-slash"],
)
def test_colon_and_slash_spellings_of_a_leaf_resolve_alike(
    tmp_path, spellings, open_files
):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    g = uproot.graphed(dict(zip((a, b), spellings)), open_files=open_files)
    assert _keys(g) == [(a, "/events;1"), (b, "/events;1")]
    assert _run(_sum_plan(g.x)) == 90.0


@pytest.mark.parametrize("open_files", [False, True])
@pytest.mark.parametrize("later", ["events:x", "events/x"])
def test_a_later_file_without_the_colon_selected_tree_is_refused(
    tmp_path, later, open_files
):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"), name="other")
    with pytest.raises(uproot.KeyInFileError, match="not found: 'events'"):
        uproot.graphed({a: "events:x", b: later}, open_files=open_files)


@pytest.mark.parametrize(
    "later",
    [
        lambda b: b + ":events/y",
        lambda b: b + ":events",
        lambda b: b,
        lambda b: [b + ":events/y", b + ":events/x"],
    ],
    ids=["other-leaf", "tree", "no-path", "listed-twice"],
)
@pytest.mark.parametrize("open_files", [False, True])
def test_a_later_entry_selecting_another_branch_is_refused(tmp_path, later, open_files):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    rest = later(b)
    files = [a + ":events/x", *(rest if isinstance(rest, list) else [rest])]
    with pytest.raises(ValueError, match="TBranch"):
        uproot.graphed(files, open_files=open_files)


@pytest.mark.parametrize("open_files", [False, True])
def test_a_later_entry_selecting_a_branch_of_a_whole_tree_is_refused(
    tmp_path, open_files, monkeypatch
):
    import uproot._graphed

    opened = []
    real = uproot._graphed._open_directory

    def spy(*args):
        opened.append(real(*args))
        return opened[-1]

    monkeypatch.setattr(uproot._graphed, "_open_directory", spy)
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    with pytest.raises(ValueError, match="selects the TBranch 'x', but the first"):
        uproot.graphed([a + ":events", b + ":events/x"], open_files=open_files)
    assert [d.file.closed for d in opened] == [True, True]


def test_an_rntuple_read_with_align_baskets_names_its_file(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    nt = os.path.join(tmp_path, "nt.root")
    with uproot.recreate(nt) as f:
        f["nt"] = ak.Array([{"x": 1.0, "y": 2.0}])
    files = [a + ":events", nt + ":nt"]
    lazy = uproot.graphed(files, align_baskets=True)
    with pytest.raises(
        NotImplementedError, match=r"the RNTuple in '.*nt\.root' has none"
    ):
        _run(_sum_plan(lazy.x))
    with pytest.raises(NotImplementedError, match="align_baskets"):
        uproot.graphed(files, align_baskets=True, open_files=True)


def test_an_rntuple_opened_to_resolve_an_entry_is_refused_with_align_baskets(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    nt = os.path.join(tmp_path, "nt.root")
    with uproot.recreate(nt) as f:
        f["events"] = ak.Array([{"x": 1.0, "y": 2.0}])
    with pytest.raises(NotImplementedError, match="an RNTuple has none"):
        uproot.graphed(
            [a + ":events", nt + ":events", nt + ":events;1"], align_baskets=True
        )


@pytest.mark.parametrize(
    ("spelling", "opens"),
    [
        (lambda a, b: [a, b, a + ":events"], "a"),
        (lambda a, b: [a + ":events", b, a], "a"),
        (lambda a, b: [a, b, b + ":events"], "ab"),
        (lambda a, b: [a + ":events/x", b + ":events/x", a + ":events/x"], "a"),
    ],
    ids=["first-again-with-path", "first-again-without", "later-twice", "leaf"],
)
def test_a_tree_listed_twice_in_two_spellings_is_a_refused_duplicate(
    tmp_path, monkeypatch, spelling, opens
):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    opened = _opens(monkeypatch)
    g = uproot.graphed(spelling(a, b))
    assert opened == [
        {"a": a, "b": b}[x] for x in opens
    ]  # a file listed once is not opened
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")


def test_two_trees_of_one_file_are_two_inputs(tmp_path, monkeypatch):
    a = os.path.join(tmp_path, "a.root")
    with uproot.recreate(a) as f:
        for name in ("events", "other"):
            f.mktree(name, {"x": "f8", "y": "f8"})
            f[name].extend({"x": np.arange(10, dtype="f8"), "y": np.zeros(10)})
    b = _make_tree(os.path.join(tmp_path, "b.root"))
    opened = _opens(monkeypatch)
    g = uproot.graphed([a + ":events", b + ":events", a + ":other"])
    assert opened == [a]
    assert _keys(g) == [(a, "/events;1"), (b, "events"), (a, "/other;1")]
    paths = uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")
    assert [len(t) for t in _trees(paths, "tree")] == [10, 10, 10]
    c = _make_tree(os.path.join(tmp_path, "c.root"))
    missing = uproot.graphed([b, c, c + ":nope"], allow_missing=True, name="missing")
    assert _keys(missing) == [(b, "/events;1"), (c, "/events;1")]  # c is listed twice


def test_explicit_steps_of_unopened_files_are_their_partitions(tmp_path):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    g = uproot.graphed(
        {
            a: {"object_path": "events", "steps": [[0, 4], [4, 10]]},
            b: {"object_path": "events", "steps": [[2, 5]]},
        }
    )
    assert [
        (p.uri, p.entry_start, p.entry_stop) for p in _source_of(g).partitions(1)
    ] == [(a, 0, 4), (a, 4, 10), (b, 2, 5)]
    assert _run(_sum_plan(g.x)) == 45.0 + 9.0


# ---- root_write ----------------------------------------------------------------------------
needs_part_write = pytest.mark.skipif(
    uproot._util.parse_version(graphed.__version__)
    < uproot._util.parse_version("0.0.9"),
    reason="root_write needs graphed 0.0.9",
)


def _trees(paths, tree="ev"):
    return [uproot.open(p)[tree].arrays() for p in paths]


def _step_name(partition):
    return f"part-{partition.blind_step:05d}.root"


@needs_part_write
def test_a_root_write_beside_a_reduction_reads_once_and_matches_graphed_write(
    tmp_path, monkeypatch
):
    src = _make_tree(os.path.join(tmp_path, "in.root"), n=20) + ":events"
    g = uproot.graphed(src)
    rec = gak.zip({"x": g.x, "y2": g.y * 2})
    reads = _reads(monkeypatch, g)
    plan = graphed.aggregate_plan(
        gak.sum(g.x),
        reduce=lambda vals: (float(vals[0]), [vals[1]]),
        combine=lambda a, b: (a[0] + b[0], a[1] + b[1]),
        empty=lambda: (0.0, []),
        writes=[
            uproot.root_write(
                rec, str(tmp_path / "composed"), name=_step_name, tree_name="ev"
            )
        ],
        steps_per_file=2,
    )
    total, composed = _run(plan)
    assert total == 190.0
    assert len(reads) == 2  # one read per partition serves the sum and the write
    reads.clear()
    standalone = uproot.graphed_write(
        rec,
        str(tmp_path / "alone"),
        steps_per_file=2,
        tree_name="ev",
        executor="thread",
    )
    assert len(reads) == 2
    assert [os.path.basename(p) for p in composed] == [
        os.path.basename(p) for p in standalone
    ]
    for got, want in zip(_trees(composed), _trees(standalone), strict=True):
        assert got.fields == want.fields == ["x", "y2"]
        assert got.tolist() == want.tolist()
    eager = uproot.open(src).arrays()
    assert (
        ak.concatenate(_trees(composed)).tolist()
        == ak.zip({"x": eager.x, "y2": eager.y * 2}).tolist()
    )


@needs_part_write
def test_every_task_of_a_root_write_writes_a_part_an_empty_one_included(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=3))
    plan = graphed.aggregate_plan(
        reduce=lambda vals: list(vals),
        combine=lambda a, b: a + b,
        empty=list,
        writes=[
            uproot.root_write(
                gak.zip({"x": g.x}), str(tmp_path / "out"), name=_step_name
            )
        ],
        steps_per_file=5,
    )
    paths = _run(plan)
    assert [len(t) for t in _trees(paths, "tree")] == [0, 1, 0, 1, 1]


@needs_part_write
def test_root_write_refusals(tmp_path, monkeypatch):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    with pytest.raises(TypeError, match="root_write writes a record's fields"):
        uproot.root_write(g.x, str(tmp_path), name=_step_name)
    monkeypatch.setattr(graphed, "__version__", "0.0.8")
    with pytest.raises(
        ModuleNotFoundError, match=r"^uproot\.root_write needs graphed 0\.0\.9 or newer"
    ):
        uproot.root_write(gak.zip({"x": g.x}), str(tmp_path), name=_step_name)


@needs_part_write
def test_root_write_refuses_a_non_graphed_array(tmp_path):
    with pytest.raises(TypeError, match="root_write expects a uproot.graphed Array"):
        uproot.root_write(np.arange(3), str(tmp_path), name=str)


@needs_part_write
def test_root_write_creates_its_destination_with_storage_options(tmp_path, monkeypatch):
    from fsspec.implementations.memory import MemoryFileSystem

    made = []
    real = MemoryFileSystem.makedirs

    def spy(self, path, exist_ok=False):
        made.append((path, self.storage_options))
        real(self, path, exist_ok=exist_ok)

    monkeypatch.setattr(MemoryFileSystem, "makedirs", spy)
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=4))
    write = uproot.root_write(
        gak.zip({"x": g.x}),
        "memory://u1758",
        name=_step_name,
        storage_options={"token": "t"},
    )
    assert made == []
    (path,) = _run(
        graphed.aggregate_plan(
            g.x,
            reduce=lambda vals: [vals[1]],
            combine=lambda a, b: a + b,
            empty=list,
            writes=[write],
        )
    )
    assert path == "memory:///u1758/part-00000.root"
    assert made == [("/u1758", {"token": "t"})] * 2
    assert MemoryFileSystem().cat_file("/u1758/part-00000.root")[:4] == b"root"


@needs_part_write
def test_root_write_takes_a_path_destination(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=4))
    write = uproot.root_write(gak.zip({"x": g.x}), tmp_path / "out", name=_step_name)
    (path,) = _run(
        graphed.aggregate_plan(
            g.x,
            reduce=lambda vals: [vals[1]],
            combine=lambda a, b: a + b,
            empty=list,
            writes=[write],
        )
    )
    assert path == str(tmp_path / "out" / "part-00000.root")
    assert uproot.open(path)["tree"]["x"].array().tolist() == [0.0, 1.0, 2.0, 3.0]


@needs_part_write
def test_root_write_reports_a_relative_destination_absolute(tmp_path, monkeypatch):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=4))
    monkeypatch.chdir(tmp_path)
    write = uproot.root_write(gak.zip({"x": g.x}), "rel", name=_step_name)
    assert write.destination == os.path.join(os.getcwd(), "rel")
    assert not (tmp_path / "rel").exists()


def test_a_filter_matching_nothing_names_the_one_opened_file(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"))
    with pytest.raises(
        ValueError, match=r"a\.root.*Only this first file is opened.*all 2 files"
    ):
        uproot.graphed([a + ":events", b + ":events"], filter_name="nomatch")
    with pytest.raises(ValueError, match="have no TBranches in common"):
        uproot.graphed(
            [a + ":events", b + ":events"], filter_name="nomatch", open_files=True
        )
