# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")


def _make_tree(path, n=10, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return str(path)


@pytest.mark.parametrize(
    "spelling",
    [
        lambda absp, relp: {absp: "events", relp: "events"},
        lambda absp, relp: [absp + ":events", relp + ":events"],
        lambda absp, relp: [absp, relp],
        lambda absp, relp: [absp + ":events", relp],
        lambda absp, relp: {Path(absp).as_uri(): "events", relp: "events"},
        lambda absp, relp: [Path(absp), relp],
        lambda absp, relp: {absp: "events", f"./{relp}": "events"},
    ],
    ids=[
        "dict-spelling",
        "colon-list",
        "bare-list",
        "colon-and-bare",
        "file-protocol-dict",
        "path-object",
        "dot-slash-rel",
    ],
)
def test_same_file_under_two_path_spellings_is_refused(tmp_path, monkeypatch, spelling):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    g = uproot.graphed(spelling(absp, relp))
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out1"), executor="thread")


def test_same_file_under_two_path_spellings_open_files_is_refused(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    g = uproot.graphed({absp: "events", relp: "events"}, open_files=True)
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out2"), executor="thread")


def test_same_file_under_two_path_spellings_known_base_form_is_refused(
    tmp_path, monkeypatch
):
    from uproot._graphed import _base_form_of

    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    tree = uproot.open(absp)["events"]
    base_form = _base_form_of(tree, ["x", "y"], False, None)

    g = uproot.graphed(
        {absp: "events", relp: "events"},
        known_base_form=base_form,
    )
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out3"), executor="thread")


@pytest.mark.parametrize(
    "paths",
    [
        lambda absp, relp: {absp: "events", relp: "/events"},
        lambda absp, relp: {absp: "events;1", relp: "events"},
        lambda absp, relp: {absp: "/events;1", relp: "/events"},
        lambda absp, relp: {absp: None, relp: "events"},
        lambda absp, relp: {absp: "events", relp: None},
        lambda absp, relp: {absp: None, relp: "/events;1"},
    ],
    ids=[
        "leading-slash",
        "cycle-suffix",
        "leading-slash-and-cycle",
        "none-and-name",
        "name-and-none",
        "none-and-slash-cycle",
    ],
)
def test_object_path_duplicate_spellings_known_base_form(tmp_path, monkeypatch, paths):
    from uproot._graphed import _base_form_of

    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    tree = uproot.open(absp)["events"]
    base_form = _base_form_of(tree, ["x", "y"], False, None)

    g = uproot.graphed(paths(absp, relp), known_base_form=base_form)
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_obj"), executor="thread")


def test_hard_link_is_refused(tmp_path):
    absp = _make_tree(tmp_path / "a.root")
    linkp = str(tmp_path / "a_hardlink.root")
    os.link(absp, linkp)

    g = uproot.graphed({absp: "events", linkp: "events"})
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_link"), executor="thread")


def test_case_difference_on_case_insensitive_filesystem(tmp_path):
    absp = _make_tree(tmp_path / "a.root")
    upperp = str(tmp_path / "A.ROOT")
    if not os.path.exists(upperp):
        pytest.skip("Filesystem is case sensitive")

    g = uproot.graphed({absp: "events", upperp: "events"})
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_case"), executor="thread")


def test_different_trees_in_same_file_under_two_path_spellings_are_accepted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "a.root"
    with uproot.recreate(a) as f:
        for name in ("events", "other"):
            f.mktree(name, {"x": "f8", "y": "f8"})
            f[name].extend({"x": np.arange(10, dtype="f8"), "y": np.zeros(10)})
    absp = str(a)
    relp = "a.root"

    g = uproot.graphed([absp + ":events", relp + ":other"])
    paths = uproot.graphed_write(g, str(tmp_path / "out4"), executor="thread")
    assert len(paths) == 2


def test_canonical_path_normalization(tmp_path):
    from uproot._util import canonical_object_path, canonical_path

    assert canonical_path(None) is None
    assert canonical_path(123) == 123

    local_p = tmp_path / "test.root"
    local_p.touch()
    expected_local = str(local_p.resolve())

    assert canonical_path(local_p) == expected_local
    assert canonical_path(str(local_p).encode()) == expected_local

    class BytesPathLike:
        def __init__(self, raw: bytes):
            self._raw = raw

        def __fspath__(self):
            return self._raw

    assert canonical_path(BytesPathLike(str(local_p).encode())) == expected_local

    file_url = local_p.as_uri()
    assert canonical_path(file_url) == expected_local
    assert canonical_path(local_p.as_uri()) == expected_local
    assert (
        canonical_path("file://somehost/tmp/test.root")
        == "file://somehost/tmp/test.root"
    )

    # remote URLs keep their protocol's semantics: "a//b", "a/b" and "a/../b" can differ
    assert canonical_path("memory://bucket/a.root") == canonical_path(
        "memory:///bucket/a.root"
    )
    assert canonical_path("memory:///bucket/a//b.root") != canonical_path(
        "memory:///bucket/a/b.root"
    )
    assert (
        canonical_path("http://User:Secret@example.com/foo/../bar.root")
        == "http://User:Secret@example.com/foo/../bar.root"
    )

    assert canonical_object_path("/events") == "events"
    assert canonical_object_path("events;1") == "events"
    assert canonical_object_path("/events;1") == "events"
    assert canonical_object_path(None) == ""
    assert canonical_object_path("/dir/events;2") == "dir/events"


def test_remote_paths_differing_by_a_double_slash_are_different_files(tmp_path):
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    urls = ["memory:///test_1767/a//b.root", "memory:///test_1767/a/b.root"]
    try:
        for url, value in zip(urls, (1.0, 99.0)):
            local = tmp_path / f"{int(value)}.root"
            with uproot.recreate(local) as f:
                f.mktree("events", {"x": "f8"})
                f["events"].extend({"x": np.array([value])})
            fs.pipe(url[len("memory://") :], local.read_bytes())

        g = uproot.graphed({url: "events" for url in urls})
        out = uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")
        assert [uproot.open(p)["tree"]["x"].array().tolist() for p in sorted(out)] == [
            [1.0],
            [99.0],
        ]
    finally:
        for url in urls:
            if fs.exists(url):
                fs.rm(url)


@pytest.mark.parametrize("none_first", [True, False])
def test_none_is_the_only_tree_whatever_its_name(tmp_path, monkeypatch, none_first):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root", name="other")
    base_form = uproot.open(absp)["other"].arrays().layout.form
    entries = [(absp, None), ("a.root", "other")]

    g = uproot.graphed(
        dict(entries if none_first else entries[::-1]), known_base_form=base_form
    )
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")


def _make_tree_at_cycle_2(path):
    with uproot.recreate(path) as f:
        for n in (3, 5):
            f.mktree("events", {"x": "f8", "y": "f8"})
            f["events"].extend({"x": np.arange(n, dtype="f8"), "y": np.zeros(n)})
    return str(path)


@pytest.mark.parametrize("object_path", ["events", None])
def test_known_base_form_reads_the_latest_cycle(tmp_path, object_path):
    path = _make_tree_at_cycle_2(tmp_path / "c.root")
    base_form = uproot.open(path)["events"].arrays().layout.form

    g = uproot.graphed({path: object_path}, known_base_form=base_form)
    out = uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")
    assert [t.num_entries for t in uproot.open(out[0]).values()] == [5]


def test_known_base_form_none_reads_the_only_tree_whatever_its_name(tmp_path):
    path = _make_tree(tmp_path / "t.root", n=4, name="tree")
    base_form = uproot.open(path)["tree"].arrays().layout.form

    g = uproot.graphed({path: None}, known_base_form=base_form)
    out = uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")
    assert [t.num_entries for t in uproot.open(out[0]).values()] == [4]


@pytest.mark.parametrize("known", [False, True])
def test_workers_read_the_path_as_given(tmp_path, known):
    from uproot._graphed import _uproot_source

    absp = _make_tree(tmp_path / "a.root")
    linkp = tmp_path / "link.root"
    linkp.symlink_to(absp)
    base_form = uproot.open(absp)["events"].arrays().layout.form if known else None

    g = uproot.graphed({str(linkp): "events"}, known_base_form=base_form)
    _, source = _uproot_source(g, "test")
    assert [file_path for file_path, *_ in source._file_tree] == [str(linkp)]


def test_none_and_name_refused_with_a_custom_source_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    base_form = uproot.open(absp)["events"].arrays().layout.form

    g = uproot.graphed(
        {absp: None, "a.root": "events"}, known_base_form=base_form, name="muons"
    )
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out"), executor="thread")
