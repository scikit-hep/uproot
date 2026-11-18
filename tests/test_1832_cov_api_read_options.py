# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


@pytest.fixture(scope="module")
def zmumu_path():
    return skhep_testdata.data_path("uproot-Zmumu.root")


@pytest.fixture(scope="module")
def zmumu(zmumu_path):
    """The known values, as read by the simplest path: one branch at a time."""
    with uproot.open(zmumu_path) as file:
        tree = file["events"]
        yield {name: tree[name].array(library="np") for name in ("M", "Run", "Q1")}


@pytest.fixture
def two_files(tmp_path):
    """``a.root`` has ``x`` (0 to 9) and ``y`` (10 times ``x``); ``b.root`` has only ``x``
    (10 to 14)."""
    a, b = str(tmp_path / "a.root"), str(tmp_path / "b.root")
    with uproot.recreate(a) as file:
        x = np.arange(10, dtype=np.int64)
        file.mktree("t", {"x": "int64", "y": "int64"}).extend({"x": x, "y": x * 10})
    with uproot.recreate(b) as file:
        x = np.arange(10, 15, dtype=np.int64)
        file.mktree("t", {"x": "int64"}).extend({"x": x})
    return a, b


def _as_numpy_columns(arrays, library, names):
    """``{name: numpy array}`` from what ``arrays`` gives by default for ``library``."""
    if library == "np":
        return {name: arrays[name] for name in names}
    elif library == "ak":
        return {name: ak.to_numpy(arrays[name]) for name in names}
    return {name: arrays[name].to_numpy() for name in names}


ENTRY_RANGES = [(None, None), (100, 2000), (-500, None), (None, -2000), (2000, 100)]


def _expected(zmumu, entry_start, entry_stop, cut):
    keep = np.zeros(len(zmumu["M"]), dtype=bool)
    keep[entry_start:entry_stop] = True
    if cut:
        keep &= zmumu["M"] > 90
    return {
        "M": zmumu["M"][keep],
        "double_M": zmumu["M"][keep] * 2,
        "Run": zmumu["Run"][keep],
    }


@pytest.mark.parametrize("library", ["np", "ak", "pd"])
@pytest.mark.parametrize("cut", [None, "M > 90"])
@pytest.mark.parametrize(("entry_start", "entry_stop"), ENTRY_RANGES)
def test_arrays_expressions_aliases_cut_and_entries(
    zmumu_path, zmumu, library, cut, entry_start, entry_stop
):
    if library == "pd":
        pytest.importorskip("pandas")
    expected = _expected(zmumu, entry_start, entry_stop, cut)
    with uproot.open(zmumu_path) as file:
        arrays = file["events"].arrays(
            ["M", "double_M", "Run"],
            cut=cut,
            aliases={"double_M": "M * 2"},
            entry_start=entry_start,
            entry_stop=entry_stop,
            library=library,
        )
    columns = _as_numpy_columns(arrays, library, expected)
    for name, values in expected.items():
        assert columns[name].tolist() == values.tolist()


@pytest.mark.parametrize("library", ["np", "ak", "pd"])
@pytest.mark.parametrize("cut", [None, "M > 90"])
@pytest.mark.parametrize(("entry_start", "entry_stop"), ENTRY_RANGES)
@pytest.mark.parametrize("step_size", [1000, "2 kB"])
def test_iterate_is_arrays_in_steps(
    zmumu_path, zmumu, library, cut, entry_start, entry_stop, step_size
):
    if library == "pd":
        pytest.importorskip("pandas")
    expected = _expected(zmumu, entry_start, entry_stop, cut)
    start, stop = uproot.behaviors.TBranch._regularize_entries_start_stop(
        len(zmumu["M"]), entry_start, entry_stop
    )
    with uproot.open(zmumu_path) as file:
        steps = list(
            file["events"].iterate(
                ["M", "double_M", "Run"],
                cut=cut,
                aliases={"double_M": "M * 2"},
                entry_start=entry_start,
                entry_stop=entry_stop,
                step_size=step_size,
                library=library,
                report=True,
            )
        )
    # the reports tile the requested range exactly
    position = start
    for _, report in steps:
        assert report.tree_entry_start == report.global_entry_start == position
        assert report.start == position
        position = report.tree_entry_stop
        assert report.stop == report.global_entry_stop == position
    assert position == stop

    columns = [_as_numpy_columns(arrays, library, expected) for arrays, _ in steps]
    for name, values in expected.items():
        joined = [v for column in columns for v in column[name].tolist()]
        assert joined == values.tolist()


@pytest.mark.parametrize(
    ("library", "how", "expected_type"),
    [
        ("np", None, dict),
        ("np", dict, dict),
        ("np", tuple, tuple),
        ("np", list, list),
        ("ak", dict, dict),
        ("ak", tuple, tuple),
        ("ak", list, list),
        ("pd", dict, dict),
        ("pd", tuple, tuple),
        ("pd", list, list),
    ],
)
def test_how(zmumu_path, zmumu, library, how, expected_type):
    if library == "pd":
        pytest.importorskip("pandas")
    with uproot.open(zmumu_path) as file:
        arrays = file["events"].arrays(
            ["Run", "M"], entry_stop=5, library=library, how=how
        )
    assert type(arrays) is expected_type
    run, mass = arrays.values() if expected_type is dict else arrays
    assert list(run) == zmumu["Run"][:5].tolist()
    assert list(mass) == zmumu["M"][:5].tolist()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"filter_name": "Q*"}, {"Q1", "Q2"}),
        ({"filter_name": "/^[EM]$|^E1$/"}, {"E1", "M"}),
        ({"filter_name": ["Run", "Event"]}, {"Run", "Event"}),
        ({"filter_typename": "int32_t"}, {"Run", "Event", "Q1", "Q2"}),
        (
            {"filter_branch": lambda branch: branch.name.startswith("px")},
            {"px1", "px2"},
        ),
    ],
)
def test_filters_select_the_same_branches_everywhere(zmumu_path, kwargs, expected):
    with uproot.open(zmumu_path) as file:
        tree = file["events"]
        assert set(tree.keys(**kwargs)) == expected
        assert set(tree.arrays(entry_stop=1, library="np", **kwargs)) == expected
        steps = list(tree.iterate(entry_stop=1, step_size=1, library="np", **kwargs))
    assert [set(step) for step in steps] == [expected]
    concatenated = uproot.concatenate(
        {zmumu_path: "events"}, entry_stop=1, library="np", **kwargs
    )
    assert set(concatenated) == expected


def test_iterate_with_a_filter_matching_nothing_yields_nothing(zmumu_path):
    with uproot.open(zmumu_path) as file:
        assert list(file["events"].iterate(filter_name="nonexistent*")) == []


@pytest.mark.parametrize(
    ("entry_start", "entry_stop", "expected"),
    [
        (None, None, list(range(15))),
        (12, None, [12, 13, 14]),
        (None, 3, [0, 1, 2]),
        (8, 12, [8, 9, 10, 11]),
        (10, 10, []),
        (-2, None, [13, 14]),
    ],
)
def test_concatenate_entry_range_across_files(
    two_files, entry_start, entry_stop, expected
):
    a, b = two_files
    out = uproot.concatenate(
        {a: "t", b: "t"},
        ["x"],
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="np",
    )
    assert out["x"].tolist() == expected


def test_concatenate_pandas_index_counts_from_the_first_file(two_files):
    pytest.importorskip("pandas")
    a, b = two_files
    out = uproot.concatenate(
        {a: "t", b: "t"}, ["x"], cut="x % 5 == 0", library="pd", entry_start=1
    )
    assert out["x"].tolist() == [5, 10]
    assert out.index.tolist() == [5, 10]


def test_concatenate_ak_with_cut_alias_and_how(two_files):
    a, b = two_files
    out = uproot.concatenate(
        {a: "t", b: "t"}, ["twice"], cut="x > 11", aliases={"twice": "x * 2"}, how=tuple
    )
    assert out[0].tolist() == [24, 26, 28]


def test_missing_branch_raises_unless_allow_missing(two_files):
    a, b = two_files
    files = {a: "t", b: "t"}

    with pytest.raises(uproot.KeyInFileError, match="y"):
        uproot.concatenate(files, ["y"], library="np")
    with pytest.raises(uproot.KeyInFileError, match="y"):
        list(uproot.iterate(files, ["y"], library="np"))

    tens = list(range(0, 100, 10))
    out = uproot.concatenate(files, ["y"], library="np", allow_missing=True)
    assert out["y"].tolist() == tens
    steps = uproot.iterate(files, ["y"], library="np", allow_missing=True)
    assert [step["y"].tolist() for step in steps] == [tens]


def test_missing_tree_with_allow_missing(two_files):
    a, b = two_files
    files = {a: "t", b: "not_there"}
    with pytest.raises(uproot.KeyInFileError, match="not_there"):
        uproot.concatenate(files, ["x"], library="np")
    out = uproot.concatenate(files, ["x"], library="np", allow_missing=True)
    assert out["x"].tolist() == list(range(10))
    steps = uproot.iterate(files, ["x"], library="np", allow_missing=True)
    assert [step["x"].tolist() for step in steps] == [list(range(10))]


def test_iterate_function_report_and_global_offset(two_files):
    a, b = two_files
    steps = list(
        uproot.iterate({a: "t", b: "t"}, ["x"], step_size=6, library="np", report=True)
    )
    assert [step["x"].tolist() for step, _ in steps] == [
        list(range(6)),
        list(range(6, 10)),
        list(range(10, 15)),
    ]

    reports = [report for _, report in steps]
    assert [(r.tree_entry_start, r.tree_entry_stop) for r in reports] == [
        (0, 6),
        (6, 10),
        (0, 5),
    ]
    assert [(r.start, r.stop) for r in reports] == [(0, 6), (6, 10), (10, 15)]
    assert [r.global_offset for r in reports] == [0, 0, 10]

    last = reports[-1]
    assert os.path.basename(last.file_path) == "b.root"
    assert last.tree.object_path == "/t;1"
    assert last.source is last.tree
    assert last.file is last.tree.file
    assert last.file.file_path == last.file_path
    source = last.file_path + ":/t;1"
    assert repr(last) == f"<Report start=10 stop=15 source={source!r}>"


def test_iterate_method_on_a_leaf_branch(two_files):
    a, _ = two_files
    with uproot.open(a) as file:
        branch = file["t"]["y"]
        steps = list(branch.iterate(step_size=4, library="np"))
        assert [list(step) for step in steps] == [["y"]] * 3
        assert np.concatenate([step["y"] for step in steps]).tolist() == list(
            range(0, 100, 10)
        )
        # a filter that rejects the branch itself leaves nothing to iterate over
        assert list(branch.iterate(filter_branch=lambda b: False)) == []


def test_expressions_on_a_leaf_branch_look_among_its_subbranches(two_files):
    a, _ = two_files
    with uproot.open(a) as file, pytest.raises(uproot.KeyInFileError, match="'y'"):
        list(file["t"]["y"].iterate("y + 1", library="np"))


@pytest.mark.parametrize(
    "expressions",
    [
        pytest.param(["x", 1], id="not a string"),
        pytest.param([("x", None, None)], id="tuple of the wrong length"),
    ],
)
def test_iterable_of_bad_expressions(two_files, expressions):
    a, _ = two_files
    with (
        uproot.open(a) as file,
        pytest.raises(
            TypeError, match="iterable of expressions must be strings or name, Interp"
        ),
    ):
        file["t"].arrays(expressions)


def test_expressions_of_the_wrong_type(two_files):
    a, _ = two_files
    with (
        uproot.open(a) as file,
        pytest.raises(TypeError, match="expressions must be None .* not 5"),
    ):
        file["t"].arrays(5)


def test_expressions_as_name_interpretation_pairs(two_files):
    a, _ = two_files
    # the bytes of x as big-endian floats
    expected = np.arange(10, dtype=">i8").view(">f8").tolist()
    with uproot.open(a) as file:
        tree = file["t"]
        as_dict = tree.arrays({"x": uproot.AsDtype(">f8")}, library="np")
        as_pairs = tree.arrays([("x", np.dtype(">f8")), "y"], library="np")
    assert as_dict["x"].tolist() == expected
    assert as_pairs["x"].tolist() == expected
    assert as_pairs["y"].tolist() == list(range(0, 100, 10))


@pytest.mark.parametrize(
    ("array_cache", "expected_limit"),
    [
        ("1 kB", 1000),
        (2048, 2048),
    ],
)
def test_array_cache_by_memory_size(two_files, array_cache, expected_limit):
    a, _ = two_files
    regularized = uproot.behaviors.TBranch._regularize_array_cache(array_cache, None)
    assert isinstance(regularized, uproot.cache.LRUArrayCache)
    assert regularized.limit == expected_limit
    with uproot.open(a) as file:
        out = file["t"]["x"].array(array_cache=array_cache, library="np")
    assert out.tolist() == list(range(10))


def test_array_cache_of_the_wrong_type(two_files):
    a, _ = two_files
    with (
        uproot.open(a) as file,
        pytest.raises(
            TypeError,
            match="array_cache must be None, a MutableMapping, or a memory size",
        ),
    ):
        file["t"]["x"].array(array_cache=1.5)


def test_concatenate_with_explicit_executors_uses_them(two_files):
    a, b = two_files

    class Counting(uproot.source.futures.TrivialExecutor):
        submitted = 0

        def submit(self, task, /, *args, **kwargs):
            Counting.submitted += 1
            return super().submit(task, *args, **kwargs)

    out = uproot.concatenate(
        {a: "t", b: "t"},
        ["x"],
        library="np",
        decompression_executor=Counting(),
        interpretation_executor=Counting(),
    )
    assert out["x"].tolist() == list(range(15))
    # one basket per file, each decompressed then interpreted
    assert Counting.submitted == 4


def test_num_entries_for_without_fsspec_source(two_files):
    a, _ = two_files
    with uproot.open(a) as file:
        one_basket = file["t"]["x"].basket_uncompressed_bytes(0)
    with uproot.open(a, handler=uproot.source.file.MemmapSource) as file:
        tree = file["t"]
        assert not isinstance(file.file.source, uproot.source.fsspec.FSSpecSource)
        # half of the only TBasket, TKey included, is half of the entries
        assert tree.num_entries_for(f"{one_basket / 2} B", ["x"]) == 5
        assert tree.num_entries_for(f"{one_basket} B", ["x"]) == 10
