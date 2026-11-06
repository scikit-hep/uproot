# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import io
import shutil
import sys
import types

import awkward
import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.interpretation.custom
import uproot.interpretation.grouped
import uproot.sink.file
import uproot.writing._cascade as cascade
from uproot.const import kStartBigFile
from uproot.interpretation.custom import CustomInterpretation
from uproot.interpretation.identify import (
    interpretation_of,
    register_interpretation,
    unregister_interpretation,
)
from uproot.writing.identify import (
    to_TArray,
    to_TAxis,
    to_TH1x,
    to_TH2x,
    to_TH3x,
    to_writable,
)
from uproot.writing.interpret import _as_TGraph


@pytest.fixture
def written_tree(tmp_path):
    with uproot.recreate(tmp_path / "written.root") as file:
        file.mktree("tree", {"x": "int32", "y": "var * float64"})
        file["tree"].extend({"x": [1], "y": awkward.Array([[1.5]])})

    with uproot.open(tmp_path / "written.root") as file:
        yield file["tree"]


class Incomplete(uproot.interpretation.Interpretation):
    pass


@pytest.mark.parametrize(
    "name, call",
    [
        ("cache_key", lambda x: x.cache_key),
        ("typename", lambda x: x.typename),
        ("numpy_dtype", lambda x: x.numpy_dtype),
        ("awkward_form", lambda x: x.awkward_form(None)),
        ("basket_array", lambda x: x.basket_array(*[None] * 8)),
        ("final_array", lambda x: x.final_array(*[None] * 7)),
    ],
)
def test_interpretation_without_a_method(name, call):
    with pytest.raises(NotImplementedError, match=rf"^Incomplete has no {name}$"):
        call(Incomplete())


def test_custom_interpretation_without_basket_array(written_tree):
    class OnlyMatches(uproot.interpretation.custom.CustomInterpretation):
        @classmethod
        def match_branch(cls, branch, context, simplify):
            return True

    register_interpretation(OnlyMatches)
    try:
        branch = written_tree["x"]
        with pytest.raises(
            NotImplementedError, match=r"^OnlyMatches has no basket_array$"
        ):
            branch.array(interpretation=interpretation_of(branch, {}))
    finally:
        unregister_interpretation(OnlyMatches)


def test_grouped_interpretation_has_no_numpy_dtype():
    path = skhep_testdata.data_path("uproot-issue-1043.root")
    with uproot.open(path) as file:
        interpretation = file["FooBar"]["Foo"].interpretation
        assert isinstance(interpretation, uproot.interpretation.grouped.AsGrouped)
        with pytest.raises(
            NotImplementedError, match=r"^AsGrouped has no numpy_dtype$"
        ):
            interpretation.numpy_dtype


@pytest.mark.parametrize(
    "name, call",
    [
        ("typename", lambda x: x.typename),
        ("numpy_dtype", lambda x: x.numpy_dtype),
        ("awkward_form", lambda x: x.awkward_form(None)),
    ],
)
def test_as_binary_is_only_for_reading_bytes(name, call):
    with pytest.raises(NotImplementedError, match=rf"^AsBinary has no {name}$"):
        call(uproot.interpretation.custom.AsBinary())


def string(text):
    return cascade.String(None, text)


def make_key(location=None, seek_location=None, parent_location=100, title="title"):
    return cascade.Key(
        location,
        10,
        5,
        string("TClass"),
        string("name"),
        string(title),
        1,
        parent_location,
        seek_location,
    )


def test_key_read_from_a_file_can_become_big():
    read = cascade.Key.deserialize(
        make_key(location=1000, seek_location=1000).serialize(), 1000, ""
    )
    assert read.big is False

    # the same thing as when FreeSegments moves up the file
    read.location = kStartBigFile + 10
    assert read.big is True
    assert read.num_bytes == 34 + 18


def test_streamer_of_another_type(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        streamer = types.SimpleNamespace(name="Bogus", class_version=1)
        with pytest.raises(
            TypeError,
            match=r"^streamers must be TStreamerInfo or RawStreamerInfo, not SimpleNamespace$",
        ):
            file.file.update_streamers([streamer])


def make_big(directory, num_bytes=kStartBigFile):
    # skip over two gigabytes of file instead of writing them, leaving a hole
    directory.file._cascading.freesegments.allocate(num_bytes)


skip_without_sparse_files = pytest.mark.skipif(
    sys.platform == "win32", reason="this needs a sparse file to be fast"
)


@skip_without_sparse_files
def test_streamers_moving_past_the_boundary_get_a_big_key(tmp_path):
    path = tmp_path / "big.root"
    with uproot.recreate(path, compression=None) as file:
        file["small"] = "small"
        make_big(file)
        # a tree needs many streamers, which don't fit in their allocation
        tree = file.mktree("tree", {"x": "float64"})
        tree.extend({"x": np.array([1.5])})

        streamers = file.file._cascading.streamers
        assert streamers.key.location > kStartBigFile
        assert streamers.key.big
        assert streamers.num_bytes <= streamers.allocation
        end_of_streamers = (
            streamers.key.location + streamers.key.num_bytes + streamers.allocation
        )
        for key in (
            file._cascading.data.get_key("tree"),
            file.file._cascading.freesegments.key,
        ):
            location = key.location if key.seek_location is None else key.seek_location
            assert not streamers.key.location <= location < end_of_streamers

    with uproot.update(path) as file:
        file["tree"].extend({"x": np.array([2.5])})

    with uproot.open(path) as file:
        assert file["tree/x"].array().tolist() == [1.5, 2.5]
        assert file.file.streamer_named("TTree").class_version == 20


@skip_without_sparse_files
def test_keys_read_from_a_small_file_become_big(tmp_path):
    path = tmp_path / "big.root"
    with uproot.recreate(path) as file:
        file["small"] = "small"
        file.mkdir("sub")

    # past 2 GiB, where a key can't hold a pointer in its 4-byte fields
    beyond_32_bits = 2**31 + 1000
    with uproot.update(path) as file:
        make_big(file, beyond_32_bits)
        file["large"] = "large"
        for i in range(40):
            file[f"padding{i}"] = "moves the directory's keys past the boundary"
        file["sub"]["inner"] = "inner"
        file.mkdir("sub2")["inner"] = "inner"

    with uproot.open(path) as file:
        assert file["large"] == "large"
        assert file["padding39"] == "moves the directory's keys past the boundary"
        assert file["sub/inner"] == "inner"
        assert file["sub2/inner"] == "inner"

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    assert not root_file.TestBit(ROOT.TFile.kRecovered)
    assert str(root_file.Get("large")) == "large"
    assert str(root_file.Get("sub/inner")) == "inner"
    root_file.Close()


def write(tmp_path, obj, name="h"):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file[name] = obj
    return path


def root_cells(hist, shape):
    return [hist.GetBinContent(i) for i in range(int(np.prod(shape)))]


class Axis:
    """An axis that follows the PlottableHistogram protocol and nothing else."""

    def __init__(self, edges, label="axis label"):
        self.edges = np.asarray(edges, dtype=float)
        self.label = label

    def __len__(self):
        return len(self.edges) - 1


class Plottable:
    """A PlottableHistogram that isn't from boost-histogram or hist."""

    def __init__(self, values, axes, variances=None, kind="COUNT", title="the title"):
        self._values = values
        self._variances = variances
        self.axes = axes
        self.kind = kind
        self.title = title

    def values(self):
        return self._values

    def variances(self):
        return self._variances

    def counts(self):
        return self._values


class PlottableIgnoringFlow(Plottable):
    """Accepts ``flow`` but has no flow bins to return."""

    def values(self, flow=False):
        return self._values

    def variances(self, flow=False):
        return self._variances


class PlottableWithFlow(Plottable):
    """Has flow bins, all of them filled with the same value."""

    def values(self, flow=False):
        if flow:
            return np.pad(self._values, 1, constant_values=5.0)
        return self._values

    def variances(self, flow=False):
        if self._variances is not None and flow:
            return np.pad(self._variances, 1, constant_values=25.0)
        return self._variances


EDGES = {
    "regular": [[0, 1, 2], [0, 2, 4, 6], [0, 5]],
    "variable": [[0, 1, 3], [0, 1, 2, 4], [0, 1, 4, 8, 9]],
}


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("edges", ["regular", "variable"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cls", [Plottable, PlottableIgnoringFlow, PlottableWithFlow])
def test_plottable_round_trip(tmp_path, ndim, edges, weighted, cls):
    axes = tuple(Axis(x) for x in EDGES[edges][:ndim])
    shape = tuple(len(axis) for axis in axes)
    values = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
    variances = values * 2 if weighted else None

    plottable = cls(values, axes, variances=variances)
    path = write(tmp_path, plottable)

    if cls is PlottableWithFlow:
        values_flow = np.pad(values, 1, constant_values=5.0)
        variances_flow = (
            np.pad(variances, 1, constant_values=25.0) if weighted else None
        )
    else:
        values_flow = np.pad(values, 1)
        variances_flow = np.pad(variances, 1) if weighted else None

    with uproot.open(path) as file:
        hist = file["h"]
        assert hist.classname == f"TH{ndim}D"
        assert hist.title == "the title"
        assert hist.values(flow=True).tolist() == values_flow.tolist()
        if weighted:
            assert hist.variances(flow=True).tolist() == variances_flow.tolist()
        assert hist.member("fEntries") == values_flow.sum()
        # ROOT statistics do not include the flow bins
        assert hist.member("fTsumw") == values.sum()
        for uproot_axis, axis in zip(hist.axes, axes, strict=True):
            assert uproot_axis.edges().tolist() == axis.edges.tolist()
            assert uproot_axis.member("fTitle") == "axis label"

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    hist = root_file.Get("h")
    assert hist.ClassName() == f"TH{ndim}D"
    expected = values_flow.T.reshape(-1).tolist()
    assert root_cells(hist, values_flow.shape) == expected
    if weighted:
        errors = [hist.GetBinError(i) ** 2 for i in range(len(expected))]
        assert errors == pytest.approx(variances_flow.T.reshape(-1).tolist())
    root_axes = [hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()][:ndim]
    for root_axis, axis in zip(root_axes, axes, strict=True):
        assert root_axis.GetNbins() == len(axis)
        assert root_axis.IsVariableBinSize() == (edges == "variable")
        lows = [root_axis.GetBinLowEdge(i + 1) for i in range(len(axis) + 1)]
        assert lows == pytest.approx(axis.edges.tolist())
    root_file.Close()


@pytest.mark.parametrize(
    "obj",
    [
        object(),
        5,
        ("not", "numbers"),
        (np.zeros(3), 5),
        # the "entries" must be 1, 2, or 3 dimensional
        (np.array(1.0), "title"),
        (np.zeros((1, 1, 1, 1)), *[np.arange(2.0)] * 4),
    ],
    ids=["object", "int", "strings", "array-int", "0d-title", "4d"],
)
def test_unrecognized(tmp_path, obj):
    with pytest.raises(TypeError, match="cannot be written to a ROOT file"):
        to_writable(obj)
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(TypeError, match="cannot be written to a ROOT file"):
            file["h"] = obj
        assert file.keys() == []


@pytest.mark.parametrize(
    "dtype, classname",
    [
        (np.int8, "TArrayC"),
        (np.int16, "TArrayS"),
        (np.int32, "TArrayI"),
        (np.int64, "TArrayL"),
        (np.float32, "TArrayF"),
        (np.float64, "TArrayD"),
        # distinct C types that have the same size as one of the above
        (np.intc, "TArrayI"),
        (np.longlong, "TArrayL"),
        (np.int_, f"TArray{'I' if np.dtype(np.int_).itemsize == 4 else 'L'}"),
        (np.double, "TArrayD"),
        # already big-endian
        (">i4", "TArrayI"),
        (">f8", "TArrayD"),
    ],
)
def test_to_TArray(dtype, classname):
    tarray = to_TArray(np.array([1, 2, 3], dtype=dtype))
    assert tarray.classname == classname
    assert tarray.member("fN") == 3
    assert list(tarray) == [1, 2, 3]
    assert tarray._data.dtype == np.dtype(dtype).newbyteorder(">")


HIST_STATS = {
    1: ["fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"],
    2: ["fTsumwy", "fTsumwy2", "fTsumwxy"],
    3: ["fTsumwz", "fTsumwz2", "fTsumwxz", "fTsumwyz"],
}


def hist_args(ndim, data, fSumw2=None):
    out = {"fName": None, "fTitle": "title", "data": data, "fEntries": 10.0}
    for stats in [HIST_STATS[x + 1] for x in range(ndim)]:
        out.update({name: 1.0 for name in stats})
    out["fSumw2"] = fSumw2
    out["fXaxis"] = to_TAxis("xaxis", "", 2, 0.0, 2.0)
    if ndim > 1:
        out["fYaxis"] = to_TAxis("yaxis", "", 2, 0.0, 2.0)
    if ndim > 2:
        out["fZaxis"] = to_TAxis("zaxis", "", 2, 0.0, 2.0)
    return out


TO_TH = {1: to_TH1x, 2: to_TH2x, 3: to_TH3x}


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype, letter",
    [
        (np.int8, "C"),
        (np.int16, "S"),
        (np.int32, "I"),
        (np.float32, "F"),
        (np.float64, "D"),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True], ids=["ndarray", "TArray"])
def test_histogram_type_follows_the_data(tmp_path, ndim, dtype, letter, wrapped):
    data = np.arange(4**ndim, dtype=dtype) % 100
    hist = TO_TH[ndim](**hist_args(ndim, to_TArray(data) if wrapped else data))
    assert hist.classname == f"TH{ndim}{letter}"

    path = write(tmp_path, hist)
    with uproot.open(path) as file:
        assert file["h"].classname == f"TH{ndim}{letter}"
        assert file["h"].values(flow=True).T.reshape(-1).tolist() == data.tolist()

    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    root_hist = root_file.Get("h")
    assert root_hist.ClassName() == f"TH{ndim}{letter}"
    assert root_cells(root_hist, data.shape) == data.tolist()
    root_file.Close()


X = np.array([1.0, 2.0, 3.0])


Y = np.array([2.0, 4.0, 8.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"x_errors": X}, "requires both x_errors and y_errors"),
        ({"y_errors": X}, "requires both x_errors and y_errors"),
        (
            {"x_errors": X, "y_errors": X, "y_errors_high": X},
            "symmetrical errors OR asymmetrical errors, but not both",
        ),
        (
            {"x_errors_low": X, "x_errors_high": X, "y_errors_low": X},
            "requires all of the following",
        ),
        ({"title": "a;b"}, "can't contain ';'"),
        ({"xAxisLabel": "a;b"}, "can't contain ';'"),
        ({"yAxisLabel": "a;b"}, "can't contain ';'"),
        ({"minY": "low"}, "minY has to be None or a number, not <class 'str'>"),
        ({"maxY": "high"}, "maxY has to be None or a number, not <class 'str'>"),
        (
            {"x_errors": X[:2], "y_errors": X},
            "Length of all error arrays has to be the same",
        ),
        (
            {
                "x_errors_low": X,
                "x_errors_high": X[:2],
                "y_errors_low": X,
                "y_errors_high": X,
            },
            "Length of errors all error arrays has to be the same",
        ),
    ],
)
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _as_TGraph(X, Y, **kwargs)


def test_tell_reports_the_position_after_the_last_operation(tmp_path):
    sink = uproot.sink.file.FileSink(str(tmp_path / "file.bin"))
    assert sink.tell() == 0
    sink.write(5, b"abc")
    assert sink.tell() == 8
    sink.read(1, 2)
    assert sink.tell() == 3
    sink.close()
    with pytest.raises(OSError, match="file sink is closed"):
        sink.tell()


def test_tell_of_a_file_like_object():
    sink = uproot.sink.file.FileSink(io.BytesIO())
    sink.write(5, b"abc")
    assert sink.tell() == 8


def root_entries(path, name, *branches):
    """Every entry of the TTree, as seen by ROOT, as a list of lists."""
    ROOT = pytest.importorskip("ROOT")

    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get(name)
    out = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        entry = []
        for branch in branches:
            value = getattr(tree, branch)
            if hasattr(value, "__len__") and not isinstance(value, str):
                length = int(getattr(tree, "n" + branch))
                value = [value[j] for j in range(length)]
            entry.append(value)
        out.append(entry)
    root_file.Close()
    return out


def test_branch_types_can_be_any_mapping(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", types.MappingProxyType({"ab": "float64"}))
        tree.extend({"ab": np.array([1.5, 2.5])})

    with uproot.open(path) as file:
        assert file["t"].keys() == ["ab"]
        assert file["t"]["ab"].typename == "double"
        assert file["t"]["ab"].array(library="np").tolist() == [1.5, 2.5]

    assert root_entries(path, "t", "ab") == [[1.5], [2.5]]


@pytest.mark.parametrize(
    "branch_type, data, expected",
    [
        (
            "(int32, float64)",
            ak.Array([(1, 1.5), (2, 2.5)]),
            [[1, 1.5], [2, 2.5]],
        ),
        (
            "var * (int32, float64)",
            ak.Array([[(1, 1.5), (2, 2.5)], []]),
            [[[1, 2], [1.5, 2.5]], [[], []]],
        ),
    ],
    ids=["flat", "jagged"],
)
def test_tuples_become_numbered_branches(tmp_path, branch_type, data, expected):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"r": branch_type}, counter_name=lambda x: "n" + x)
        tree.extend({"r": data})

    with uproot.open(path) as file:
        assert file["t"]["r_0"].array().tolist() == [x[0] for x in expected]
        assert file["t"]["r_1"].array().tolist() == [x[1] for x in expected]


def test_update_leaves_out_an_uninterpretable_branch(tmp_path):
    path = tmp_path / "out.root"
    shutil.copy(skhep_testdata.data_path("uproot-issue-400.root"), path)

    with uproot.update(path) as file:
        tree = file["orange"]
        with pytest.raises(
            NotImplementedError, match="TBranches that it can't interpret"
        ):
            tree.extend({"orange": np.array([1], dtype="i4")})


class ObjectDtype(CustomInterpretation):
    numpy_dtype = np.dtype("O")

    @classmethod
    def match_branch(cls, branch, context, simplify):
        return branch.name == "x"


class NoneDtype(ObjectDtype):
    numpy_dtype = None


class MissingDtype(CustomInterpretation):
    @classmethod
    def match_branch(cls, branch, context, simplify):
        return branch.name == "x"


@pytest.fixture
def plain_tree(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.mktree("t", {"x": "i4"}).extend({"x": np.array([1, 2], dtype="i4")})
    return path


@pytest.mark.parametrize("custom", [NoneDtype, MissingDtype])
def test_update_leaves_out_a_custom_interpretation_without_dtype(plain_tree, custom):
    uproot.register_interpretation(custom)
    try:
        with uproot.update(plain_tree) as file:
            with pytest.raises(NotImplementedError, match=r"extend"):
                file["t"].extend({"x": np.array([3], dtype="i4")})
    finally:
        uproot.unregister_interpretation(custom)

    with uproot.open(plain_tree) as file:
        assert file["t/x"].array(library="np").tolist() == [1, 2]


def test_categorical_strings(tmp_path):
    categories = ak.Array(["one", "three"]).layout
    index = ak.index.Index64([0, 1, 0])

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"s": "string"})
        tree.extend({"s": ak.Array(ak.contents.IndexedArray(index, categories))})

    with uproot.open(path) as file:
        assert file["t/s"].array().tolist() == ["one", "three", "one"]

    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("t")
    found = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        found.append(tree.GetLeaf("s").GetValueString())
    assert found == ["one", "three", "one"]
    root_file.Close()


@pytest.mark.parametrize(
    "array, counts, message",
    [
        (
            ak.Array([1, 2, 3]),
            [1, 1, 1],
            "jagged branch 'x' can't be filled with this",
        ),
        (
            ak.to_regular(ak.Array([[1], [2], [3]])),
            [1, 1, 1],
            "jagged branch 'x' can't be filled with this",
        ),
        (
            ak.Array([[1, None], [], [3]]),
            [2, 0, 1],
            "jagged branch 'x' can't be filled with lists of this",
        ),
        (
            ak.Array([["one"], [], ["three"]]),
            [1, 0, 1],
            "jagged branch 'x' can't be filled with lists of this",
        ),
    ],
    ids=["flat", "regular", "option", "strings"],
)
def test_jagged_branch_rejects_other_arrays(tmp_path, array, counts, message):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"x": "var * int32"})
        with pytest.raises(TypeError, match=message):
            tree.extend({"x": array, "nx": np.array(counts)})


def histogram():
    return np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0])


def make_nested(file):
    file["a"] = "one"
    file["a"] = "two"
    file["sub/b"] = histogram()
    file["sub/deeper/c"] = "three"
    file["d"] = histogram()


def test_listing_names(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)

        everything = [
            "a;1",
            "a;2",
            "sub;1",
            "sub/b;1",
            "sub/deeper;1",
            "sub/deeper/c;1",
            "d;1",
        ]
        assert list(file) == everything
        assert list(file._ipython_key_completions_()) == everything
        assert file.keys(recursive=False) == ["a;1", "a;2", "sub;1", "d;1"]
        # without cycle numbers, each name once
        assert file.keys(cycle=False, recursive=False) == ["a", "sub", "d"]

        # as when reading, both the name in the subdirectory and the path must match
        assert file.keys(filter_name="*b") == ["sub;1", "sub/b;1"]
        assert file.keys(filter_name="*c", cycle=False) == ["sub/deeper/c"]
        assert file.keys(filter_name="b") == []
        assert file.keys(filter_classname="TH1D") == ["sub/b;1", "d;1"]
        written = [file.keys(cycle=cycle) for cycle in (True, False)]

    with uproot.open(tmp_path / "out.root") as file:
        assert [file.keys(cycle=cycle) for cycle in (True, False)] == written


def test_values_and_items(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        make_nested(file)

        # each name once, as its highest cycle, as when reading
        assert file.values(filter_classname="TObjString") == ["two", "three"]
        assert file.values(filter_name="nothing") == []

        # each name once, as its highest cycle
        assert file.keys(filter_classname="TObjString", cycle=False) == [
            "a",
            "sub/deeper/c",
        ]
        assert file.items(filter_classname="TObjString", cycle=False) == [
            ("a", "two"),
            ("sub/deeper/c", "three"),
        ]
        assert file.items(filter_classname="TObjString", recursive=False) == [
            ("a;1", "one"),
            ("a;2", "two"),
        ]

    with uproot.open(tmp_path / "out.root") as file:
        assert file.values(filter_classname="TObjString") == ["two", "three"]


def test_copy_from_directories(tmp_path):
    source = tmp_path / "source.root"
    with uproot.recreate(source) as file:
        file["top"] = "one"
        file["sub/first"] = "two"
        file["sub/second"] = histogram()
        file.mkdir("empty")

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        with uproot.open(source) as readable:
            file.copy_from(readable)

    with uproot.open(path) as file:
        assert sorted(file.keys(cycle=False)) == [
            "empty",
            "sub",
            "sub/first",
            "sub/second",
            "top",
        ]
        assert file["sub/first"] == "two"

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    assert str(root_file.Get("sub/first")) == "two"
    assert root_file.Get("sub/second").GetBinContent(2) == 2
    assert root_file.GetDirectory("empty").GetListOfKeys().GetSize() == 0
    root_file.Close()


def test_a_streamer_after_a_rejected_one_with_the_same_name(tmp_path):
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as file:
        streamer = file.file.streamer_named("TAttLine", 1)
    bogus = types.SimpleNamespace(name="TAttLine", class_version=1)

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        with pytest.raises(TypeError, match="not SimpleNamespace"):
            file.file.update_streamers([bogus])
        file.file.update_streamers([streamer])

    with uproot.open(path) as file:
        assert list(file.file.streamers) == ["TAttLine"]
        written = file.file.streamer_named("TAttLine", 1)
        assert [x.name for x in written.elements] == [
            "fLineColor",
            "fLineStyle",
            "fLineWidth",
        ]


def test_a_rejected_batch_of_streamers_adds_none_of_them(tmp_path):
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as file:
        line = file.file.streamer_named("TAttLine", 1)
        fill = file.file.streamer_named("TAttFill", 1)
    bogus = types.SimpleNamespace(name="Bogus", class_version=1)

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        # the valid streamer before the invalid one must not be half-added
        with pytest.raises(TypeError, match="not SimpleNamespace"):
            file.file.update_streamers([line, bogus])
        file.file.update_streamers([line, fill])

    with uproot.open(path) as file:
        assert sorted(file.file.streamers) == ["TAttFill", "TAttLine"]
        assert [x.name for x in file.file.streamer_named("TAttLine", 1).elements] == [
            "fLineColor",
            "fLineStyle",
            "fLineWidth",
        ]
        assert [x.name for x in file.file.streamer_named("TAttFill", 1).elements] == [
            "fFillColor",
            "fFillStyle",
        ]


@pytest.fixture
def two_cycles_of_each(tmp_path):
    # h;1 is a TH1D and h;2 an RNTuple; d/x;1 and d/x;2 are TH1Ds of 1 and 2 entries
    path = tmp_path / "cycles.root"
    with uproot.recreate(path) as file:
        file["h"] = np.histogram([1, 2, 3])
        file["d/x"] = np.histogram([1])
        file["d/x"] = np.histogram([1, 2])
    with uproot.update(path) as file:
        file["h"] = {"x": np.array([1, 2, 3])}
    return path


def summary(directory, **kwargs):
    def describe(obj):
        if isinstance(obj, (uproot.ReadOnlyDirectory, uproot.WritableDirectory)):
            return "directory"
        elif isinstance(obj, uproot.writing.writable.WritableNTuple):
            return "ROOT::RNTuple"
        elif obj.classname == "TH1D":
            return f"TH1D of {obj.member('fEntries'):g}"
        else:
            return obj.classname

    return (
        list(directory.iterkeys(**kwargs)),
        list(directory.iterclassnames(**kwargs)),
        [(name, describe(obj)) for name, obj in directory.iteritems(**kwargs)],
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"cycle": False},
            (
                ["h", "d", "d/x"],
                [("h", "ROOT::RNTuple"), ("d", "TDirectory"), ("d/x", "TH1D")],
                [("h", "ROOT::RNTuple"), ("d", "directory"), ("d/x", "TH1D of 2")],
            ),
        ),
        (
            # the highest cycle that passes the filter
            {"cycle": False, "filter_classname": "TH1*"},
            (
                ["h", "d/x"],
                [("h", "TH1D"), ("d/x", "TH1D")],
                [("h", "TH1D of 3"), ("d/x", "TH1D of 2")],
            ),
        ),
        (
            {"cycle": False, "filter_name": "d/*"},
            (["d/x"], [("d/x", "TH1D")], [("d/x", "TH1D of 2")]),
        ),
        (
            {"cycle": True, "filter_name": "d/x"},
            (
                ["d/x;1", "d/x;2"],
                [("d/x;1", "TH1D"), ("d/x;2", "TH1D")],
                [("d/x;1", "TH1D of 1"), ("d/x;2", "TH1D of 2")],
            ),
        ),
    ],
)
def test_names_without_cycles_are_their_highest_cycle(
    two_cycles_of_each, kwargs, expected
):
    with uproot.update(two_cycles_of_each) as file:
        assert summary(file, **kwargs) == expected
    with uproot.open(two_cycles_of_each) as file:
        assert summary(file, **kwargs) == expected
