# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import types

import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.writing.identify import (
    to_TArray,
    to_TAxis,
    to_TH1x,
    to_TH2x,
    to_TH3x,
    to_THashList,
    to_TList,
    to_TObjString,
    to_TProfile,
    to_TProfile2D,
    to_TProfile3D,
    to_TString,
    to_writable,
)


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


def test_plottable_kind():
    plottable = Plottable(np.ones(1), (Axis([0, 1]),), kind="SOMETHING")
    with pytest.raises(ValueError, match="kind='COUNT' or 'MEAN'"):
        to_writable(plottable)


def test_plottable_dimensions():
    plottable = Plottable(np.ones((1, 1, 1, 1)), tuple(Axis([0, 1]) for _ in range(4)))
    with pytest.raises(ValueError, match="between 1 and 3 axes"):
        to_writable(plottable)


@pytest.mark.parametrize("ndim", [2, 3])
def test_plottable_profile_dimensions(ndim):
    plottable = Plottable(
        np.ones((1,) * ndim), tuple(Axis([0, 1]) for _ in range(ndim)), kind="MEAN"
    )
    with pytest.raises(NotImplementedError, match=f"{ndim}D PlottableHistogram"):
        to_writable(plottable)


class ProfileLike(PlottableWithFlow):
    """A profile with the structure of an Uproot TProfile, but not a Model."""

    def __init__(self, model):
        self._model = model
        self._bases = model._bases
        self.member = model.member
        self.axes = (Axis(model.axes[0].edges()),)
        self.kind = "MEAN"
        self.title = model.title

    def values(self, flow=False):
        return self._model.values(flow=flow)

    def variances(self, flow=False):
        return self._model.variances(flow=flow)


def test_plottable_profile_from_members(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as file:
        model = file["hprof"]

    writable = to_writable(ProfileLike(model))
    assert writable.classname == "TProfile"

    path = write(tmp_path, writable)

    with uproot.open(path) as file:
        assert file["h"].values().tolist() == model.values().tolist()
        assert file["h"].member("fTsumwy") == model.member("fTsumwy")

    root_file = ROOT.TFile.Open(str(path))
    profile = root_file.Get("h")
    assert profile.GetEntries() == model.member("fEntries")
    assert [profile.GetBinContent(i + 1) for i in range(len(model.values()))] == (
        pytest.approx(model.values().tolist())
    )
    bin_entries = model.member("fBinEntries")
    assert [profile.GetBinEntries(i) for i in range(len(bin_entries))] == (
        list(bin_entries)
    )
    root_file.Close()


def test_categorical_profile_without_overflow(tmp_path):
    hist = pytest.importorskip("hist")
    ROOT = pytest.importorskip("ROOT")

    profile = hist.Hist(
        hist.axis.StrCategory(["a", "b"], overflow=False, name="x"),
        storage=hist.storage.Mean(),
    )
    profile.fill(["a", "a", "b"], sample=[1.0, 3.0, 5.0])

    root_file = ROOT.TFile.Open(str(write(tmp_path, profile)))
    root_profile = root_file.Get("h")
    assert root_profile.ClassName() == "TProfile"
    assert [root_profile.GetBinEntries(i) for i in range(4)] == [0, 2, 1, 0]
    assert [root_profile.GetBinContent(i) for i in range(1, 3)] == [2, 5]
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


def test_pyroot_object_must_be_a_TObject():
    ROOT = pytest.importorskip("ROOT")
    with pytest.raises(TypeError, match="only instances of TObject .* not TDatime"):
        to_writable(ROOT.TDatime())


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("title", [None, "the title"])
def test_tuple_with_variable_edges(tmp_path, ndim, title):
    ROOT = pytest.importorskip("ROOT")

    edges = [np.array(x, dtype=np.float64) for x in EDGES["variable"][:ndim]]
    shape = tuple(len(x) - 1 for x in edges)
    entries = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)

    obj = (entries, *edges) if title is None else (entries, *edges, title)
    path = write(tmp_path, obj)

    with uproot.open(path) as file:
        assert file["h"].title == ("" if title is None else title)
        assert file["h"].values().tolist() == entries.tolist()
        assert file["h"].axes[ndim - 1].edges().tolist() == edges[-1].tolist()

    root_file = ROOT.TFile.Open(str(path))
    hist = root_file.Get("h")
    assert hist.GetDimension() == ndim
    assert hist.GetEntries() == entries.sum()
    root_axes = [hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()][:ndim]
    for root_axis, axis_edges in zip(root_axes, edges, strict=True):
        assert root_axis.IsVariableBinSize()
        lows = [root_axis.GetBinLowEdge(i + 1) for i in range(len(axis_edges))]
        assert lows == axis_edges.tolist()
    flow = np.pad(entries, 1).T.reshape(-1).tolist()
    assert root_cells(hist, np.array(shape) + 2) == flow
    root_file.Close()


def test_mapping_becomes_rntuple(tmp_path):
    x = np.array([1.5, 2.5, 3.5])
    path = write(tmp_path, types.MappingProxyType({"x": x}), name="ntuple")
    with uproot.open(path) as file:
        assert file["ntuple"].classname == "ROOT::RNTuple"
        assert file["ntuple"].arrays()["x"].tolist() == x.tolist()


@pytest.mark.parametrize(
    "dtype, classname",
    [
        (np.int8, "TArrayC"),
        (np.int16, "TArrayS"),
        (np.int32, "TArrayI"),
        (np.int64, "TArrayL"),
        (np.float32, "TArrayF"),
        (np.float64, "TArrayD"),
    ],
)
def test_to_TArray(dtype, classname):
    tarray = to_TArray(np.array([1, 2, 3], dtype=dtype))
    assert tarray.classname == classname
    assert tarray.member("fN") == 3
    assert list(tarray) == [1, 2, 3]
    assert tarray._data.dtype == np.dtype(dtype).newbyteorder(">")


def test_to_TArray_invalid():
    with pytest.raises(ValueError, match="must be one-dimensional"):
        to_TArray(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="signed integer or floating-point type"):
        to_TArray(np.zeros(2, dtype=np.uint8))


@pytest.mark.parametrize(
    "function, name", [(to_TList, "TList"), (to_THashList, "THashList")]
)
def test_lists_only_hold_models(function, name):
    with pytest.raises(TypeError, match=f"to convert to {name} must only contain ROOT"):
        function(["not a Model"])


def test_list_of_unwritable_objects_is_not_writable():
    assert to_TList([to_TObjString("writable")])._deeply_writable
    assert not to_TList([uproot.models.TObject.Model_TObject.empty()])._deeply_writable


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
    ROOT = pytest.importorskip("ROOT")

    data = np.arange(4**ndim, dtype=dtype) % 100
    hist = TO_TH[ndim](**hist_args(ndim, to_TArray(data) if wrapped else data))
    assert hist.classname == f"TH{ndim}{letter}"

    path = write(tmp_path, hist)
    with uproot.open(path) as file:
        assert file["h"].classname == f"TH{ndim}{letter}"
        assert file["h"].values(flow=True).T.reshape(-1).tolist() == data.tolist()

    root_file = ROOT.TFile.Open(str(path))
    root_hist = root_file.Get("h")
    assert root_hist.ClassName() == f"TH{ndim}{letter}"
    assert root_cells(root_hist, data.shape) == data.tolist()
    root_file.Close()


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_histogram_type_of_64bit_integers(ndim):
    data = np.arange(4**ndim, dtype=np.int64)
    with pytest.raises(TypeError, match=r"no TH1\* subclasses correspond to TArrayL"):
        TO_TH[ndim](**hist_args(ndim, data))


def test_histogram_arguments_can_be_models(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    fXbins = to_TArray(np.array([0.0, 0.5, 2.0]))
    hist = to_TH1x(
        **{
            **hist_args(
                1, to_TArray(np.array([0.0, 1.0, 2.0, 0.0])), to_TArray(np.ones(4))
            ),
            "fXaxis": to_TAxis(
                "xaxis",
                "",
                2,
                0.0,
                2.0,
                fXbins=fXbins,
                fTimeFormat=to_TString("%d/%m/%y"),
            ),
        },
        fContour=to_TArray(np.array([0.5, 1.5])),
        fOption=to_TString("E"),
        fFunctions=to_TList([]),
        fBuffer=np.array([], dtype=np.float64),
    )

    root_file = ROOT.TFile.Open(str(write(tmp_path, hist)))
    root_hist = root_file.Get("h")
    assert root_hist.GetOption() == "E"
    assert root_hist.GetContour() == 2
    assert root_hist.GetContourLevel(1) == 1.5
    assert root_hist.GetSumw2N() == 4
    assert root_hist.GetListOfFunctions().GetSize() == 0
    assert root_hist.GetBinContent(2) == 2
    assert root_hist.GetXaxis().GetBinUpEdge(1) == 0.5
    assert root_hist.GetXaxis().GetTimeFormat() == "%d/%m/%y"
    root_file.Close()


def test_histogram_arguments_as_ndarrays_and_lists(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    function = uproot.writing.identify.to_TObjString("not actually a function")
    hist = to_TH1x(
        **hist_args(1, np.array([0.0, 1.0, 2.0, 0.0]), np.ones(4)),
        fContour=np.array([0.5, 1.5]),
        fFunctions=[function],
    )

    root_file = ROOT.TFile.Open(str(write(tmp_path, hist)))
    root_hist = root_file.Get("h")
    assert root_hist.GetContourLevel(0) == 0.5
    assert root_hist.GetSumw2N() == 4
    assert str(root_hist.GetListOfFunctions().At(0)) == "not actually a function"
    root_file.Close()


@pytest.mark.parametrize("wrapped", [False, True], ids=["ndarray", "TArray"])
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"fSumw2": np.ones(4, dtype=np.float32)},
            "fSumw2 must be an array of float64",
        ),
        (
            {"fContour": np.ones(4, dtype=np.float32)},
            "fContour must be an array of float64",
        ),
    ],
    ids=["fSumw2", "fContour"],
)
def test_histogram_arguments_must_be_float64(wrapped, kwargs, message):
    args = hist_args(1, np.ones(4))
    args.update(
        {k: to_TArray(v) if wrapped else v for k, v in kwargs.items()},
    )
    with pytest.raises(TypeError, match=message):
        to_TH1x(**args)


TO_TPROFILE = {1: to_TProfile, 2: to_TProfile2D, 3: to_TProfile3D}
PROFILE_NAME = {1: "TProfile", 2: "TProfile2D", 3: "TProfile3D"}


def profile_args(ndim, data, fBinEntries, fBinSumw2):
    out = hist_args(ndim, data, fSumw2=np.full(4**ndim, 4.0))
    if ndim == 1:
        out.update({"fTsumwy": 1.0, "fTsumwy2": 1.0})
    if ndim == 2:
        out.update({"fTsumwz": 1.0, "fTsumwz2": 1.0})
    if ndim == 3:
        out.update({"fTsumwt": 1.0, "fTsumwt2": 1.0})
    out.update({"fBinEntries": fBinEntries, "fBinSumw2": fBinSumw2})
    return out


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "field, message",
    [
        ("data", "{} requires an array of float64"),
        ("fBinEntries", "fBinEntries must be an array of float64"),
        ("fBinSumw2", "fBinSumw2 must be an array of float64"),
    ],
)
def test_profile_arrays_must_be_float64(ndim, field, message):
    arrays = {
        "data": np.ones(4**ndim),
        "fBinEntries": np.ones(4**ndim),
        "fBinSumw2": np.ones(4**ndim),
    }
    arrays[field] = np.ones(4**ndim, dtype=np.float32)
    with pytest.raises(TypeError, match=message.format(PROFILE_NAME[ndim])):
        TO_TPROFILE[ndim](**profile_args(ndim, **arrays))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_profile_arrays_can_be_models(tmp_path, ndim):
    ROOT = pytest.importorskip("ROOT")

    fBinEntries = np.zeros(4**ndim)
    fBinEntries[-2] = 2.0
    data = np.zeros(4**ndim)
    data[-2] = 6.0

    profile = TO_TPROFILE[ndim](
        **profile_args(ndim, data, to_TArray(fBinEntries), to_TArray(fBinEntries))
    )
    assert profile.classname == PROFILE_NAME[ndim]

    root_file = ROOT.TFile.Open(str(write(tmp_path, profile)))
    root_profile = root_file.Get("h")
    assert root_profile.ClassName() == PROFILE_NAME[ndim]
    assert root_profile.GetBinEntries(4**ndim - 2) == 2
    assert root_profile.GetBinContent(4**ndim - 2) == 3
    assert root_profile.GetBinEntries(1) == 0
    root_file.Close()


@pytest.mark.parametrize(
    "edges",
    [
        ([0.0, 1.0, 2.0],),
        ([0.0, 1.0, 2.0], [10.0, 20.0, 40.0, 80.0]),
        ([0.0, 1.0, 2.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 500.0]),
    ],
    ids=["TH1", "TH2", "TH3"],
)
def test_numpy_histogram_axes(edges, tmp_path):
    entries = np.ones([len(x) - 1 for x in edges])
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["h"] = (entries, *[np.array(x) for x in edges])

    with uproot.open(path) as file:
        histogram = file["h"]
        assert histogram.values().tolist() == entries.tolist()
        for name, expected in zip(["fXaxis", "fYaxis", "fZaxis"], edges, strict=False):
            axis = histogram.member(name)
            assert axis.member("fNbins") == len(expected) - 1
            assert axis.member("fXmin") == expected[0]
            assert axis.member("fXmax") == expected[-1]
            assert axis.edges().tolist() == expected
