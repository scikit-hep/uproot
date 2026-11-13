# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import subprocess
import sys

import numpy as np
import pytest

import uproot

ROOT = pytest.importorskip("ROOT")


DIMENSIONS = {
    "TH1D": 1,
    "TH1F": 1,
    "TH1I": 1,
    "TH2D": 2,
    "TH3D": 3,
    "TProfile": 1,
    "TProfile2D": 2,
    "TProfile3D": 3,
}


def _make(kind, name, weighted):
    nd = DIMENSIONS[kind]
    is_profile = kind.startswith("TProfile")
    bins = [(5, 0.0, 1.0), (4, 0.0, 1.0), (3, 0.0, 1.0)][:nd]
    obj = getattr(ROOT, kind)(name, f"title of {name}", *sum(bins, ()))
    if kind == "TH1D" and not weighted:
        for i, label in enumerate("abcde"):
            obj.GetXaxis().SetBinLabel(i + 1, label)
    if weighted:
        obj.Sumw2()

    rng = np.random.default_rng(12345)
    for _ in range(200):
        args = list(rng.uniform(-0.2, 1.2, nd + is_profile))
        if weighted:
            args.append(rng.uniform(0.5, 2.0))
        obj.Fill(*args)
    return obj


def _assert_same(expected, actual):
    assert actual.ClassName() == expected.ClassName()
    assert actual.GetName() == expected.GetName()
    assert actual.GetTitle() == expected.GetTitle()
    assert actual.GetEntries() == expected.GetEntries()
    assert actual.GetSumw2N() == expected.GetSumw2N()

    stats_expected, stats_actual = np.zeros(20), np.zeros(20)
    expected.GetStats(stats_expected)
    actual.GetStats(stats_actual)
    assert stats_actual.tolist() == stats_expected.tolist()

    methods = ["GetBinContent", "GetBinError"]
    if hasattr(expected, "GetBinEntries"):
        methods += ["GetBinEntries", "GetBinEffectiveEntries"]
        assert actual.GetErrorOption() == expected.GetErrorOption()
    for method in methods:
        assert [getattr(actual, method)(i) for i in range(actual.GetNcells())] == [
            getattr(expected, method)(i) for i in range(expected.GetNcells())
        ]

    for axis in "XYZ":
        axis_expected = getattr(expected, f"Get{axis}axis")()
        axis_actual = getattr(actual, f"Get{axis}axis")()
        assert axis_actual.GetNbins() == axis_expected.GetNbins()
        assert axis_actual.GetXmin() == axis_expected.GetXmin()
        assert axis_actual.GetXmax() == axis_expected.GetXmax()
        assert [
            axis_actual.GetBinLabel(i) for i in range(1, axis_actual.GetNbins() + 1)
        ] == [
            axis_expected.GetBinLabel(i) for i in range(1, axis_expected.GetNbins() + 1)
        ]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", DIMENSIONS)
def test_read_from_root_file(tmp_path, kind, weighted):
    filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(filename, "RECREATE")
    _make(kind, "obj", weighted).Write()
    fout.Close()

    fin = ROOT.TFile(filename)
    with uproot.open(filename) as file:
        _assert_same(fin.Get("obj"), uproot.to_pyroot(file["obj"]))
    fin.Close()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", DIMENSIONS)
def test_from_pyroot(tmp_path, kind, weighted):
    obj = _make(kind, f"obj_1768_from_pyroot_{kind}_{weighted}", weighted)
    obj.SetDirectory(ROOT.nullptr)
    converted = uproot.from_pyroot(obj)
    _assert_same(obj, uproot.to_pyroot(converted))

    filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(filename) as file:
        file[obj.GetName()] = converted
    fin = ROOT.TFile(filename)
    _assert_same(obj, fin.Get(obj.GetName()))
    fin.Close()


@pytest.mark.parametrize(
    ("ndim", "storage"),
    [
        (ndim, storage)
        for ndim in (1, 2, 3)
        for storage in ("Double", "Weight", "Mean", "WeightedMean")
        # Uproot can't convert 2D or 3D profiles from hist yet
        if ndim == 1 or "Mean" not in storage
    ],
)
def test_converted_from_hist(tmp_path, ndim, storage):
    hist = pytest.importorskip("hist")

    axes = [hist.axis.Regular(5, 0, 1), hist.axis.Variable([0, 0.2, 0.7, 1])]
    axes.append(hist.axis.Regular(3, 0, 1))
    h = hist.Hist(*axes[:ndim], storage=getattr(hist.storage, storage)())
    rng = np.random.default_rng(12345)
    kwargs = {}
    if storage in ("Weight", "WeightedMean"):
        kwargs["weight"] = rng.uniform(0.5, 2.0, 200)
    if storage in ("Mean", "WeightedMean"):
        kwargs["sample"] = rng.uniform(-1, 1, 200)
    h.fill(*rng.uniform(-0.2, 1.2, (ndim, 200)), **kwargs)

    filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(filename) as file:
        file["obj"] = h

    fin = ROOT.TFile(filename)
    _assert_same(fin.Get("obj"), uproot.to_pyroot(h, "obj"))
    fin.Close()


def test_converted_from_numpy():
    counts, edges = np.histogram([0.1, 0.5, 0.5, 0.9], bins=4, range=(0, 1))
    h = uproot.to_pyroot((counts, edges), "h_1768_numpy")
    assert [h.GetBinContent(i) for i in range(1, 5)] == counts.tolist()
    assert h.GetXaxis().GetXmax() == 1


GRAPH_ERRORS = {
    "TGraph": [],
    "TGraphErrors": ["x_errors", "y_errors"],
    "TGraphAsymmErrors": [
        "x_errors_low",
        "x_errors_high",
        "y_errors_low",
        "y_errors_high",
    ],
}
GRAPH_GETTERS = {
    "x": "GetX",
    "y": "GetY",
    "x_errors": "GetEX",
    "y_errors": "GetEY",
    "x_errors_low": "GetEXlow",
    "x_errors_high": "GetEXhigh",
    "y_errors_low": "GetEYlow",
    "y_errors_high": "GetEYhigh",
}


def _assert_same_graph(expected, actual):
    assert actual.ClassName() == expected.ClassName()
    assert actual.GetName() == expected.GetName()
    assert actual.GetTitle() == expected.GetTitle()
    assert actual.GetMinimum() == expected.GetMinimum()
    assert actual.GetMaximum() == expected.GetMaximum()
    n = expected.GetN()
    assert actual.GetN() == n
    for method in GRAPH_GETTERS.values():
        values_expected = getattr(expected, method)()
        values_actual = getattr(actual, method)()
        if values_expected:
            assert list(values_actual)[:n] == list(values_expected)[:n]
        else:
            assert not values_actual


@pytest.mark.parametrize("kind", GRAPH_ERRORS)
def test_graph(tmp_path, kind):
    rng = np.random.default_rng(12345)
    columns = {name: rng.uniform(0, 1, 7) for name in ["x", "y", *GRAPH_ERRORS[kind]]}

    root_filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(root_filename, "RECREATE")
    graph = getattr(ROOT, kind)(7, *columns.values())
    graph.SetNameTitle("graph", "title of graph")
    graph.Write()
    fout.Close()

    uproot_filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(uproot_filename) as file:
        file["graph"] = uproot.as_TGraph(columns, title="title of graph")

    fin = ROOT.TFile(uproot_filename)
    graph = fin.Get("graph")
    for column, values in columns.items():
        assert list(getattr(graph, GRAPH_GETTERS[column])())[:7] == values.tolist()
    fin.Close()

    for filename in [root_filename, uproot_filename]:
        fin = ROOT.TFile(filename)
        with uproot.open(filename) as file:
            _assert_same_graph(fin.Get("graph"), uproot.to_pyroot(file["graph"]))
        fin.Close()

    fin = ROOT.TFile(uproot_filename)
    _assert_same_graph(
        fin.Get("graph"),
        uproot.to_pyroot(uproot.as_TGraph(columns, title="title of graph"), "graph"),
    )
    fin.Close()


def test_streamer_info_registration():
    # A TMessage carries no StreamerInfo, so ROOT can't read the TGraph version
    # that Uproot writes unless to_pyroot registers it. Check this in a new
    # process, since opening any file written by Uproot would also register it.
    script = (
        "import uproot\n"
        "data = {'x': [1.0, 2.0], 'y': [3.0, 4.0]}\n"
        "graph = uproot.to_pyroot(uproot.as_TGraph(data), 'g')\n"
        "print(graph.GetN(), graph.GetName(), list(graph.GetY()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.stdout.splitlines()[-1] == "2 g [3.0, 4.0]"


def test_from_pyroot_nested_classes():
    # The StreamerInfo for a class is fetched from ROOT with the first object of
    # that class, so it must be fetched again for a later one that contains
    # other classes. Check this in a new process, where no other test has
    # fetched them.
    script = (
        "import ROOT, uproot\n"
        "ROOT.TH1.AddDirectory(False)\n"
        "uproot.from_pyroot(ROOT.TH1D('plain', '', 3, 0, 3))\n"
        "h = ROOT.TH1D('with_function', '', 3, 0, 3)\n"
        "f = ROOT.TF1('f', 'x', 0, 3)\n"
        "ROOT.SetOwnership(f, False)\n"
        "h.GetListOfFunctions().Add(f)\n"
        "print(type(uproot.from_pyroot(h).member('fFunctions')[0]).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.stdout.splitlines()[-1].startswith("Model_TF1_v")


def test_tobjstring(tmp_path):
    root_filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(root_filename, "RECREATE")
    fout.WriteObject(ROOT.TObjString("some text"), "string")
    fout.Close()

    with uproot.open(root_filename) as file:
        string = file["string"]
        assert not string.is_memberwise
        assert string == string.member("fString") == "some text"
        assert uproot.to_pyroot(string).GetString() == "some text"
        assert uproot.from_pyroot(ROOT.TObjString("other")).member("fString") == "other"

        uproot_filename = os.path.join(tmp_path, "uproot_written.root")
        with uproot.recreate(uproot_filename) as fout:
            fout["string"] = string

    fin = ROOT.TFile(uproot_filename)
    assert fin.Get("string").GetString() == "some text"
    fin.Close()


@pytest.mark.parametrize("kind", ["TList", "TObjArray"])
def test_write_list(tmp_path, kind):
    hists = [ROOT.TH1D(f"h_1768_{kind}_{i}", "", 3, 0, 3) for i in range(3)]
    collection = getattr(ROOT, kind)()
    # a TObjArray can have empty slots
    slots = [0, 1, 2] if kind == "TList" else [0, 2, 3]
    for i, h in enumerate(hists):
        h.SetDirectory(ROOT.nullptr)
        h.Fill(i)
        if kind == "TList":
            collection.Add(h)
        else:
            collection.AddAtAndExpand(h, slots[i])

    root_filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(root_filename, "RECREATE")
    fout.WriteObject(collection, "collection", "SingleKey")
    fout.Close()

    # its items are converted to the versions that Uproot writes
    uproot_filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.open(root_filename) as fin, uproot.recreate(uproot_filename) as fout:
        fout["collection"] = fin["collection"]

    fin = ROOT.TFile(uproot_filename)
    collection = fin.Get("collection")
    assert collection.GetEntries() == 3
    for i, h in enumerate(hists):
        _assert_same(h, collection.At(slots[i]))
    fin.Close()


def test_pyroot_errors_and_copies():
    h = ROOT.TH1D("h_1768_copy", "", 3, 0, 3)
    h.SetDirectory(ROOT.nullptr)
    h.Fill(1.5)
    copy = uproot.to_pyroot(h, "h_1768_copied")
    assert copy.GetName() == "h_1768_copied"
    assert copy.GetBinContent(2) == 1

    tree = ROOT.TTree("t_1768", "")
    tree.SetDirectory(ROOT.nullptr)
    with pytest.raises(TypeError, match="TTrees can't be converted"):
        uproot.from_pyroot(tree)

    ROOT.gInterpreter.Declare(
        "class _uproot_1768_interpreted : public TObject {"
        " ClassDef(_uproot_1768_interpreted, 1) };"
    )
    with pytest.raises(TypeError, match="interpreted class"):
        uproot.from_pyroot(ROOT._uproot_1768_interpreted())


@pytest.mark.parametrize("compressed", [False, True])
def test_write_pyroot(tmp_path, compressed):
    # PyROOT objects are written as ROOT serializes them, but with references to
    # other objects within them (labels, list items, ...) relative to their keys
    labeled = _make("TH1D", f"h_1768_write_labeled_{compressed}", False)
    labeled.SetDirectory(ROOT.nullptr)
    labeled.SetStats(False)
    labeled.SetOption("hist")
    labeled.GetXaxis().SetRange(2, 4)
    labeled.GetXaxis().CenterTitle()
    hists = [_make("TH2D", f"h_1768_write_{compressed}_{i}", True) for i in range(3)]
    collection = ROOT.TList()
    for h in hists:
        h.SetDirectory(ROOT.nullptr)
        collection.Add(h)
    graph = ROOT.TGraph(2, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    graph.SetNameTitle("graph", "title;x title;y title")
    graph.GetHistogram()
    graph.SetEditable(False)
    poly = ROOT.TH2Poly(f"poly_1768_write_{compressed}", "", 0, 2, 0, 2)
    poly.SetDirectory(ROOT.nullptr)
    for a in (0.0, 1.0):
        x, y = np.array([a, a + 1, a + 1, a]), np.array([a, a, a + 1, a + 1])
        poly.AddBin(ROOT.TGraph(4, x, y))
    poly.Fill(1.5, 1.5, 3)

    filename = os.path.join(tmp_path, "uproot_written.root")
    compression = uproot.ZLIB(1) if compressed else None
    with uproot.recreate(filename, compression=compression) as fout:
        fout["labeled"] = labeled
        fout["collection"] = collection
        fout["graph"] = graph
        fout["poly"] = poly

    fin = ROOT.TFile(filename)
    out = fin.Get("labeled")
    labeled.SetName("labeled")  # as written
    _assert_same(labeled, out)
    assert out.TestBit(ROOT.TH1.kNoStats)
    assert out.GetOption() == "hist"
    assert (out.GetXaxis().GetFirst(), out.GetXaxis().GetLast()) == (2, 4)
    assert out.GetXaxis().GetCenterTitle()
    assert fin.Get("collection").GetEntries() == 3
    for i, h in enumerate(hists):
        _assert_same(h, fin.Get("collection").At(i))
    assert fin.Get("graph").GetXaxis().GetTitle() == "x title"
    assert not fin.Get("graph").GetEditable()
    assert fin.Get("poly").GetBinContent(2) == 3
    fin.Close()

    # and Uproot reads them
    with uproot.open(filename) as fin:
        assert fin["labeled"].axis().labels() == list("abcde")
        assert len(fin["collection"]) == 3


def test_write_pyroot_big_keys(tmp_path, monkeypatch):
    # keys after 2 GB have 8-byte pointers, so they're bigger; pretend that
    # starts after 3 kB, so that most of these keys are big
    monkeypatch.setattr(uproot.const, "kStartBigFile", 3000)
    hists = [_make("TH1D", f"h_1768_big_{i}", False) for i in range(10)]
    filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(filename) as fout:
        for h in hists:
            h.SetDirectory(ROOT.nullptr)
            fout[h.GetName()] = h

    fin = ROOT.TFile(filename)
    for h in hists:
        _assert_same(h, fin.Get(h.GetName()))
    fin.Close()


def test_write_pyroot_streamer_info(tmp_path):
    # The file gets the StreamerInfo of everything in a PyROOT object, even if
    # an earlier object of the same class didn't contain all of it. Check this
    # in a new process, where no other test has converted these classes.
    filename = os.path.join(tmp_path, "uproot_written.root")
    script = (
        "import sys, ROOT, uproot\n"
        "ROOT.TH1.AddDirectory(False)\n"
        "uproot.from_pyroot(ROOT.TH1D('plain', '', 3, 0, 3))\n"
        "h = ROOT.TH1D('with_function', '', 3, 0, 3)\n"
        "f = ROOT.TF1('f', 'x', 0, 3)\n"
        "ROOT.SetOwnership(f, False)\n"
        "h.GetListOfFunctions().Add(f)\n"
        "with uproot.recreate(sys.argv[1]) as fout:\n"
        "    fout['h'] = h\n"
    )
    subprocess.run(
        [sys.executable, "-c", script, filename],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    with uproot.open(filename) as fin:
        assert "TF1" in fin.file.streamers
        assert "TH1D" in fin.file.streamers


def test_write_pyroot_side_effects(tmp_path):
    # writing a PyROOT object doesn't change ROOT's current directory
    root_file = ROOT.TFile(os.path.join(tmp_path, "root_written.root"), "RECREATE")
    h = ROOT.TH1D("h_1768_side_effects", "", 3, 0, 3)
    h.SetDirectory(ROOT.nullptr)
    with uproot.recreate(os.path.join(tmp_path, "uproot_written.root")) as fout:
        fout["h"] = h
    assert ROOT.gDirectory.GetName() == root_file.GetName()
    root_file.Close()

    # and it's written as ROOT would write it, although getting its StreamerInfo
    # also writes it: a TCanvas writes new colors only the first time. This is in a
    # new process, so that the color is only known from the file.
    filename = os.path.join(tmp_path, "canvas.root")
    script = (
        "import sys, ROOT, uproot\n"
        "ROOT.gROOT.SetBatch(True)\n"
        "color = ROOT.TColor(1768, 0.1, 0.7, 0.3, 'color_1768', 1.0)\n"
        "canvas = ROOT.TCanvas('canvas_1768', '')\n"
        "with uproot.recreate(sys.argv[1]) as fout:\n"
        "    fout['canvas'] = canvas\n"
    )
    subprocess.run(
        [sys.executable, "-c", script, filename],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert not ROOT.gROOT.GetColor(1768)
    fin = ROOT.TFile(filename)
    fin.Get("canvas")
    color = ROOT.gROOT.GetColor(1768)
    assert (color.GetRed(), color.GetGreen(), color.GetBlue()) == pytest.approx(
        (0.1, 0.7, 0.3)
    )


def test_write_bits_and_strings(tmp_path):
    # TObject bits (e.g. SetStats(0), axis ranges) and TString members (e.g.
    # the draw option) of objects read from files are kept when written
    h = ROOT.TH1D("h_1768_bits", "", 5, 0, 5)
    h.SetDirectory(ROOT.nullptr)
    h.Fill(1.5)
    h.SetStats(False)
    h.SetOption("hist")
    h.GetXaxis().SetRange(2, 4)
    h.GetXaxis().CenterTitle()
    h.GetXaxis().SetTimeFormat("%H:%M")

    root_filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(root_filename, "RECREATE")
    h.Write()
    fout.Close()

    uproot_filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.open(root_filename) as fin:
        with uproot.recreate(uproot_filename) as fout:
            fout[h.GetName()] = fin[h.GetName()]
        converted = uproot.to_pyroot(fin[h.GetName()])

    fin = ROOT.TFile(uproot_filename)
    for out in [fin.Get(h.GetName()), converted]:
        assert out.TestBit(ROOT.TH1.kNoStats)
        assert out.GetOption() == "hist"
        assert (out.GetXaxis().GetFirst(), out.GetXaxis().GetLast()) == (2, 4)
        assert out.GetXaxis().GetCenterTitle()
        assert out.GetXaxis().GetTimeFormat() == h.GetXaxis().GetTimeFormat()
    fin.Close()
