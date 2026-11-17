# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import importlib.util
import sys
import warnings

import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")

needs_awkward_pandas = pytest.mark.skipif(
    importlib.util.find_spec("awkward_pandas") is None,
    reason="awkward-pandas is not installed",
)


@pytest.fixture
def events():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as tree:
        yield tree


@needs_awkward_pandas
def test_jagged_data_warns(events):
    with pytest.warns(FutureWarning, match="Uproot 5.8 will stop using awkward-pandas"):
        df = events.arrays(["Muon_Px", "NMuon"], library="pd")
    assert df["Muon_Px"].dtype == "awkward"

    with pytest.warns(FutureWarning, match="install akimbo"):
        events["Muon_Px"].array(library="pd")


def test_flat_data_does_not_warn(events):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = events.arrays(["NMuon", "MET_px"], library="pd")
    assert str(df["NMuon"].dtype) == "int32"


@needs_awkward_pandas
def test_shown_once_with_default_filters(events):
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        # the default filters use the module's registry, which other tests may
        # have filled in already
        uproot.interpretation.library.__dict__.pop("__warningregistry__", None)
        events.arrays(["Muon_Px", "Muon_Py"], library="pd")
        for arrays in events.iterate(["Jet_Px"], step_size=1000, library="pd"):
            pass
    assert [type(w.message) for w in caught] == [FutureWarning]


def test_missing_awkward_pandas_error(events, monkeypatch):
    monkeypatch.setitem(sys.modules, "awkward_pandas", None)
    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            ModuleNotFoundError, match="Uproot 5.8 will not need awkward-pandas"
        ),
    ):
        events.arrays(["Muon_Px"], library="pd")
