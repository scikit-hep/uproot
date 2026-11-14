import skhep_testdata as skhtd
import uproot
import pytest
import numpy as np

ak = pytest.importorskip("awkward")


@pytest.fixture(scope="module")
def physlite_file():
    """Fixture that returns an open EventData rntp from the test ROOT file."""
    file_path = skhtd.data_path("uproot-physlite-rntuple_v1-0-0-0.root")
    with uproot.open(file_path) as f:
        assert "EventData" in f, "'EventData' RNTuple not found in file"
        assert (
            f["EventData"].classname == "ROOT::RNTuple"
        ), "EventData is not an RNTuple"
        yield f["EventData"]  # keeps file open


def test_analysis_muons_kinematics(physlite_file):
    """Test that kinematic variables of AnalysisMuons can be read and match expected length."""
    cols = [
        "AnalysisMuonsAuxDyn:pt",
        "AnalysisMuonsAuxDyn:eta",
        "AnalysisMuonsAuxDyn:phi",
    ]

    # Check if cols are in rntp
    arrays = {}
    for col in cols:
        assert col in physlite_file.keys(), f"Column '{col}' not found"
        arrays[col] = physlite_file[col].array()

    # Check same structure, number of total muons, and values
    n_expected_muons = 88
    # Expected mean values for pt, eta and phi rounded to 2 decimals
    expected_means = [28217.96, -0.21, -0.18]

    for i, (col, arr) in enumerate(arrays.items()):
        assert isinstance(arr, ak.Array), f"{col} is not an Awkward Array"
        assert (
            len(ak.flatten(arr)) == n_expected_muons
        ), f"{col} does not match expected muon count"
        assert (
            round(ak.mean(arr), 2) == expected_means[i]
        ), f"{col} mean value does not match the expected one"


def test_event_info(physlite_file):
    """Test that eventInfo variables can be read and match expected first event."""
    cols = [
        "EventInfoAuxDyn:eventNumber",
        "EventInfoAuxDyn:averageInteractionsPerCrossing",
        "EventInfoAuxDyn:lumiBlock",
    ]

    first_event = {}
    for col in cols:
        assert col in physlite_file.keys(), f"Column '{col}' not found"
        first_event[col] = physlite_file[col].array()[0]

    # Check first event values
    # expected event info values: event number, pile-up, lumiBlock
    expected_values = [293298001, 26.5, 1]

    for i, (col, value) in enumerate(first_event.items()):
        assert (
            value == expected_values[i]
        ), f"First event {col} doest not match the expected value"


def test_truth_muon_containers(physlite_file):
    """Test that truth muon variables can be read and match expected values."""
    cols = [
        "TruthMuons",  # AOD Container
        "TruthMuonsAuxDyn:pdgId",
        "TruthMuonsAuxDyn:m",
    ]

    # Check for columns in the RNTuple
    arrays = {}
    for col in cols:
        assert col in physlite_file.keys(), f"Column '{col}' not found"
        arrays[col] = physlite_file[col].array()

    # Check values
    mass_evt_0 = 105.7
    AOD_type = []  # C++ class definitions are ignored
    mu_pdgid = [13, -13]

    assert (
        arrays["TruthMuons"].fields == AOD_type
    ), f"TruthMuons fields have changed, {arrays['TruthMuons'].fields} instead of {AOD_type}"
    assert np.isclose(
        ak.flatten(arrays["TruthMuonsAuxDyn:m"])[0], mass_evt_0
    ), "Truth mass of first event does not match expected value"
    assert np.all(
        np.isin(ak.to_numpy(ak.flatten(arrays["TruthMuonsAuxDyn:pdgId"])), mu_pdgid)
    ), "Retrieved pdgids are not 13/-13"
    assert arrays["TruthMuonsAuxDyn:pdgId"][:6].tolist() == [
        [13],
        [-13],
        [-13],
        [13],
        [13],
        [],
    ]
