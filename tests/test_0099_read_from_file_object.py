# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot
import uproot.source.fsspec
import uproot.source.object


@pytest.mark.parametrize(
    "handler",
    [None, uproot.source.object.ObjectSource],
)
def test_read_from_file_like_object(handler):
    with open(skhep_testdata.data_path("uproot-Zmumu.root"), "rb") as f:
        assert uproot.open({f: "events"}, handler=handler)["px1"].array(library="np")[
            :10
        ].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
            -19.8623073126,
            -19.8623073126,
            -20.1773731496,
            71.1437106445,
            51.0504859191,
        ]
