# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


@pytest.fixture(scope="module")
def delphes_tree():
    with uproot.open(skhep_testdata.data_path("uproot-delphes-pr442.root")) as f:
        yield f["Delphes"]


def test_read_delphes_np(delphes_tree):
    branch = delphes_tree["Jet/Jet.SoftDroppedP4[5]"]
    nparray = branch.array(library="np")
    assert nparray.shape == (25,)
    assert nparray[0].shape == (2, 5)
    assert nparray[0][0, 0].members["fE"] == 126.46277691787493

    branch = delphes_tree["GenJet04/GenJet04.Constituents"]
    array = branch.array(library="np")
    assert array.shape == (25,)
    assert isinstance(array[0][0], uproot.models.TRef.Model_TRefArray)
    assert len(array[0][0]) == 17
    assert array[0][0].refs[:5].tolist() == [2175, 2356, 2360, 2355, 2247]


def test_read_delphes_ak(delphes_tree):
    awkward = pytest.importorskip("awkward")

    branch = delphes_tree["Jet/Jet.SoftDroppedP4[5]"]
    array = branch.array(library="ak")
    assert array[0, 0, 0].fE == 126.46277691787493
    assert awkward.all(awkward.num(array, axis=2) == 5)

    form = branch.interpretation.awkward_form(branch.file)
    assert form.content.size == 5
    assert form.content.content.parameters == {"__record__": "TLorentzVector"}
    assert form.content.content.fields == array.layout.form.content.content.fields
    branch = delphes_tree["GenJet04/GenJet04.Constituents"]
    array = branch.array(library="ak")
    assert set(array.fields) == {"fSize", "fName", "refs"}
