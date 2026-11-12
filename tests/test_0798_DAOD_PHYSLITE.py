# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


@pytest.mark.parametrize("is_forth", [False, True])
def test_AnalysisJetsAuxDyn_GhostTrack(is_forth):
    expected_type = '2 * var * var * struct[{m_persKey: uint32, m_persIndex: uint32}, parameters={"__record__": "ElementLink<DataVector<xAOD::IParticle>>"}]'
    expected_data = [
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 490246363, "m_persIndex": 5},
        {"m_persKey": 0, "m_persIndex": 0},
        {"m_persKey": 490246363, "m_persIndex": 3},
        {"m_persKey": 490246363, "m_persIndex": 0},
        {"m_persKey": 490246363, "m_persIndex": 4},
    ]

    with uproot.open(skhep_testdata.data_path("uproot-issue-798.root")) as file:
        branch = file["CollectionTree"]["AnalysisJetsAuxDyn.GhostTrack"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        array = branch.array(interp, library="ak", entry_stop=2)
        assert str(array.type) == expected_type
        assert array[0, 0].tolist() == expected_data
        assert array.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_TruthBosonAuxDyn_childLinks(is_forth):
    expected_type = '2 * var * var * struct[{m_persKey: uint32, m_persIndex: uint32}, parameters={"__record__": "ElementLink<DataVector<xAOD::TruthParticle_v1>>"}]'
    expected_data = [
        [],
        [
            {"m_persKey": 375408000, "m_persIndex": 0},
            {"m_persKey": 13267281, "m_persIndex": 0},
            {"m_persKey": 368360608, "m_persIndex": 0},
        ],
    ]

    with uproot.open(skhep_testdata.data_path("uproot-issue-798.root")) as file:
        branch = file["CollectionTree"]["TruthBosonAuxDyn.childLinks"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        array = branch.array(interp, library="ak", entry_stop=2)
        assert str(array.type) == expected_type
        assert array[0].tolist() == expected_data
        assert array.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_TruthPhotonsAuxDyn_parentLinks(is_forth):
    expected_type = '2 * var * var * struct[{m_persKey: uint32, m_persIndex: uint32}, parameters={"__record__": "ElementLink<DataVector<xAOD::TruthParticle_v1>>"}]'
    expected_data = [
        [{"m_persKey": 614719239, "m_persIndex": 1}],
        [],
        [],
        [],
        [{"m_persKey": 779635413, "m_persIndex": 2}],
        [{"m_persKey": 779635413, "m_persIndex": 3}],
        [{"m_persKey": 779635413, "m_persIndex": 3}],
    ]

    with uproot.open(skhep_testdata.data_path("uproot-issue-798.root")) as file:
        branch = file["CollectionTree"]["TruthPhotonsAuxDyn.parentLinks"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        array = branch.array(interp, library="ak", entry_stop=2)
        assert str(array.type) == expected_type
        assert array[0].tolist() == expected_data
        assert array.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_TruthTopAuxDyn_parentLinks(is_forth):
    expected_type = '2 * var * var * struct[{m_persKey: uint32, m_persIndex: uint32}, parameters={"__record__": "ElementLink<DataVector<xAOD::TruthParticle_v1>>"}]'
    expected_data = [
        [],
        [],
        [{"m_persKey": 660928181, "m_persIndex": 0}],
        [{"m_persKey": 660928181, "m_persIndex": 1}],
    ]

    with uproot.open(skhep_testdata.data_path("uproot-issue-798.root")) as file:
        branch = file["CollectionTree"]["TruthTopAuxDyn.parentLinks"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        array = branch.array(interp, library="ak", entry_stop=2)
        assert str(array.type) == expected_type
        assert array[0].tolist() == expected_data
        assert array.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None
