#!/usr/bin/env python3

import awkward
import pytest
import skhep_testdata
import uproot

VECTOR_VECTOR_ELEMENTLINK_BRANCHES = [
    "AnalysisHLT_e12_lhloose_nod0_2mu10AuxDyn.TrigMatchedObjects",
    "AnalysisElectronsAuxDyn.caloClusterLinks",
    "AnalysisPhotonsAuxDyn.vertexLinks",
    "TruthMuonsAuxDyn.childLinks",
    "AnalysisElectronsAuxDyn.trackParticleLinks",
    "PrimaryVerticesAuxDyn.neutralParticleLinks",
    "AnalysisTauJetsAuxDyn.tauTrackLinks",
]


@pytest.mark.parametrize("key", VECTOR_VECTOR_ELEMENTLINK_BRANCHES)
def test_pickup_vector_vector_elementlink(key):
    with uproot.open(
        {skhep_testdata.data_path("uproot-issue-123a.root"): "CollectionTree"}
    ) as tree:
        branch = tree[key]
        assert branch.interpretation._complete_forth_code is not None
        form = branch.interpretation._form
        assert form["class"] == "ListOffsetArray"
        assert form["content"]["class"] == "ListOffsetArray"
        assert form["content"]["content"]["fields"] == ["m_persKey", "m_persIndex"]
        assert form["content"]["content"]["parameters"]["__record__"].startswith(
            "ElementLink<DataVector<"
        )


def test_consistent_library_np_vector_vector_elementlink():
    arrays_np = {}
    with uproot.open(
        {skhep_testdata.data_path("uproot-issue-123a.root"): "CollectionTree"}
    ) as tree:
        for key in VECTOR_VECTOR_ELEMENTLINK_BRANCHES:
            arrays_np[key] = tree[key].array(library="np", entry_stop=25)
    arrays_ak = {}
    with uproot.open(
        {skhep_testdata.data_path("uproot-issue-123a.root"): "CollectionTree"}
    ) as tree:
        for key in VECTOR_VECTOR_ELEMENTLINK_BRANCHES:
            arrays_ak[key] = tree[key].array(entry_stop=25)
    for key in arrays_np:
        array_ak = arrays_ak[key]
        array_np = uproot.interpretation.library._object_to_awkward_array(
            awkward, array_ak.layout.form.to_dict(), arrays_np[key]
        )
        for field in array_ak.fields:
            assert awkward.all(array_np[field] == array_ak[field])
