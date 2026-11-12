# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")
pytest.importorskip("dask_awkward")


def test():
    nMuon_doc = "slimmedMuons after basic selection (pt > 15 || (pt > 3 && (passed('CutBasedIdLoose') || passed('SoftCutBasedId') || passed('SoftMvaId') || passed('CutBasedIdGlobalHighPt') || passed('CutBasedIdTrkHighPt'))))"

    with uproot.open(skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root"))[
        "Events/nMuon"
    ] as branch:
        assert branch.title == nMuon_doc

        assert (
            str(branch.array(ak_add_doc=True).type)
            == f'200 * uint32[parameters={{"__doc__": "{nMuon_doc}"}}]'
        )

    lazy = uproot.dask(
        skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events",
        ak_add_doc=True,
    )

    assert (
        str(lazy["nMuon"].type)
        == f'?? * uint32[parameters={{"__doc__": "{nMuon_doc}"}}]'
    )

    assert (
        str(lazy["nMuon"].compute().type)
        == f'200 * uint32[parameters={{"__doc__": "{nMuon_doc}"}}]'
    )
