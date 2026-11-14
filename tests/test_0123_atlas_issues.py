# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_version():
    assert uproot.classname_decode(
        uproot.classname_encode("xAOD::MissingETAuxAssociationMap_v2")
    ) == ("xAOD::MissingETAuxAssociationMap_v2", None)
    assert uproot.classname_decode(
        uproot.classname_encode("xAOD::MissingETAuxAssociationMap_v2", 9)
    ) == ("xAOD::MissingETAuxAssociationMap_v2", 9)
