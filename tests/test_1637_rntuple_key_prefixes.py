# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot
import uproot.models.RNTuple

pytest.importorskip("awkward")


class KeyScan:
    """The check that ``_key_prefixes`` replaces: a scan over every key."""

    def __init__(self, keys):
        self.keys = list(keys)

    def __contains__(self, path):
        return any(key == path or key.startswith(f"{path}.") for key in self.keys)


def test_key_prefixes_matches_the_scan():
    keys = ["a", "b.c", "b.c.d", "e.f.g", "h.", "x..y", ".z", "Muon.pt", "Muons"]
    prefixes = uproot.models.RNTuple._key_prefixes(keys)
    scan = KeyScan(keys)

    paths = {None, "", "Muo", "Mu", "c", "d", "f.g", "b.c.d.e", "Muon.p"}
    for key in keys:
        paths.update(key[:i] for i in range(len(key) + 1))
    for path in paths:
        assert (path in prefixes) == (path in scan), path

    assert uproot.models.RNTuple._key_prefixes([]) == set()


def all_forms(ntuple):
    forms = [
        ntuple.to_akform(),
        ntuple.to_akform(ak_add_doc=True),
        ntuple.to_akform(filter_name="no_field_is_called_this"),
        ntuple.to_akform(filter_name=[field.name for field in ntuple.fields]),
    ]
    for key in ntuple.keys():
        forms.append(ntuple.to_akform(filter_name=key))
        forms.append(ntuple[key].to_akform())
        forms.append(ntuple[key].to_akform(ak_add_doc=True))
    return forms


@pytest.mark.parametrize(
    ("filename", "key"),
    [
        ("ntpl001_staff_rntuple_v1-0-0-0.root", "Staff"),
        ("Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root", "Events"),
        ("test_nested_structs_rntuple_v1-0-0-0.root", "ntuple"),
        ("test_stl_containers_rntuple_v1-0-0-0.root", "ntuple"),
        ("test_class_inheritance_rntuple_v1-0-0-1.root", "rntpl"),
        ("test_emptystruct_invalidvar_rntuple_v1-0-0-0.root", "ntuple"),
        ("test_atomic_bitset_rntuple_v1-0-0-0.root", "ntuple"),
        ("test_int_vfloat_tlv_vtlv_rntuple_v1-0-0-0.root", "ntuple"),
    ],
)
def test_forms_match_the_scan(filename, key, monkeypatch):
    with uproot.open(skhep_testdata.data_path(filename)) as file:
        ntuple = file[key]
        forms = all_forms(ntuple)
        monkeypatch.setattr(uproot.models.RNTuple, "_key_prefixes", KeyScan)
        assert all_forms(ntuple) == forms
