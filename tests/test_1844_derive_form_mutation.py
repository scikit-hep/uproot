# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import copy

import pytest

import uproot
from uproot._awkwardforth import Node


def test_derive_form_fixed_size_array_of_lists():
    # AsArray with an inner shape (e.g. std::vector<float> fV[3] in a split
    # TClonesArray) puts ListOffsetArray -> RegularArray in a single node, whose
    # child is another list node.
    leaf = Node(
        "node3",
        form_details={
            "class": "NumpyArray",
            "primitive": "float32",
            "form_key": "node3",
        },
    )
    vector = Node(
        "node2",
        form_details={
            "class": "ListOffsetArray",
            "offsets": "i64",
            "parameters": {},
            "form_key": "node2",
        },
        children=[leaf],
    )
    array = Node(
        "node1",
        form_details={
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {"class": "RegularArray", "size": 3},
            "parameters": {},
            "form_key": "node1",
        },
        children=[vector],
    )

    original = copy.deepcopy(array.form_details)

    # derive_form is called once per entry until the form has no NULLs
    forms = [array.derive_form() for _ in range(3)]

    assert array.form_details == original
    assert forms[0] == forms[1] == forms[2]

    regular = forms[0]["content"]
    assert regular["class"] == "RegularArray"
    assert regular["content"]["form_key"] == "node2"
    assert regular["content"]["content"] == leaf.form_details


_HITS_1844 = """
#include <vector>
#include "TClonesArray.h"
#include "TFile.h"
#include "TObject.h"
#include "TString.h"
#include "TTree.h"

class Hit1844 : public TObject {
public:
    TString fNames[3];
    std::vector<std::vector<float>> fVV[2];
    ClassDef(Hit1844, 1)
};

void write_hits_1844(const char *filename) {
    TFile f(filename, "RECREATE");
    TTree t("t", "t");
    TClonesArray *hits = new TClonesArray("Hit1844");
    t.Branch("hits", &hits, 32000, 99);
    // entry 0: no hits; entries 1-2: hits with empty vectors; entries 3-4: filled
    for (int entry = 0; entry < 5; entry++) {
        hits->Clear();
        for (int i = 0; i < entry; i++) {
            Hit1844 *hit = new ((*hits)[i]) Hit1844();
            for (int j = 0; j < 3; j++) hit->fNames[j] = TString::Format("e%d_i%d_s%d", entry, i, j);
            if (entry >= 3) {
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k <= j; k++)
                        hit->fVV[j].push_back(std::vector<float>(k + 1, entry + 0.5f * i));
            }
        }
        t.Fill();
    }
    t.Write();
    f.Close();
    delete hits;
}
"""


@pytest.fixture(scope="module")
def hits_file(tmp_path_factory):
    ROOT = pytest.importorskip("ROOT")
    if not hasattr(ROOT, "write_hits_1844"):
        assert ROOT.gInterpreter.Declare(_HITS_1844)
    filename = str(tmp_path_factory.mktemp("root") / "tclonesarray.root")
    ROOT.write_hits_1844(filename)
    return filename


@pytest.mark.parametrize("branch_name", ["hits.fNames[3]", "hits.fVV[2]"])
def test_tclonesarray_fixed_size_array_of_lists(hits_file, branch_name):
    with uproot.open(hits_file) as file:
        branch = file["t"][branch_name]
        assert branch.interpretation._forth
        forth = branch.array(library="ak").tolist()

        # the same branch, read without AwkwardForth
        interp = uproot.interpretation.objects.AsObjects(
            branch.interpretation.model, branch
        )
        interp._forth = False
        python = branch.array(interpretation=interp, library="ak").tolist()

    assert forth == python
