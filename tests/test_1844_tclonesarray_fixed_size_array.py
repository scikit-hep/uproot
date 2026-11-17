# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot

ROOT = pytest.importorskip("ROOT")

ROOT.gInterpreter.Declare("""
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
""")


@pytest.mark.parametrize("branch_name", ["hits.fNames[3]", "hits.fVV[2]"])
def test_tclonesarray_fixed_size_array_of_lists(tmp_path, branch_name):
    filename = str(tmp_path / "tclonesarray.root")
    ROOT.write_hits_1844(filename)

    with uproot.open(filename) as file:
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
