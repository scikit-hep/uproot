# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot
from uproot.models.TTable import Model_TTable


class Model_StIOEvent(uproot.model.Model):
    def read_members(self, chunk, cursor, context, file):
        self._bases.append(
            uproot.models.TObject.Model_TObject.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fObj"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )


@pytest.fixture(scope="module")
def datafile():
    yield skhep_testdata.data_path("uproot-issue-418.root")


@pytest.fixture
def custom_classes():
    yield dict(
        uproot.classes,
        StIOEvent=Model_StIOEvent,
        St_g2t_event=Model_TTable,
        St_particle=Model_TTable,
        St_g2t_vertex=Model_TTable,
        St_g2t_track=Model_TTable,
        St_g2t_pythia=Model_TTable,
        St_g2t_tpc_hit=Model_TTable,
        St_g2t_ctf_hit=Model_TTable,
        St_g2t_emc_hit=Model_TTable,
        St_g2t_vpd_hit=Model_TTable,
        St_dst_bfc_status=Model_TTable,
    )


@pytest.fixture
def geant_branch(datafile, custom_classes):
    with uproot.open(datafile, custom_classes=custom_classes) as f:
        yield f["geantBranch.0000000123.0000000322"]


def test_geant_dot_root(geant_branch):
    items = {obj.all_members["fName"]: obj for obj in geant_branch.members["fObj"]}
    assert items["g2t_pythia"].data["subprocess_id"] == 1
