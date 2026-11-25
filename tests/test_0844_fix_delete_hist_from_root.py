# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
import os
import numpy
import pytest
import uproot
import struct

ROOT = pytest.importorskip("ROOT")


def test_delete_from_file_with_deleted_histogram_at_the_end(tmp_path):
    h0f = ROOT.TH1F("h0f", "Random numbers", 200, 0, 10)
    h1f = ROOT.TH1F("h1f", "Random numbers", 200, 0, 10)
    h2f = ROOT.TH1F("h2f", "Random numbers", 200, 0, 10)

    filename = os.path.join(tmp_path, "hist_del_test_equal.root")
    tfile = ROOT.TFile(filename, "RECREATE")

    h0f.Write()
    h1f.Write()
    h2f.Write()
    tfile.Close()

    with uproot.update(filename) as f:
        assert f.keys() == ["h0f;1", "h1f;1", "h2f;1"]
        del f["h2f;1"]
        del f["h1f;1"]
        del f["h0f;1"]
        assert f.keys() == []


def test_locations_recreate_update(tmp_path):
    filename = os.path.join(tmp_path, "uproot_test_locations.root")

    with uproot.recreate(filename) as f:
        file_size = os.path.getsize(filename)
        f["hnf0"] = numpy.histogram(numpy.random.normal(0, 1, 100000))
        f["hnf1"] = numpy.histogram(numpy.random.normal(0, 1, 100000))
        f["hnf2"] = numpy.histogram(numpy.random.normal(0, 1, 100000))

        location_recreate_0 = f._cascading.data.get_key("hnf0", 1).location
        location_recreate_1 = f._cascading.data.get_key("hnf1", 1).location
        location_recreate_2 = f._cascading.data.get_key("hnf2", 1).location

        seek_location_recreate_0 = f._cascading.data.get_key("hnf0", 1).seek_location
        seek_location_recreate_1 = f._cascading.data.get_key("hnf1", 1).seek_location
        seek_location_recreate_2 = f._cascading.data.get_key("hnf2", 1).seek_location

    with uproot.update(filename) as g:
        assert location_recreate_0 == g._cascading.data.get_key("hnf0", 1).location
        assert location_recreate_1 == g._cascading.data.get_key("hnf1", 1).location
        assert location_recreate_2 == g._cascading.data.get_key("hnf2", 1).location

        assert (
            seek_location_recreate_0
            == g._cascading.data.get_key("hnf0", 1).seek_location
        )
        assert (
            seek_location_recreate_1
            == g._cascading.data.get_key("hnf1", 1).seek_location
        )
        assert (
            seek_location_recreate_2
            == g._cascading.data.get_key("hnf2", 1).seek_location
        )
