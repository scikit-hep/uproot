# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

hist = pytest.importorskip("hist")
boost_histogram = pytest.importorskip("boost_histogram")


def test_issue_1534_wiso_reproducer(tmp_path):
    newfile = tmp_path / "issue_1534.root"

    h = hist.Hist(
        hist.axis.IntCategory([10, 20, 30]),
        hist.axis.Regular(100, 0.0, 10.0),
        storage=hist.storage.Mean(),
    )
    h.fill(10, 2.5, sample=5.0)
    h.fill(20, 7.5, sample=15.0)

    with uproot.recreate(newfile) as fout:
        fout["h"] = h

    with uproot.open(newfile) as fin:
        p2 = fin["h"]
        assert p2.classname == "TProfile2D"
        assert p2.member("fEntries") == 2.0

        vals = p2.values()
        assert vals[0, 25] == pytest.approx(5.0)
        assert vals[1, 75] == pytest.approx(15.0)

        h_roundtrip = p2.to_hist()
        assert h_roundtrip.storage_type is hist.storage.Mean
        assert h_roundtrip.values()[0, 25] == pytest.approx(5.0)
        assert h_roundtrip.values()[1, 75] == pytest.approx(15.0)


def test_write_tprofile2d_mean(tmp_path):
    newfile = tmp_path / "test_p2_mean.root"

    h = hist.Hist(
        hist.axis.Regular(4, 0.0, 4.0, name="x"),
        hist.axis.Regular(5, 0.0, 5.0, name="y"),
        storage=hist.storage.Mean(),
    )
    h.fill([1.0, 1.0], [2.0, 2.0], sample=[10.0, 20.0])
    h.fill([3.0], [4.0], sample=[30.0])

    with uproot.recreate(newfile) as fout:
        fout["p2"] = h

    with uproot.open(newfile) as fin:
        p2 = fin["p2"]
        assert p2.classname == "TProfile2D"
        assert p2.member("fEntries") == 3.0

        vals = p2.values()
        assert vals[1, 2] == pytest.approx(15.0)
        assert vals[3, 4] == pytest.approx(30.0)

        counts = p2.counts()
        assert counts[1, 2] == pytest.approx(2.0)
        assert counts[3, 4] == pytest.approx(1.0)

        h_roundtrip = p2.to_hist()
        assert h_roundtrip.storage_type is hist.storage.Mean
        assert h_roundtrip.values()[1, 2] == pytest.approx(15.0)
        assert h_roundtrip.values()[3, 4] == pytest.approx(30.0)


def test_write_tprofile2d_weighted_mean_and_categories(tmp_path):
    newfile = tmp_path / "test_p2_weighted.root"

    h = hist.Hist(
        hist.axis.IntCategory([10, 20, 30], name="cat"),
        hist.axis.Regular(5, 0.0, 5.0, name="y"),
        storage=hist.storage.WeightedMean(),
    )
    h.fill([10, 10], [1.0, 1.0], sample=[5.0, 15.0], weight=[1.0, 2.0])
    h.fill([20], [3.0], sample=[40.0], weight=[1.5])

    with uproot.recreate(newfile) as fout:
        fout["p2_w"] = h

    with uproot.open(newfile) as fin:
        p2_w = fin["p2_w"]
        assert p2_w.classname == "TProfile2D"
        assert p2_w.member("fEntries") == 4.5

        vals = p2_w.values()
        assert vals[0, 1] == pytest.approx((1.0 * 5.0 + 2.0 * 15.0) / 3.0)
        assert vals[1, 3] == pytest.approx(40.0)

        h_roundtrip = p2_w.to_hist()
        assert h_roundtrip.storage_type is hist.storage.WeightedMean
        assert h_roundtrip.values()[0, 1] == pytest.approx(vals[0, 1])


def test_write_tprofile3d_mean(tmp_path):
    newfile = tmp_path / "test_p3_mean.root"

    h = hist.Hist(
        hist.axis.Regular(3, 0.0, 3.0, name="x"),
        hist.axis.Regular(3, 0.0, 3.0, name="y"),
        hist.axis.Regular(3, 0.0, 3.0, name="z"),
        storage=hist.storage.Mean(),
    )
    h.fill([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], sample=[12.0, 24.0])

    with uproot.recreate(newfile) as fout:
        fout["p3"] = h

    with uproot.open(newfile) as fin:
        p3 = fin["p3"]
        assert p3.classname == "TProfile3D"
        assert p3.member("fEntries") == 2.0

        vals = p3.values()
        assert vals[1, 1, 1] == pytest.approx(18.0)

        h_roundtrip = p3.to_hist()
        assert h_roundtrip.storage_type is hist.storage.Mean
        assert h_roundtrip.values()[1, 1, 1] == pytest.approx(18.0)


def test_write_tprofile3d_weighted_mean_and_categories(tmp_path):
    newfile = tmp_path / "test_p3_weighted.root"

    h = hist.Hist(
        hist.axis.IntCategory([10, 20], name="cat"),
        hist.axis.Regular(3, 0.0, 3.0, name="y"),
        hist.axis.Regular(3, 0.0, 3.0, name="z"),
        storage=hist.storage.WeightedMean(),
    )
    h.fill([10], [1.0], [1.0], sample=[50.0], weight=[3.0])

    with uproot.recreate(newfile) as fout:
        fout["p3_w"] = h

    with uproot.open(newfile) as fin:
        p3_w = fin["p3_w"]
        assert p3_w.classname == "TProfile3D"
        assert p3_w.member("fEntries") == 3.0

        vals = p3_w.values()
        assert vals[0, 1, 1] == pytest.approx(50.0)

        h_roundtrip = p3_w.to_hist()
        assert h_roundtrip.storage_type is hist.storage.WeightedMean
        assert h_roundtrip.values()[0, 1, 1] == pytest.approx(50.0)


def test_tprofile2d_cern_root_roundtrip(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-issue-227a.root")
    roundtrip_path = tmp_path / "roundtrip_p2.root"

    with uproot.open(orig_path) as orig_file:
        orig_p2 = orig_file["hprof2d"]
        h2 = orig_p2.to_boost()

    with uproot.recreate(roundtrip_path) as fout:
        fout["hprof2d"] = h2

    with uproot.open(roundtrip_path) as new_file:
        new_p2 = new_file["hprof2d"]

        for member in [
            "fEntries",
            "fTsumw",
            "fTsumw2",
            "fTsumwx",
            "fTsumwx2",
            "fTsumwy",
            "fTsumwy2",
            "fTsumwxy",
            "fTsumwz",
            "fTsumwz2",
            "fErrorMode",
        ]:
            assert orig_p2.member(member) == new_p2.member(member)

        (orig_raw,) = orig_p2.base(uproot.models.TArray.Model_TArray)
        (new_raw,) = new_p2.base(uproot.models.TArray.Model_TArray)
        numpy.testing.assert_allclose(orig_raw, new_raw)
        numpy.testing.assert_allclose(
            orig_p2.member("fBinEntries"), new_p2.member("fBinEntries")
        )
        numpy.testing.assert_allclose(orig_p2.member("fSumw2"), new_p2.member("fSumw2"))
        numpy.testing.assert_allclose(orig_p2.values(), new_p2.values())
        numpy.testing.assert_allclose(orig_p2.errors(), new_p2.errors())

    orig_p2._members["fSumw2"] = numpy.array([])
    b2_fallback = orig_p2.to_boost()
    assert b2_fallback.values().shape == (2, 3)


def test_tprofile3d_cern_root_roundtrip(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-issue-227b.root")
    roundtrip_path = tmp_path / "roundtrip_p3.root"

    with uproot.open(orig_path) as orig_file:
        orig_p3 = orig_file["hprof3d"]
        h3 = orig_p3.to_boost()

    with uproot.recreate(roundtrip_path) as fout:
        fout["hprof3d"] = h3

    with uproot.open(roundtrip_path) as new_file:
        new_p3 = new_file["hprof3d"]

        for member in [
            "fEntries",
            "fTsumw",
            "fTsumw2",
            "fTsumwx",
            "fTsumwx2",
            "fTsumwy",
            "fTsumwy2",
            "fTsumwxy",
            "fTsumwz",
            "fTsumwz2",
            "fTsumwxz",
            "fTsumwyz",
            "fTsumwt",
            "fTsumwt2",
            "fErrorMode",
        ]:
            assert orig_p3.member(member) == new_p3.member(member)

        (orig_raw,) = orig_p3.base(uproot.models.TArray.Model_TArray)
        (new_raw,) = new_p3.base(uproot.models.TArray.Model_TArray)
        numpy.testing.assert_allclose(orig_raw, new_raw)
        numpy.testing.assert_allclose(
            orig_p3.member("fBinEntries"), new_p3.member("fBinEntries")
        )
        numpy.testing.assert_allclose(orig_p3.member("fSumw2"), new_p3.member("fSumw2"))
        numpy.testing.assert_allclose(orig_p3.values(), new_p3.values())
        numpy.testing.assert_allclose(orig_p3.errors(), new_p3.errors())

    orig_p3._members["fSumw2"] = numpy.array([])
    b3_fallback = orig_p3.to_boost()
    assert b3_fallback.values().shape == (2, 3, 4)


def test_recreate_copy_tprofile2d_and_3d(tmp_path):
    origfile = tmp_path / "orig.root"
    copyfile = tmp_path / "copy.root"

    h2 = hist.Hist(
        hist.axis.Regular(2, 0.0, 2.0),
        hist.axis.Regular(2, 0.0, 2.0),
        storage=hist.storage.Mean(),
    )
    h2.fill([0.5], [0.5], sample=[10.0])

    h3 = hist.Hist(
        hist.axis.Regular(2, 0.0, 2.0),
        hist.axis.Regular(2, 0.0, 2.0),
        hist.axis.Regular(2, 0.0, 2.0),
        storage=hist.storage.Mean(),
    )
    h3.fill([0.5], [0.5], [0.5], sample=[20.0])

    with uproot.recreate(origfile) as fout:
        fout["h2"] = h2
        fout["h3"] = h3

    with uproot.open(origfile) as fin:
        p2_read = fin["h2"]
        p3_read = fin["h3"]

        with uproot.recreate(copyfile) as fout:
            fout["h2_copy"] = p2_read
            fout["h3_copy"] = p3_read

    with uproot.open(copyfile) as fin:
        assert fin["h2_copy"].member("fEntries") == 1.0
        assert fin["h2_copy"].values()[0, 0] == pytest.approx(10.0)
        assert fin["h3_copy"].member("fEntries") == 1.0
        assert fin["h3_copy"].values()[0, 0, 0] == pytest.approx(20.0)


def test_mock_plottable_histogram_without_bases():
    class MockHist2:
        kind = "MEAN"
        axes = [
            uproot.writing.identify.to_TAxis("x", "", 2, 0, 2),
            uproot.writing.identify.to_TAxis("y", "", 2, 0, 2),
        ]

        def values(self, flow=False):
            return numpy.zeros((4, 4) if flow else (2, 2))

        def variances(self, flow=False):
            return None

        def counts(self, flow=False):
            return numpy.zeros((4, 4) if flow else (2, 2))

    with pytest.raises(
        NotImplementedError, match="2D PlottableHistogram with kind='MEAN'"
    ):
        uproot.writing.identify.to_writable(MockHist2())

    class MockHist3:
        kind = "MEAN"
        axes = [
            uproot.writing.identify.to_TAxis("x", "", 2, 0, 2),
            uproot.writing.identify.to_TAxis("y", "", 2, 0, 2),
            uproot.writing.identify.to_TAxis("z", "", 2, 0, 2),
        ]

        def values(self, flow=False):
            return numpy.zeros((4, 4, 4) if flow else (2, 2, 2))

        def variances(self, flow=False):
            return None

        def counts(self, flow=False):
            return numpy.zeros((4, 4, 4) if flow else (2, 2, 2))

    with pytest.raises(
        NotImplementedError, match="3D PlottableHistogram with kind='MEAN'"
    ):
        uproot.writing.identify.to_writable(MockHist3())


def test_write_th2_and_th3_count(tmp_path):
    newfile = tmp_path / "test_count.root"

    h2 = hist.Hist(hist.axis.Regular(2, 0.0, 2.0), hist.axis.Regular(2, 0.0, 2.0))
    h2.fill([0.5], [0.5])

    h3 = hist.Hist(
        hist.axis.Regular(2, 0.0, 2.0),
        hist.axis.Regular(2, 0.0, 2.0),
        hist.axis.Regular(2, 0.0, 2.0),
    )
    h3.fill([0.5], [0.5], [0.5])

    with uproot.recreate(newfile) as fout:
        fout["h2"] = h2
        fout["h3"] = h3

    with uproot.open(newfile) as fin:
        assert fin["h2"].classname == "TH2D"
        assert fin["h3"].classname == "TH3D"
