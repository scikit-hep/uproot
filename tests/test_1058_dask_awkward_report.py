import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize(
    "allowed", [True, (OSError,), (FileNotFoundError, RuntimeError)]
)
def test_with_report(allowed):
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = "/some/file/that/doesnt/exist"
    files = [test_path1, test_path2, test_path3]
    collection, report = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=allowed,
    )
    computed, creport = dask.compute(collection, report)
    assert len(computed) == 2 * 2304  # the bad file is left out
    assert creport[0].exception is None  # test_path1 is good
    assert creport[1].exception is None  # test_path2 is good
    assert creport[2].exception == "FileNotFoundError"  # test_path3 is a bad file
    assert test_path3 in creport[2].message


def test_with_report_exception_missed():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = "/some/file/that/doesnt/exist"
    files = [test_path1, test_path2, test_path3]
    collection, report = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=(
            ValueError,  # allow ValueErrors, but that's not what happens!
        ),
    )
    with pytest.raises(FileNotFoundError, match=test_path3):
        dask.compute(collection, report)


def test_with_report_known_divisions():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root")
    files = {test_path: {"object_path": "events", "steps": [[0, 1152], [1152, 2304]]}}
    collection = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=False,
    )
    assert collection.known_divisions
    assert collection.divisions == (0, 1152, 2304)
    with uproot.open(test_path + ":events") as tree:
        expected = tree["M"].array()
    (ccollection,) = dask.compute(collection)
    assert ccollection.M.tolist() == expected.tolist()
    collection, report = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=True,
    )
    assert not collection.known_divisions
    assert collection.divisions == (None, None, None)
    ccollection, creport = dask.compute(collection, report)
    assert ccollection.M.tolist() == expected.tolist()
    assert creport.exception.tolist() == [None, None]
