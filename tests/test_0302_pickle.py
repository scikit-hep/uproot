# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pickle
import shutil

import pytest
import skhep_testdata

import uproot


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.file.MemmapSource,
        uproot.source.fsspec.FSSpecSource,
    ],
)
def test_pickle_roundtrip_local(handler):
    with uproot.open(
        skhep_testdata.data_path("uproot-small-dy-withoffsets.root"), handler=handler
    ) as f:
        pkl = pickle.dumps(f["tree"])

    branch = pickle.loads(pkl)["Muon_pt"]
    muonpt1 = branch.array(library="np", entry_start=10, entry_stop=20)
    assert [x.tolist() for x in muonpt1] == [
        [20.60145378112793],
        [50.36957550048828, 41.21387481689453, 3.1869382858276367],
        [51.685970306396484, 35.227813720703125],
        [],
        [],
        [],
        [],
        [23.073759078979492],
        [32.921417236328125, 8.922308921813965, 4.368383407592773],
        [51.9132194519043, 31.930095672607422],
    ]


ZMUMU_M = [
    82.4626915551,
    83.6262040052,
    83.3084646667,
    82.1493728809,
    90.4691230355,
    89.7576631707,
    89.7739431721,
    90.4855320463,
    91.7737014813,
    91.9488195857,
]


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.http.HTTPSource,
        uproot.source.fsspec.FSSpecSource,
    ],
)
def test_pickle_roundtrip_http(http_server, handler):
    pytest.importorskip("aiohttp")

    with uproot.open(f"{http_server}/uproot-Zmumu.root", handler=handler) as f:
        pkl = pickle.dumps(f["events"])

    tree = pickle.loads(pkl)
    assert tree.num_entries == 2304
    assert list(tree["M"].array(entry_stop=10)) == ZMUMU_M


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.xrootd.XRootDSource,
        uproot.source.fsspec.FSSpecSource,
    ],
)
@pytest.mark.xrootd
def test_pickle_roundtrip_xrootd(xrootd_server, handler):
    url, directory = xrootd_server
    shutil.copy(skhep_testdata.data_path("uproot-Zmumu.root"), directory)

    with uproot.open(f"{url}/uproot-Zmumu.root", handler=handler) as f:
        pkl = pickle.dumps(f["events"])

    tree = pickle.loads(pkl)
    assert tree.num_entries == 2304
    assert list(tree["M"].array(entry_stop=10)) == ZMUMU_M
