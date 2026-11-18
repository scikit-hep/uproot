# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import gc
import importlib.metadata
import sys
import types

import pytest

import uproot
import uproot.extras


def test_message_without_keys_or_location():
    assert str(uproot.KeyInFileError("x")) == "not found: 'x'"


def test_message_with_everything():
    error = uproot.KeyInFileError(
        "muons",
        because="it's a test",
        cycle=3,
        keys=["jets", "muon", "electrons"],
        file_path="file.root",
        object_path="/tree",
    )
    assert str(error) == (
        "not found: 'muons' with cycle 3 because it's a test\n\n"
        "    Available keys: 'muon', 'jets', 'electrons'\n"
        "\nin file file.root\nin object /tree"
    )


def test_message_with_any_cycle():
    assert (
        str(uproot.KeyInFileError("x", cycle="any"))
        == "not found: 'x' (with any cycle number)"
    )


def test_message_with_no_available_keys():
    assert str(uproot.KeyInFileError("x", keys=[])) == (
        "not found: 'x'\n\n    Available keys: (none!)\n"
    )


def test_message_truncates_long_list_of_keys():
    keys = [f"branch_{i:03d}" for i in range(50)]
    message = str(uproot.KeyInFileError("branch_000", keys=keys))
    shown = message.split("Available keys: ")[1].strip()

    assert shown.startswith("'branch_000', ")
    assert shown.endswith("...")
    assert len(shown) < 225
    assert "branch_049" not in shown


def test_message_narrows_huge_list_of_keys_by_prefix():
    # only the keys sharing a first character with the missing key are compared
    keys = [f"a{i}" for i in range(1001)] + ["b1", "b22"]
    message = str(uproot.KeyInFileError("b2", keys=keys))
    assert message == "not found: 'b2'\n\n    Available keys: 'b22', 'b1'\n"


@pytest.mark.parametrize(
    "function, blocked, message",
    [
        (uproot.extras.pandas, ["pandas"], "install the 'pandas' package"),
        (uproot.extras.XRootD_client, ["XRootD"], "Install XRootD python bindings"),
        (uproot.extras.isal, ["isal"], "install the 'isal' package"),
        (uproot.extras.deflate, ["deflate"], "install the 'deflate' package"),
        (
            uproot.extras.boost_histogram,
            ["boost_histogram"],
            "install the 'boost-histogram' package",
        ),
        (uproot.extras.hist, ["hist"], "install the 'hist' package"),
        (
            uproot.extras.dask,
            ["dask", "dask.blockwise"],
            "for uproot.dask with 'library=\"np\"'",
        ),
        (
            uproot.extras.dask_array,
            ["dask", "dask.array"],
            "for uproot.dask with 'library=\"np\"'",
        ),
        (
            uproot.extras.dask_awkward,
            ["dask_awkward"],
            "install 'dask' and the 'dask-awkward' package",
        ),
        (
            uproot.extras.awkward_pandas,
            ["awkward_pandas"],
            "install the 'awkward-pandas' package",
        ),
        (uproot.extras.cupy, ["cupy"], "Cupy is required for GDS reading"),
        (uproot.extras.kvikio, ["kvikio"], "Kvikio is required for GDS reading"),
        (uproot.extras.nvcomp, ["nvidia"], "nvCOMP is required for GDS reading"),
    ],
)
def test_missing_dependency_message(function, blocked, message, monkeypatch):
    monkeypatch.setenv("XRD_RUNFORKHANDLER", "0")  # restored after XRootD_client
    for module in blocked:
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ModuleNotFoundError, match=message):
        function()


def test_dask_awkward_too_old(monkeypatch):
    fake = types.SimpleNamespace(__version__="2023.9.0")
    monkeypatch.setitem(sys.modules, "dask_awkward", fake)
    with pytest.raises(
        ModuleNotFoundError,
        match="dask-awkward 2023.10.0 or newer; you have dask-awkward 2023.9.0",
    ):
        uproot.extras.dask_awkward()


def fake_versions(monkeypatch, versions):
    def version(name):
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)


@pytest.mark.parametrize(
    "versions, expected",
    [
        ({"xrootd": "5.6.0"}, "5.6.0"),
        ({"pyxrootd": "4.11.0"}, "4.11.0"),
        ({}, None),
    ],
)
def test_xrootd_version(versions, expected, monkeypatch):
    fake_versions(monkeypatch, versions)
    assert uproot.extras.xrootd_version() == expected


@pytest.mark.parametrize(
    "versions, expected",
    [
        ({"xrootd": "5.0.0"}, True),
        ({"xrootd": "5.1.0"}, False),
        ({"xrootd": "v20230101-self-built"}, False),  # assumed to be new
        ({}, False),  # not installed
    ],
)
def test_older_xrootd(versions, expected, monkeypatch):
    fake_versions(monkeypatch, versions)
    assert uproot.extras.older_xrootd("5.1.0") is expected


class FakeXRootDFile:
    def __init__(self, is_open):
        self.opened = is_open
        self.close_calls = 0

    def is_open(self):
        return self.opened

    def close(self):
        self.close_calls += 1


class DeadXRootDFile(FakeXRootDFile):
    def is_open(self):
        raise ReferenceError("weakly-referenced object no longer exists")


def test_old_xrootd_closes_files_at_exit(monkeypatch):
    client = types.SimpleNamespace(
        file=types.SimpleNamespace(File=FakeXRootDFile),
    )
    monkeypatch.setitem(sys.modules, "XRootD", types.SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "XRootD.client", client)
    monkeypatch.setenv("XRD_RUNFORKHANDLER", "0")  # restored after XRootD_client
    fake_versions(monkeypatch, {"xrootd": "5.0.0"})

    registered = []
    monkeypatch.setattr(uproot.extras.atexit, "register", registered.append)

    assert uproot.extras.XRootD_client() is client
    assert len(registered) == 1

    opened, closed, dead = (
        FakeXRootDFile(True),
        FakeXRootDFile(False),
        DeadXRootDFile(True),
    )
    # the real heap can hold objects that warn when inspected, like ROOT's lazy modules
    monkeypatch.setattr(gc, "get_objects", lambda: [opened, closed, dead, object()])
    registered[0]()

    assert opened.close_calls == 1
    assert closed.close_calls == 0
    assert dead.close_calls == 0
