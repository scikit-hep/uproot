# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec

import sys
import os
import pathlib
import fsspec
import numpy as np
import awkward as ak

is_windows = sys.platform.startswith("win")


def test_fsspec_writing_no_integration(tmp_path):
    uri = os.path.join(tmp_path, "some", "path", "file.root")
    with fsspec.open(uri, mode="wb") as file_obj:
        # write a simple root file
        with uproot.recreate(file_obj) as f:
            f.mktree("tree", {"x": np.array([1, 2, 3])})

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize("scheme", ["", "file://", "simplecache::file://"])
def test_fsspec_writing_local(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.recreate(uri) as f:
        f.mktree("tree", {"x": np.array([1, 2, 3])})

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


# https://github.com/scikit-hep/uproot/issues/325
@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file:",  # this is not a valid schema, but we should support it
        "file://",
        "simplecache::file://",
    ],
)
@pytest.mark.parametrize(
    "filename", ["file.root", "file%2Eroot", "my%E2%80%92file.root", "my%20file.root"]
)
@pytest.mark.parametrize(
    "slash_prefix",
    [""] if is_windows else ["", "/"],
)
def test_fsspec_writing_local_uri(tmp_path, scheme, slash_prefix, filename, request):
    os.chdir(tmp_path)

    try:
        uri = scheme + slash_prefix + os.path.join(tmp_path, "some", "path", filename)

        with uproot.create(uri) as f:
            f.mktree("tree", {"x": np.array([1, 2, 3])})
        with uproot.open(uri) as f:
            assert f["tree"]["x"].array().tolist() == [1, 2, 3]

    finally:
        os.chdir(request.config.invocation_params.dir)


@pytest.mark.parametrize(
    "input_value",
    [
        "\\file.root",
        "\\file%2Eroot",
        "\\my%E2%80%92file.root",
        "\\my%20file.root",
        "file:\\file.root",
        "file:\\file%2Eroot",
        "file:\\my%E2%80%92file.root",
        "file:\\my%20file.root",
        "file://\\file.root",
        "file://\\file%2Eroot",
        "file://\\my%E2%80%92file.root",
        "file://\\my%20file.root",
        "simplecache::file://\\file.root",
        "simplecache::file://\\file%2Eroot",
        "simplecache::file://\\my%E2%80%92file.root",
        "simplecache::file://\\my%20file.root",
    ],
)
def test_fsspec_backslash_prefix(input_value):
    # for slash_prefix `\` avoid testing the creation of files and only check if the uri is parsed correctly
    url, obj = uproot._util.file_object_path_split(input_value)
    assert obj is None
    assert url == input_value


@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file://",
        "simplecache::file://",
        # "memory://", # uncomment when https://github.com/fsspec/filesystem_spec/pull/1426 is available in pypi
        "simplecache::memory://",
    ],
)
def test_fsspec_writing_create(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.create(uri) as f:
        f.mktree("tree", {"x": np.array([1, 2, 3])})

    with pytest.raises(FileExistsError, match="path exists and refusing to overwrite"):
        uproot.create(uri)
    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_issue_1029(tmp_path):
    # https://github.com/scikit-hep/uproot/issues/1029
    urlpath = os.path.join(tmp_path, "some", "path", "file.root")
    urlpath = pathlib.Path(urlpath)

    with uproot.recreate(urlpath) as f:
        f.mktree("tree_1", {"x": np.array([1, 2, 3])})

    with uproot.update(urlpath) as f:
        f.mktree("tree_2", {"y": np.array([4, 5, 6])})

    with uproot.open(urlpath) as f:
        assert f["tree_1"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree_2"]["y"].array().tolist() == [4, 5, 6]


def test_fsspec_writing_http(http_server):
    pytest.importorskip("aiohttp")

    uri = f"{http_server}/file.root"

    # TODO: review this when fsspec supports writing to http
    with pytest.raises(NotImplementedError):
        uproot.recreate(uri)


@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file://",
        "simplecache::file://",
    ],
)
def test_fsspec_writing_local_update(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.recreate(uri) as f:
        f.mktree("tree1", {"x": np.array([1, 2, 3])})

    with uproot.update(uri) as f:
        f.mktree("tree2", {"y": np.array([4, 5, 6])})

    # read data and compare
    with uproot.open(uri) as f:
        assert f["tree1"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree2"]["y"].array().tolist() == [4, 5, 6]


@pytest.mark.parametrize(
    "scheme",
    [
        "ssh://",
        "simplecache::ssh://",
    ],
)
def test_fsspec_writing_ssh(tmp_path, scheme):
    pytest.importorskip("paramiko")
    import paramiko
    import getpass

    user = getpass.getuser()
    host = "localhost"
    port = 22

    # only test this if we can connect to the host (this will work in GitHub Actions)
    try:
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user)
    except (
        paramiko.ssh_exception.SSHException,
        paramiko.ssh_exception.NoValidConnectionsError,
    ) as e:
        pytest.skip(f"ssh connection to host failed: {e}")

    local_path = os.path.join(tmp_path, "file.root")
    uri = f"{scheme}{user}@{host}:{port}{local_path}"

    with uproot.recreate(uri) as f:
        f.mktree("tree", {"x": np.array([1, 2, 3])})

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "scheme",
    [
        "memory://",
        "simplecache::memory://",
    ],
)
def test_fsspec_writing_memory(tmp_path, scheme):
    uri = f"{scheme}{tmp_path}/file.root"

    with uproot.recreate(uri) as f:
        f.mktree("tree", {"x": np.array([1, 2, 3])})

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_write_append_fsspec_xrootd(xrootd_server):
    remote_path, _ = xrootd_server
    filename = "file.root"
    remote_file_path = os.path.join(remote_path, filename)
    array = ak.Array({"x": [1, 2, 3], "y": [4, 5, 6]})
    file = uproot.recreate(remote_file_path)
    file.mktree("tree", array)
    file.close()
    array2 = ak.Array({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7]})
    file = uproot.update(remote_file_path)
    file.mktree("other_tree", array2)
    file.close()
    with uproot.open(remote_file_path) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree"]["y"].array().tolist() == [4, 5, 6]
        assert f["other_tree"]["x"].array().tolist() == [1, 2, 3, 4]
        assert f["other_tree"]["y"].array().tolist() == [4, 5, 6, 7]
