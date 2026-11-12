# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
Tests for path objects that carry storage_options (e.g. UPath from universal-pathlib).
The fix allows users to bundle a URL and its fsspec storage options (auth headers,
credentials, etc.) into a single path object and pass it directly to uproot functions.
"""

import pytest
import skhep_testdata

import uproot
import uproot.source.fsspec


class PathWithStorageOptions:
    """
    Minimal stand-in for universal_pathlib.UPath: a path object that pairs a
    URL string with fsspec storage_options.  No upath dependency required.
    """

    def __init__(self, url, storage_options=None):
        self._url = url
        self.storage_options = storage_options or {}

    def __str__(self):
        return self._url

    def __repr__(self):
        return f"PathWithStorageOptions({self._url!r}, {self.storage_options!r})"


def test_open_propagates_storage_options():
    """storage_options on a path object must reach FSSpecSource._fsspec_options."""
    path = PathWithStorageOptions(
        skhep_testdata.data_path("uproot-HZZ.root"),
        storage_options={"headers": {"Authorization": "Bearer test-token"}},
    )
    with uproot.open(path) as f:
        source = f._file._source
        assert isinstance(source, uproot.source.fsspec.FSSpecSource)
        assert source._fsspec_options.get("headers") == {
            "Authorization": "Bearer test-token"
        }


def test_open_with_storage_options_reads_data():
    """A path object with storage_options must open and read data correctly."""
    path = PathWithStorageOptions(
        skhep_testdata.data_path("uproot-HZZ.root"),
        storage_options={"headers": {"Authorization": "Bearer test-token"}},
    )
    with uproot.open(path) as f:
        tree = f["events"]
        assert tree.num_entries == 2421
        assert tree["NJet"].array(library="np")[:3].tolist() == [0, 1, 0]


def test_open_empty_storage_options_is_noop():
    """A path object with empty storage_options behaves identically to a plain string."""
    plain_path = skhep_testdata.data_path("uproot-HZZ.root")
    path_obj = PathWithStorageOptions(plain_path, storage_options={})
    with uproot.open(path_obj) as f:
        assert f.file_path == plain_path
        assert f._file._source._fsspec_options == {}
        assert f["events"].num_entries == 2421


def test_open_explicit_kwarg_wins_over_storage_options():
    """Explicit kwargs to uproot.open() must take precedence over storage_options."""
    path = PathWithStorageOptions(
        skhep_testdata.data_path("uproot-HZZ.root"),
        storage_options={"headers": {"Authorization": "Bearer from-path"}},
    )
    # The caller's explicit kwarg should override the path's storage_options
    with uproot.open(path, headers={"Authorization": "Bearer explicit"}) as f:
        source = f._file._source
        assert source._fsspec_options.get("headers") == {
            "Authorization": "Bearer explicit"
        }
