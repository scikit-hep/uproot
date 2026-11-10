# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot

pytest.importorskip("s3fs")


def test_s3_fail(s3_server):
    bucket_url, storage_options = s3_server
    with pytest.raises(FileNotFoundError, match="does not exist"):
        with uproot.source.fsspec.FSSpecSource(
            f"{bucket_url}/does-not-exist", **storage_options
        ) as source:
            source.chunk(0, 100)
