# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot


def test_writable_directory_delitem_nonexistent_raises_key_error(tmp_path):
    """
    Deleting a nonexistent key from a WritableDirectory must raise a KeyError
    subclass (uproot.exceptions.KeyInFileError), not AttributeError.

    Before the fix, _del() called key.seek_location without checking whether
    get_key() returned None, resulting in:
        AttributeError: 'NoneType' object has no attribute 'seek_location'
    """
    fname = tmp_path / "del_nonexistent.root"
    f = uproot.recreate(fname)
    f["t"] = {"x": np.array([1, 2, 3], dtype="i4")}
    with pytest.raises(uproot.KeyInFileError, match="nonexistent_key"):
        del f["nonexistent_key"]
    assert f.keys(cycle=False) == ["t"]
    f.close()
