# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
from __future__ import annotations

import os
import pathlib
import shutil
from functools import wraps

import pytest
import skhep_testdata

try:
    from pytest_pyodide import run_in_pyodide
    from pytest_pyodide.decorator import copy_files_to_pyodide
except ImportError:
    pytest.skip("Pyodide is not available", allow_module_level=True)


# copy skhep_testdata files to testdata directory (needed for @copy_files_to_pyodide)
def ensure_testdata(filename):
    if not pathlib.Path("skhep_testdata/" + filename).is_file():
        filepath = skhep_testdata.data_path(filename)
        os.makedirs("skhep_testdata", exist_ok=True)
        shutil.copyfile(filepath, "skhep_testdata/" + filename)


def run_test_in_pyodide(test_file=None, **kwargs):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(selenium):
            if test_file is not None:
                ensure_testdata(test_file)

            @copy_files_to_pyodide(
                file_list=[("dist", "dist")]
                + (
                    []
                    if test_file is None
                    else [("skhep_testdata/" + test_file, test_file)]
                ),
                install_wheels=True,
            )
            def inner_func(selenium):
                run_in_pyodide(**kwargs)(test_func)(selenium)

            return inner_func(selenium)

        return wrapper

    return decorator
