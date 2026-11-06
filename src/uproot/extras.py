# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines functions that import external libraries used by Uproot, but not
required by an Uproot installation. (Uproot only requires NumPy).

If a library cannot be imported, these functions raise ``ModuleNotFoundError`` with
error messages containing instructions on how to install the library.
"""

from __future__ import annotations

import atexit
import importlib.metadata
import os

import packaging.version

from uproot._util import parse_version


def pandas():
    """
    Imports and returns ``pandas``.
    """
    try:
        import pandas
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'pandas' package with:

    pip install pandas

or

    conda install pandas""") from err
    else:
        return pandas


def XRootD_client():
    """
    Imports and returns ``XRootD.client`` (after setting the
    ```XRD_RUNFORKHANDLER`` environment variable to ``"1"``, to allow
    multiprocessing).
    """
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import XRootD
        import XRootD.client

    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """Install XRootD python bindings with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        ) from err

    if older_xrootd("5.1.0"):
        # This is registered after calling "import XRootD.client" so it is ran
        # before XRootD.client.finalize.finalize()
        @atexit.register
        def cleanup_open_files():
            """Clean up any open xrootd file objects at exit

            Required to avoid deadlocks from XRootD, for details see:
            * https://github.com/scikit-hep/uproot/issues/504
            * https://github.com/xrootd/xrootd/pull/1260
            """
            import gc

            for obj in gc.get_objects():
                try:
                    isopen = isinstance(obj, XRootD.client.file.File) and obj.is_open()
                except ReferenceError:
                    pass
                else:
                    if isopen:
                        obj.close()

    return XRootD.client


def older_xrootd(min_version):
    """
    Check if the installed XRootD bindings are newer than a given version
    without importing. Defaults to False if XRootD is not installed. Unrecognized
    versions (i.e. self-built XRootD, whose version numbers are strings)
    return False: that is, they're assumed to be new, so that no warnings
    are raised.
    """
    version = xrootd_version()
    if version is None:
        return False
    else:
        try:
            return parse_version(version) < parse_version(min_version)
        except (TypeError, packaging.version.InvalidVersion):
            return False


def xrootd_version():
    """
    Gets the XRootD version if installed, otherwise returns None.
    """
    try:
        return importlib.metadata.version("xrootd")
    except ModuleNotFoundError:
        try:
            # Versions before 4.11.1 used pyxrootd as the package name
            return importlib.metadata.version("pyxrootd")
        except ModuleNotFoundError:
            return None


def isal():
    """
    Import and return ``isal``.
    """
    try:
        import isal
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'isal' package with:

    pip install isal

or

    conda install python-isal""") from err
    else:
        return isal


def deflate():
    """
    Import and return ``deflate``.
    """
    try:
        import deflate
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'deflate' package with:

    pip install deflate

or

    conda install libdeflate""") from err
    else:
        return deflate


def boost_histogram():
    """
    Imports and returns ``boost-histogram``.
    """
    try:
        import boost_histogram
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'boost-histogram' package with:

    pip install boost-histogram

or

    conda install -c conda-forge boost-histogram""") from err
    else:
        return boost_histogram


def hist():
    """
    Imports and returns ``hist``.
    """
    try:
        import hist
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'hist' package with:

    pip install hist""") from err
    else:
        return hist


def dask():
    """
    Imports and returns ``dask``.
    """
    try:
        import dask
        import dask.blockwise
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask with 'library="np"', install the complete 'dask' package with:
    pip install "dask[complete]"
or
    conda install dask"""
        ) from err
    else:
        return dask


def dask_array():
    """
    Imports and returns ``dask.array``.
    """
    try:
        import dask.array as da
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask with 'library="np"', install the complete 'dask' package with:
    pip install "dask[complete]"
or
    conda install dask"""
        ) from err
    else:
        return da


def dask_awkward():
    """
    Imports and returns ``dask_awkward``.
    """
    try:
        import dask_awkward
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask, install 'dask' and the 'dask-awkward' package with:
    pip install "dask[complete] dask-awkward"
or
    conda install -c conda-forge dask dask-awkward"""
        ) from err
    if parse_version(dask_awkward.__version__) >= parse_version("2023.10.0"):
        return dask_awkward
    else:
        raise ModuleNotFoundError(
            f"Uproot 5.1+ can only be used with dask-awkward 2023.10.0 or newer; you have dask-awkward {dask_awkward.__version__}"
        )


_GRAPHED_MIN = "0.0.5"


def graphed(minimum=_GRAPHED_MIN, caller="uproot.graphed"):
    """
    Imports and returns ``graphed`` (with the submodules Uproot uses), at least version ``minimum``,
    which ``caller`` needs.
    """
    try:
        import graphed
        import graphed.aggregate
        import graphed.awkward
        import graphed.awkward.projection
        import graphed.core
        import graphed.provenance
        import graphed.write
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(f"""for {caller}, install the 'graphed' package with:
    pip install graphed""") from err
    if parse_version(graphed.__version__) < parse_version(minimum):
        raise ModuleNotFoundError(
            f"{caller} needs graphed {minimum} or newer; you have graphed {graphed.__version__}"
        )
    # a recorded op's provenance is the analyst's line, not the Uproot frame that recorded it
    graphed.provenance.register_internal("uproot")
    return graphed


def graphed_executors():
    """
    Imports and returns ``graphed_executors`` (with the submodules Uproot uses).
    """
    try:
        import graphed_executors
        import graphed_executors.local
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.graphed_write, install the 'graphed-executors' package with:
    pip install graphed-executors"""
        ) from err
    else:
        return graphed_executors


def akimbo_pandas():
    """
    Imports and returns ``akimbo.pandas``.
    """
    try:
        import akimbo.pandas
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""install the 'akimbo' package with:
    pip install akimbo
or
    conda install -c conda-forge akimbo""") from err
    else:
        return akimbo.pandas


def cupy():
    """
    Imports and returns ``cupy``.
    """
    try:
        import cupy
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """Cupy is required for GDS reading to work. Please install GDS dependencies with:
    `python3 -m pip install uproot[GDS_cuX]`
where X is the cuda major version on user's system (11 and 12 currently supported). Cuda major version can be checked by calling `nvidia-smi --version` or `nvcc --version` if available."""
        ) from err
    else:
        return cupy


def kvikio():
    """
    Imports and returns ``kvikio``.
    """
    try:
        import kvikio
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """Kvikio is required for GDS reading to work. Please install GDS dependencies with:
    `python3 -m pip install uproot[GDS_cuX]`
where X is the cuda major version on user's system (12 currently supported, 13 is untested). Cuda major version can be checked by calling `nvidia-smi --version` or `nvcc --version` if available."""
        ) from err
    else:
        return kvikio


def nvcomp():
    """
    Imports and returns ``kvikio.nvcomp_codec`` which is not exposed by ``kvikio`` alone.
    """
    try:
        from nvidia import nvcomp
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """nvCOMP is required for GDS reading to work. Please install GDS dependencies with:
    `python3 -m pip install uproot[GDS_cuX]`
where X is the cuda major version on user's system (12 currently supported, 13 is untested). Cuda major version can be checked by calling `nvidia-smi --version` or `nvcc --version` if available."""
        ) from err
    else:
        return nvcomp
