# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## What this is

Uproot is a library for reading and writing ROOT files (the CERN HEP data format) in pure Python and NumPy — an I/O library only, with no C++ ROOT dependency.

## Commands

```bash
uv pip install -e . --group=dev      # editable install with dev dependencies

uv run pytest tests                  # full test suite
uv run pytest tests/test_1603_num_entries.py            # one test file
uv run pytest tests/test_1603_num_entries.py::test_name # one test

prek -a --quiet                      # lint/format (pre-commit: black, ruff, etc.)
```

- Test files are named `test_<issue-or-PR-number>_<short_description>.py`. New tests for a fix or feature go in a new file numbered after the issue/PR.
- Tests pull sample ROOT files via `skhep_testdata`; many tests need network access. Markers: `slow`, `network`, `distributed`, `xrootd`.
- Pytest treats warnings as errors (`filterwarnings = error`).
- Ruff lints `src/` but not `tests/`; isort requires `from __future__ import annotations` in every module. Every source file starts with the license header line `# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE`.
- Version is derived from git tags via hatch-vcs (`src/uproot/version.py` is generated; don't edit).

## Architecture

The read path is layered: **source → reading → model → behavior → interpretation**.

1. **Physical layer — `src/uproot/source/`**: gets raw bytes without interpreting them. `Source` implementations for memory-mapped/plain files, HTTP, XRootD, fsspec, and Python file-like objects; data is delivered as `Chunk`s and navigated with a `Cursor` (which tracks a byte position and does the low-level struct unpacking). `src/uproot/sink/` is the writable counterpart.

2. **File structure — `src/uproot/reading.py`**: `uproot.open` and the three classes that are *not* modeled like other ROOT objects: `ReadOnlyFile` (TFile), `ReadOnlyDirectory` (TDirectory), and TKey. Directories act as Mappings from object names to deserialized objects.

3. **Models — `src/uproot/model.py` and `src/uproot/models/`**: every object read from a ROOT file is a `Model` subclass. ROOT classes are versioned, so `DispatchByVersion` reads the version bytes and selects a `VersionedModel` (e.g. `Model_TH1F_v3`). `src/uproot/models/` holds hand-written/pre-generated models for common classes; for everything else, model classes are **generated at runtime** from the file's embedded `TStreamerInfo` (`streamers.py` + `deserialization.py`) and placed in the `uproot.dynamic` module. C++ class names are encoded to Python identifiers via `uproot.model.classname_encode` (e.g. `ROOT::RThing` → `Model_ROOT_3a3a_RThing`). `dev/make-models.py` regenerates `models/TH.py` and `models/TGraph.py`.

4. **Behaviors — `src/uproot/behaviors/`**: user-facing methods/properties mixed into models by naming convention, discovered by `uproot.behavior.behavior_of`: a submodule named after the encoded class name (without the `Model_` prefix) containing a class with the encoded name. E.g. histogram `.values()`/`.to_hist()` on `TH1`, and the main array-reading API (`arrays`, `iterate`, `concatenate`) on `behaviors/TBranch.py`'s `HasBranches`.

5. **Interpretation — `src/uproot/interpretation/`**: turns raw TBasket bytes into arrays. `identify.py` picks an `Interpretation` (numerical, jagged, strings, objects, grouped) for each branch; `library.py` maps results to the output library (`np` NumPy, `ak` Awkward — the default, `pd` Pandas). `containers.py` handles `std::vector` etc.; `_awkwardforth.py` generates AwkwardForth code to fast-path object-type branches. `language/python.py` evaluates expressions passed to `arrays()`.

- **Writing — `src/uproot/writing/`**: `uproot.create`/`recreate`/`update`. Internals use a "cascade" design (`_cascade.py`, `_cascadetree.py`, `_cascadentuple.py`) where high-level writes cascade into updates of dependent file structures (freesegments, streamers, directories).
- **RNTuple** (ROOT's TTree successor) is supported alongside TTree: `models/RNTuple.py`, `behaviors/RNTuple.py`, and the writing cascade.
- **Dask/parallelism**: `uproot.dask` lives in `_dask.py` (+ `writing/_dask_write.py`); thread-pool machinery in `source/futures.py`.
- `tests-cuda/` (GPU/kvikio) and `tests-wasm/` (Pyodide) have separate CI jobs and dependency groups (`test-gpu-cu12/13`, `test-pyodide`, `test-xrootd`).

## Pull requests

Per CONTRIBUTING.md: significant AI assistance must be disclosed in the PR description, and PRs require meaningful human review — the human contributor remains the author.
