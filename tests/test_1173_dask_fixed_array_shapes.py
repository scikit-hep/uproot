# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import numpy as np
import pytest

import uproot

pytest.importorskip("dask.array")


@pytest.mark.parametrize(
    ("open_files", "reduce"), [(True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize(
    ("dtype", "shape"), [(np.float64, (4,)), (np.float32, (3,)), (np.float64, ())]
)
def test_inconsistent_array_dtypes(tmp_path, open_files, dtype, shape, reduce):
    files = {}
    for name, base, inner in [
        ("first.root", np.float64, (3,)),
        ("different.root", dtype, shape),
    ]:
        path = tmp_path / name
        values = np.zeros((5, *inner), dtype=base)
        with uproot.recreate(path) as output:
            output.mktree("tree", {"values": np.dtype((base, inner))}).extend(
                {"values": values}
            )
        files[path] = "tree"

    if open_files:
        with pytest.raises(
            ValueError, match=r"inconsistent NumPy dtype.*different\.root"
        ):
            uproot.dask(files, library="np", open_files=True)
    else:
        array = uproot.dask(files, library="np", open_files=False)["values"]
        if reduce:
            array = array.sum(axis=1)
        with pytest.raises(
            ValueError, match=r"inconsistent NumPy dtype.*different\.root"
        ):
            array.compute()


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize("inner_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("num_entries", [0, 10])
def test_fixed_array_shapes(tmp_path, open_files, inner_shape, num_entries):
    path = tmp_path / "arrays.root"
    values = np.arange(num_entries * np.prod(inner_shape, dtype=int), dtype=np.float64)
    values = values.reshape((num_entries, *inner_shape))
    with uproot.recreate(path) as output:
        tree = output.mktree("tree", {"values": np.dtype((np.float64, inner_shape))})
        tree.extend({"values": values})

    array = uproot.dask(
        {path: "tree"}, library="np", open_files=open_files, steps_per_file=3
    )["values"]
    assert array.ndim == values.ndim
    assert array.shape[1:] == inner_shape
    if open_files:
        assert array.shape == values.shape
    assert array.chunks[1:] == tuple((size,) for size in inner_shape)
    np.testing.assert_array_equal(array.compute(), values)
    if inner_shape:
        np.testing.assert_array_equal(array.sum(axis=-1).compute(), values.sum(axis=-1))
        np.testing.assert_array_equal(array[..., 1].compute(), values[..., 1])


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize("explicit_steps", [True, False])
def test_fixed_arrays_multiple_files(tmp_path, open_files, explicit_steps):
    values = np.arange(60, dtype=np.int32).reshape(10, 2, 3)
    files = {}
    for index in range(2):
        path = tmp_path / f"arrays-{index}.root"
        with uproot.recreate(path) as output:
            tree = output.mktree("tree", {"values": np.dtype((np.int32, (2, 3)))})
            tree.extend({"values": values + index})
        files[path] = (
            {"object_path": "tree", "steps": [0, 4, 10]} if explicit_steps else "tree"
        )

    array = uproot.dask(files, library="np", open_files=open_files)["values"]
    expected = np.concatenate([values, values + 1])
    assert array.shape[1:] == (2, 3)
    if open_files or explicit_steps:
        assert array.shape == expected.shape
        np.testing.assert_array_equal(array[8:12, 1, :].compute(), expected[8:12, 1, :])
    np.testing.assert_array_equal(array.compute(), expected)
    np.testing.assert_array_equal(
        array.sum(axis=(1, 2)).compute(), expected.sum(axis=(1, 2))
    )
