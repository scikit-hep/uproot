# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed_write``: one ``TTree`` file per partition through ``dask_write``'s
``ak_to_root``, as a ``graphed.write`` plan."""

from __future__ import annotations

import os
import pickle

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")

import graphed  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import Plan, SequentialRunner  # noqa: E402
from graphed_executors.local import ProcessPoolExecutor, ThreadExecutor  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402


def _make_tree(path, n=20, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return path + ":" + name


def _back(paths, tree="tree"):
    return ak.concatenate([uproot.open(p)[tree].arrays() for p in paths])


def test_writes_one_ttree_file_per_partition(tmp_path):
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src)
    outdir = os.path.join(tmp_path, "out")
    paths = uproot.graphed_write(g, outdir, steps_per_file=4, executor="thread")
    # fsspec reports local Windows paths with "/" separators
    assert [os.path.normpath(p) for p in paths] == [
        os.path.join(outdir, f"part-0000{i}.root") for i in range(4)
    ]
    assert all(uproot.open(p)["tree"].classname == "TTree" for p in paths)
    pieces = [uproot.open(p)["tree"].arrays() for p in paths]
    assert [len(p) for p in pieces] == [5, 5, 5, 5]
    back = ak.concatenate(pieces)
    ref = uproot.open(src).arrays()
    assert ak.array_equal(back.x, ref.x) and ak.array_equal(back.y, ref.y)


def test_prefix_tree_name_and_multifile_numbering(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"), n=6)
    b = _make_tree(os.path.join(tmp_path, "b.root"), n=4)
    g = uproot.graphed([a, b])
    paths = uproot.graphed_write(
        g,
        os.path.join(tmp_path, "out"),
        steps_per_file=2,
        prefix="skim",
        tree_name="ev",
        executor="thread",
    )
    assert [os.path.basename(p) for p in paths] == [
        f"skim-0000{i}.root" for i in range(4)
    ]
    pieces = [uproot.open(p)["ev"].arrays().x.tolist() for p in paths]
    assert pieces == [[0, 1, 2], [3, 4, 5], [0, 1], [2, 3]]


def test_nested_records_flatten_to_joined_branch_names(tmp_path):
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src)
    paths = uproot.graphed_write(
        gak.zip({"v": gak.zip({"x": g.x, "y": g.y})}),
        os.path.join(tmp_path, "out"),
        executor="thread",
    )
    assert uproot.open(paths[0])["tree"].keys() == ["v_x", "v_y"]


def test_compute_false_returns_the_plan_and_writes_nothing(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    outdir = os.path.join(tmp_path, "out")
    plan = uproot.graphed_write(g, outdir, steps_per_file=2, compute=False)
    assert isinstance(plan, Plan) and len(plan.tasks) == 2
    assert os.listdir(outdir) == []
    # ships to a process pool unchanged
    paths = SequentialRunner().run(pickle.loads(pickle.dumps(plan))).value
    assert sorted(os.listdir(outdir)) == [os.path.basename(p) for p in paths]
    assert _back(paths).x.tolist() == list(range(20))


def test_executor_is_a_name_an_instance_or_a_class(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    out = os.path.join(tmp_path, "out")
    paths = uproot.graphed_write(g, out, executor=SequentialRunner())
    assert _back(paths).x.tolist() == list(range(20))
    paths = uproot.graphed_write(g, out, executor=ThreadExecutor, max_workers=2)
    assert _back(paths).x.tolist() == list(range(20))
    with pytest.raises(ValueError, match="executor must be 'process' or 'thread'"):
        uproot.graphed_write(g, out, executor="threads")


def test_process_pool_with_a_behavior_backend_by_import_ref(tmp_path):
    vector = pytest.importorskip("vector")
    vector.register_awkward()
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src, behavior=vector.backends.awkward.behavior)
    v = gak.with_name(gak.zip({"x": g.x, "y": g.y}), "Vector2D")
    rec = gak.zip({"rho": v.rho})
    out = os.path.join(tmp_path, "out")
    import functools

    from graphed.awkward import AwkwardBackend

    by_value = functools.partial(
        AwkwardBackend, behavior=vector.backends.awkward.behavior
    )
    with pytest.raises(pickle.PicklingError):  # the behavior dict holds lambdas
        pickle.dumps(uproot.graphed_write(rec, out, compute=False, backend=by_value))
    paths = uproot.graphed_write(
        rec,
        out,
        steps_per_file=2,
        executor=ProcessPoolExecutor,
        max_workers=1,
        backend="tests.graphed_helpers.vector_backend_ref:make_backend",
    )
    ref = uproot.open(src).arrays()
    assert ak.array_equal(_back(paths).rho, np.hypot(ref.x, ref.y))


def test_derived_record_is_written_and_a_fieldless_expression_is_refused(tmp_path):
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src)
    out = os.path.join(tmp_path, "out")
    paths = uproot.graphed_write(
        gak.zip({"x": g.x, "doubled": g.x * 2.0 + 1.0}),
        out,
        steps_per_file=3,
        executor="thread",
    )
    back = _back(paths)
    assert back.fields == ["x", "doubled"]
    assert ak.array_equal(back.doubled, uproot.open(src).arrays().x * 2.0 + 1.0)
    with pytest.raises(TypeError, match="has no fields"):
        uproot.graphed_write(g.x + g.y, out, executor="thread")
    assert sorted(os.listdir(out)) == [os.path.basename(p) for p in paths]


def test_compression_names_match_dask_write(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    out = os.path.join(tmp_path, "out")
    for name in ("zlib", "ZLIB", "lz4", "ZSTD", "lzma"):
        (path,) = uproot.graphed_write(g, out, compression=name, executor="thread")
        assert uproot.open(path).file.compression.name.lower() == name.lower()
    with pytest.raises(ValueError, match="unrecognized compression algorithm: bogus"):
        uproot.graphed_write(g, out, compression="bogus", executor="thread")


def test_tree_options_reach_mktree(tmp_path):
    hzz = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    g = uproot.graphed(hzz, filter_name=["Jet_Px", "NJet"])
    (path,) = uproot.graphed_write(
        gak.zip({"jet": gak.zip({"px": g.Jet_Px})}),
        os.path.join(tmp_path, "out"),
        executor="thread",
        title="skim",
        field_name=lambda outer, inner: inner if outer == "" else f"{outer}.{inner}",
        counter_name=lambda counted: "num_" + counted,
    )
    tree = uproot.open(path)["tree"]
    assert tree.title == "skim"
    assert tree.keys() == ["num_jet", "jet.px"]
    assert ak.array_equal(tree["jet.px"].array(), uproot.open(hzz)["Jet_Px"].array())


def test_remote_destinations_go_through_fsspec(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    paths = uproot.graphed_write(
        g,
        "memory://bucket/skims",
        steps_per_file=2,
        executor="thread",
        storage_options={},
    )
    assert paths == [f"memory:///bucket/skims/part-0000{i}.root" for i in range(2)]
    assert _back(paths).x.tolist() == list(range(20))


def test_steps_beyond_the_entry_count_write_no_empty_part(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=3))
    paths = uproot.graphed_write(
        g, os.path.join(tmp_path, "out"), steps_per_file=5, executor="thread"
    )
    assert [os.path.basename(p) for p in paths] == [
        "part-00001.root",
        "part-00003.root",
        "part-00004.root",
    ]  # steps 0 and 2 resolve empty and write nothing
    assert _back(paths).x.tolist() == [0.0, 1.0, 2.0]


def test_a_missing_tree_under_allow_missing_writes_no_part(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"), name="other")
    opened = uproot.graphed(a)
    form = opened.session.form(opened).tt.layout.form
    g = uproot.graphed(
        [a, b.replace("other", "events")],
        known_base_form=form,
        allow_missing=True,
        name="g",
    )
    paths = uproot.graphed_write(g, os.path.join(tmp_path, "out"), executor="thread")
    assert [os.path.basename(p) for p in paths] == ["part-00000.root"]
    assert _back(paths).x.tolist() == list(range(20))


def test_externals_reach_the_write_workers(tmp_path):
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src)
    paths = uproot.graphed_write(
        gak.zip({"a": graphed.apply(lambda x: x * 2, g.x)}),
        os.path.join(tmp_path, "out"),
        steps_per_file=2,
        executor="thread",
    )
    assert ak.array_equal(_back(paths).a, 2 * uproot.open(src).arrays().x)


@pytest.mark.filterwarnings("ignore:.*is deprecated; pass session=:DeprecationWarning")
def test_refusals_happen_before_anything_is_written(tmp_path):
    src = _make_tree(os.path.join(tmp_path, "in.root"))
    out = os.path.join(tmp_path, "out")
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(
            uproot.graphed([src, src]), out, steps_per_file=2, executor="thread"
        )
    g = uproot.graphed(src, name="g")
    with pytest.raises(
        graphed.GraphedError, match="'slice' reduces the partitioned axis"
    ):
        uproot.graphed_write(
            gak.zip({"x": g.x[2:8]}), out, steps_per_file=2, executor="thread"
        )
    with pytest.raises(TypeError, match="explicit 'steps'"):
        uproot.graphed_write(
            uproot.graphed(
                {
                    src.rsplit(":", 1)[0]: {
                        "object_path": "events",
                        "steps": [[0, 5], [5, 20]],
                    }
                },
                name="steps",
            ),
            out,
            executor="thread",
        )
    with pytest.raises(TypeError, match="expects a uproot.graphed Array"):
        uproot.graphed_write(ak.Array([1.0]), out)
    from graphed.awkward import AwkwardBackend, from_awkward

    mem = from_awkward(graphed.Session(AwkwardBackend()), "mem", ak.Array({"x": [1.0]}))
    with pytest.raises(TypeError, match="exactly one uproot.graphed source"):
        uproot.graphed_write(mem, out)
    assert not os.path.exists(out)


def test_a_mapped_source_writes_through_its_mapping(tmp_path):
    where = (
        skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"
    )
    g = uproot.graphed(
        where, filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping()
    )
    _nid, source = uproot._graphed._uproot_source(g, "test")
    seen = []
    real = source.read_range
    source.read_range = lambda tree, keys, start, stop: (
        seen.append(tuple(keys)),
        real(tree, keys, start, stop),
    )[1]
    paths = uproot.graphed_write(
        gak.zip({"pt2": g.Jet.pt2}),
        os.path.join(tmp_path, "out"),
        steps_per_file=2,
        executor="thread",
    )
    assert seen == [
        ("Jet_pt", "nJet"),
        ("Jet_pt", "nJet"),
    ]  # the declared branches, not Jet_eta
    assert ak.array_equal(_back(paths).pt2, mini.eager_events(where).Jet.pt2)
