# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""Regression tests for executor, chunk, and unknown-class fixes (see #1688).

* a submit racing with shutdown was accepted and then orphaned behind a sentinel
* ``ThreadPoolExecutor.submit`` accepted keyword arguments and discarded them
* a Chunk that failed (or skipped) its length check returned the short buffer later
* unknown model classes were cached by classname alone, ignoring the version
"""

from __future__ import annotations

import threading
import time

import pytest

import uproot
import uproot._util
import uproot.model
import uproot.source.chunk
import uproot.source.futures


class DummyResource:
    file_path = "dummy"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("use_resources", [False, True])
def test_submit_during_shutdown_is_rejected(use_resources):
    if use_resources:
        executor = uproot.source.futures.ResourceThreadPoolExecutor([DummyResource()])
        submit = lambda task: executor.submit(
            uproot.source.futures.ResourceFuture(task)
        )
        error = OSError
    else:
        executor = uproot.source.futures.ThreadPoolExecutor(1)
        submit = executor.submit
        error = RuntimeError
    message = "resource is closed" if use_resources else "cannot submit a task after"
    # occupy the only worker so that shutdown's join() blocks until released
    started, release = threading.Event(), threading.Event()

    def block(*args):
        started.set()
        release.wait(10)
        return "done"

    blocked = submit(block)
    assert started.wait(10)
    assert not executor.closed

    closer = threading.Thread(target=executor.shutdown)
    closer.start()
    try:
        deadline = time.monotonic() + 10
        while not executor.closed:
            assert time.monotonic() < deadline, "shutdown never marked executor closed"
            time.sleep(0.001)
        # shutdown is still joining the busy worker; a late submit must be
        # rejected rather than queued behind the sentinel and never run
        with pytest.raises(error, match=message):
            submit(lambda *args: 42)
    finally:
        release.set()
        closer.join(10)

    assert blocked.result(timeout=10) == "done"
    with pytest.raises(error, match=message):
        submit(lambda *args: 42)


@pytest.mark.parametrize(
    "executor",
    [uproot.source.futures.TrivialExecutor, uproot.source.futures.ThreadPoolExecutor],
)
def test_submit_forwards_kwargs(executor):
    executor = executor()
    try:
        future = executor.submit(lambda a, x=None, *, y: (a, x, y), 1, x=2, y=3)
        assert future.result(timeout=10) == (1, 2, 3)
    finally:
        executor.shutdown()


class _FakeSource:
    file_path = "fake"


def test_chunk_length_is_checked_on_every_access():
    def make_chunk(data):
        return uproot.source.chunk.Chunk(
            _FakeSource(), 0, 5, uproot.source.futures.TrivialFuture(data)
        )

    full = make_chunk(b"abcde")
    assert full.raw_data.tobytes() == b"abcde"
    assert full.remainder(2, None, {}).tobytes() == b"cde"

    short = make_chunk(b"abc")
    # weaker checks pass (and load the data)...
    short.wait(insist=False)
    assert short.get(0, 3, None, {}).tobytes() == b"abc"
    # ...but stricter ones still raise, every time
    for _ in range(2):
        with pytest.raises(OSError, match="expected Chunk of length 5"):
            short.raw_data
        with pytest.raises(OSError, match="expected Chunk of length 5"):
            short.remainder(0, None, {})
        with pytest.raises(OSError, match="expected Chunk of length 5"):
            short.get(0, 4, None, {})


class _StreamerlessFile:
    custom_classes = None

    def streamer_named(self, classname, version):
        return None


def test_unknown_classes_are_cached_per_version(reset_classes):
    versionless = uproot.model.unknown_class("MyClass")
    assert versionless.__name__ == "Unknown_MyClass"
    assert issubclass(versionless, uproot.model.UnknownClass)

    dispatch = uproot._util.new_class(
        uproot.model.classname_encode("MyClass"),
        (uproot.model.DispatchByVersion,),
        {"known_versions": {}},
    )
    v1 = dispatch.new_class(_StreamerlessFile(), 1)
    v2 = dispatch.new_class(_StreamerlessFile(), 2)
    assert v1.__name__ == "Unknown_MyClass_v1"
    assert v2.__name__ == "Unknown_MyClass_v2"
    assert issubclass(v1, uproot.model.UnknownClassVersion)

    # asking again returns the cached classes rather than building new ones
    assert dispatch.new_class(_StreamerlessFile(), 1) is v1
    assert uproot.model.unknown_class("MyClass") is versionless
    assert set(uproot.unknown_classes) == {
        "Unknown_MyClass",
        "Unknown_MyClass_v1",
        "Unknown_MyClass_v2",
    }
