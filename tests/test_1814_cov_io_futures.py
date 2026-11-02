# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import queue
import threading

import pytest

import uproot
from uproot.source import futures


class Resource:
    file_path = "resource.root"

    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.exited += 1


class MinimalExecutor(futures.Executor):
    def submit(self, task, /, *args, **kwargs):
        return super().submit(task, *args, **kwargs)


def test_executor_interface():
    executor = MinimalExecutor()
    assert repr(executor).startswith("<MinimalExecutor at 0x")
    assert executor.closed is False
    assert executor.shutdown() is None
    with pytest.raises(NotImplementedError):
        executor.submit(print)


def test_trivial_future_calls_callback_immediately():
    future = futures.TrivialFuture(123)
    called = []
    future.add_done_callback(called.append)
    assert called == [future]
    assert future.result() == 123


def test_future_cannot_run_twice():
    future = futures.Future(lambda: 123, ())
    future._run()
    assert future.result() == 123

    future._run()
    with pytest.raises(RuntimeError, match="cannot run Future twice"):
        future.result()


def test_future_result_times_out():
    future = futures.Future(lambda: 123, ())
    with pytest.raises(TimeoutError, match="Future.result timed out"):
        future.result(timeout=0)


def test_worker_runs_futures_until_it_gets_none():
    work_queue = queue.Queue()
    worker = futures.Worker(work_queue)
    assert worker.work_queue is work_queue

    future = futures.Future(lambda x: x + 1, (1,))
    work_queue.put(future)
    work_queue.put(None)
    worker.run()  # synchronously: it stops at the None

    assert future.result(timeout=0) == 2


def test_thread_pool_executor():
    executor = futures.ThreadPoolExecutor(3)
    try:
        assert repr(executor).startswith("<ThreadPoolExecutor (3 workers) at 0x")
        assert executor.num_workers == 3
        assert executor.max_workers == 3
        assert len(executor.workers) == 3
        assert executor.submit(lambda x, y: x + y, 1, y=2).result() == 3
    finally:
        executor.shutdown()

    assert executor.closed is True
    assert all(not worker.is_alive() for worker in executor.workers)
    with pytest.raises(RuntimeError, match="cannot submit a task after shutdown"):
        executor.submit(print)


def test_thread_pool_executor_propagates_exceptions():
    def task():
        raise ValueError("from the worker")

    executor = futures.ThreadPoolExecutor(1)
    try:
        with pytest.raises(ValueError, match="from the worker"):
            executor.submit(task).result()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("with_notify", [True, False])
def test_resource_future_set_excinfo(with_notify):
    notified = []
    future = futures.ResourceFuture(lambda resource: 123)
    if with_notify:
        future._set_notify(lambda: notified.append(True))

    error = ValueError("failed elsewhere")
    future._set_excinfo(error)
    with pytest.raises(ValueError, match="failed elsewhere"):
        future.result(timeout=0)
    assert notified == ([True] if with_notify else [])


def test_resource_future_set_excinfo_after_it_finished_changes_nothing():
    notified = []
    future = futures.ResourceFuture(lambda resource: resource)
    future._set_notify(lambda: notified.append(True))
    future._run(123)
    assert notified == [True]

    future._set_excinfo(ValueError("too late"))
    assert future.result(timeout=0) == 123
    assert notified == [True]


def test_resource_future_run_keeps_exception():
    def task(resource):
        raise ValueError(resource)

    future = futures.ResourceFuture(task)
    future._run("from the task")
    with pytest.raises(ValueError, match="from the task"):
        future.result(timeout=0)


def test_resource_thread_pool_needs_a_worker():
    with pytest.raises(ValueError, match="at least one worker is required"):
        futures.ResourceThreadPoolExecutor([])


def test_resource_thread_pool_gives_each_task_its_workers_resource():
    resources = [Resource(), Resource()]
    executor = futures.ResourceThreadPoolExecutor(resources)

    # both workers must be holding a task at the same time, so neither can take both
    barrier = threading.Barrier(2, timeout=60)

    def task(resource):
        barrier.wait()
        return resource

    first = futures.ResourceFuture(task)
    second = futures.ResourceFuture(task)
    assert executor.submit(first) is first
    executor.submit(second)
    assert {id(first.result()), id(second.result())} == {id(x) for x in resources}

    executor.close()

    assert executor.closed is True
    assert [x.exited for x in resources] == [1, 1]
    with pytest.raises(OSError, match="resource is closed for file resource.root"):
        executor.submit(futures.ResourceFuture(task))


def test_resource_thread_pool_context_manager():
    resource = Resource()
    executor = futures.ResourceThreadPoolExecutor([resource])
    executor.__enter__()
    assert resource.entered == 1
    executor.__exit__(None, None, None)
    assert resource.exited == 1


def test_resource_trivial_executor():
    resource = Resource()
    executor = futures.ResourceTrivialExecutor(resource)
    assert executor.closed is False

    executor.__enter__()
    assert resource.entered == 1

    future = executor.submit(futures.ResourceFuture(lambda x: x))
    assert future.result(timeout=0) is resource

    executor.close()
    assert executor.closed is True
    assert resource.exited == 1
    with pytest.raises(OSError, match="resource is closed for file resource.root"):
        executor.submit(futures.ResourceFuture(lambda x: x))


def test_slice_future_in_wasm_does_not_pass_a_timeout(monkeypatch):
    class PyodideFuture:
        def result(self):  # takes no timeout
            return [b"abc", b"def"]

    monkeypatch.setattr(uproot._util, "wasm", True)
    future = uproot.source.coalesce.SliceFuture(PyodideFuture(), 1)
    assert future.result(timeout=10) == b"def"


def test_coalesce_requests_in_wasm_notifies_directly(monkeypatch):
    class Source:
        file_path = "file.root"

    def submit(ranges):
        return futures.TrivialFuture([b"abcd"])

    monkeypatch.setattr(uproot._util, "wasm", True)
    notifications = queue.Queue()
    chunks = uproot.source.coalesce.coalesce_requests(
        [(0, 2), (2, 4)], submit, Source(), notifications
    )

    assert notifications.qsize() == 2
    assert [notifications.get_nowait() for _ in chunks] == chunks
    assert [chunk.raw_data.tobytes() for chunk in chunks] == [b"ab", b"cd"]
