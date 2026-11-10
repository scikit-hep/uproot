# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a Python-like Future and Executor for Uproot in three levels:

1. :doc:`uproot.source.futures.TrivialFuture` and
   :doc:`uproot.source.futures.TrivialExecutor`: interface only, all activity
   is synchronous.
2. :doc:`uproot.source.futures.Future`, :doc:`uproot.source.futures.Worker`,
   and :doc:`uproot.source.futures.ThreadPoolExecutor`: similar to Python's
   own Future, Thread, and ThreadPoolExecutor, though only a minimal
   implementation is provided. These provide a base class for the following.
3. :doc:`uproot.source.futures.ResourceFuture`,
   :doc:`uproot.source.futures.ResourceWorker`,
   and :doc:`uproot.source.futures.ResourceThreadPoolExecutor`: like the above
   except that a :doc:`uproot.source.chunk.Resource` is associated with every
   worker. When the threads are shut down, the resources (i.e. file handles)
   are released.

These classes implement a *subset* of Python's Future and Executor interfaces.
"""

from __future__ import annotations

import os
import queue
import threading
from abc import ABC, abstractmethod


class Executor(ABC):
    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):012x}>"

    @abstractmethod
    def submit(self, task, /, *args, **kwargs):
        """
        Submit a task to be run in the background and return a Future object
        representing that task.
        """
        raise NotImplementedError

    def shutdown(self, wait: bool = True):
        """
        Stop the executor and free its resources.
        """
        return

    @property
    def closed(self) -> bool:
        """
        True once the executor has started shutting down and no longer accepts
        new tasks.
        """
        return False


##################### use-case 1: trivial Futures/Executor (satisfying formalities)


class TrivialFuture:
    """
    Formally satisfies the interface for a :doc:`uproot.source.futures.Future`
    object, but it is already complete at the time when it is constructed.
    """

    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback, *, context=None):
        """
        The callback is called immediately.
        """
        callback(self)

    def result(self, timeout=None):
        """
        The result of this (Trivial)Future.
        """
        return self._result


class TrivialExecutor(Executor):
    """
    Formally satisfies the interface for a
    :doc:`uproot.source.futures.ThreadPoolExecutor`, but the
    :ref:`uproot.source.futures.TrivialExecutor.submit` method computes its
    ``task`` synchronously.
    """

    def submit(self, task, /, *args, **kwargs):
        """
        Immediately runs ``task(*args, **kwargs)``.
        """
        return TrivialFuture(task(*args, **kwargs))


##################### use-case 2: Python-like Futures/Executor for compute


class Future:
    """
    Args:
        task (function): The function to evaluate.
        args (tuple): Positional arguments for the function.
        kwargs (None or dict): Keyword arguments for the function.

    Like Python 3 ``concurrent.futures.Future`` except that it has only
    the subset of the interface Uproot needs.

    The :doc:`uproot.source.futures.ResourceFuture` extends this class.
    """

    def __init__(self, task, args, kwargs=None):
        self._task = task
        self._args = args
        self._kwargs = {} if kwargs is None else kwargs
        self._finished = threading.Event()
        self._result = None
        self._excinfo = None

    def result(self, timeout=None):
        """
        Waits until the task completes (with a ``timeout``) and returns its
        result.

        If the task raises an exception in its background thread, this function
        raises that exception on the thread on which it is called.
        """
        if not self._finished.wait(timeout=timeout):
            raise TimeoutError("Future.result timed out")
        if self._excinfo is None:
            return self._result
        else:
            raise self._excinfo

    def _run(self):
        try:
            if self._task is None:
                raise RuntimeError("cannot run Future twice")
            self._result = self._task(*self._args, **self._kwargs)
        except Exception as err:
            self._excinfo = err
        self._finished.set()
        self._task = None
        self._args = ()
        self._kwargs = {}


class Worker(threading.Thread):
    """
    Args:
        work_queue (``queue.Queue``): The worker calls ``get`` on this queue
            for tasks in the form of :doc:`uproot.source.futures.Future`
            objects and runs them. If it ever gets a None value, the thread
            is stopped.

    A ``threading.Thread`` for the
    :doc:`uproot.source.futures.ThreadPoolExecutor`.
    """

    def __init__(self, work_queue: queue.Queue):
        super().__init__()
        self.daemon = True
        self._work_queue = work_queue

    @property
    def work_queue(self) -> queue.Queue:
        """
        The worker calls ``get`` on this queue for tasks in the form of
        :doc:`uproot.source.futures.Future` objects and runs them. If it ever
        gets a None value, the thread is stopped.
        """
        return self._work_queue

    def run(self):
        """
        Listens to the :ref:`uproot.source.futures.Worker.work_queue` and
        executes each :doc:`uproot.source.futures.Future` it receives until it
        receives None.
        """
        future = None
        while True:
            del future  # don't hang onto a reference while waiting for more work
            future = self._work_queue.get()
            if future is None:
                break
            assert not isinstance(future, ResourceFuture)
            future._run()


class ThreadPoolExecutor(Executor):
    """
    Args:
        max_workers (None or int): The maximum number of workers to start.
        In the current implementation this is exactly the number of workers.
        If None, use ``os.cpu_count()``.

    Like Python 3 ``concurrent.futures.ThreadPoolExecutor`` except that it has
    only the subset of the interface Uproot needs.

    The :doc:`uproot.source.futures.ResourceThreadPoolExecutor` extends this
    class.
    """

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers or os.cpu_count()
        self._closed = False
        self._shutdown_lock = threading.Lock()

        self._work_queue = queue.Queue()
        self._workers = []
        for _ in range(self._max_workers):
            self._workers.append(Worker(self._work_queue))
        for worker in self._workers:
            worker.start()

    def __repr__(self):
        return f"<{self.__class__.__name__} ({len(self._workers)} workers) at 0x{id(self):012x}>"

    @property
    def max_workers(self) -> int:
        """
        The maximum number of workers.
        """
        return self._max_workers

    @property
    def num_workers(self) -> int:
        """
        The number of workers.
        """
        return len(self._workers)

    @property
    def workers(self) -> list[Worker]:
        """
        A list of workers (:doc:`uproot.source.futures.Worker`).
        """
        return self._workers

    def submit(self, task, /, *args, **kwargs):
        """
        Pass the ``task``, ``args``, and ``kwargs`` onto the workers'
        :ref:`uproot.source.futures.Worker.work_queue` as a
        :doc:`uproot.source.futures.Future` so that it will be executed when
        one is available.
        """
        future = Future(task, args, kwargs)
        with self._shutdown_lock:
            if self._closed:
                raise RuntimeError("cannot submit a task after shutdown")
            self._work_queue.put(future)
        return future

    @property
    def closed(self) -> bool:
        """
        True if :ref:`uproot.source.futures.ThreadPoolExecutor.shutdown` has
        been started; False otherwise.
        """
        return self._closed

    def shutdown(self, wait: bool = True):
        """
        Stop every :doc:`uproot.source.futures.Worker` by putting one None per
        worker on the :ref:`uproot.source.futures.Worker.work_queue` and
        joining each worker thread.
        """
        # submit holds the same lock while it checks _closed and enqueues, so
        # no task can land behind a sentinel, where no worker would reach it
        with self._shutdown_lock:
            self._closed = True
            for _ in self._workers:
                self._work_queue.put(None)
        for worker in self._workers:
            worker.join()


##################### use-case 3: worker-bound resources for I/O


class ResourceFuture(Future):
    """
    Args:
        task (function): The function to evaluate with a
            :doc:`uproot.source.chunk.Resource` as its first argument.

    A :doc:`uproot.source.futures.Future` that uses the
    :doc:`uproot.source.chunk.Resource` associated with the
    :doc:`uproot.source.futures.ResourceWorker` that runs it.
    """

    def __init__(self, task):
        super().__init__(task, None)
        self._notify = None

    def _set_notify(self, notify):
        """
        Set the ``notify`` function that is called when this task is complete.
        """
        self._notify = notify

    def _set_excinfo(self, excinfo):
        if not self._finished.is_set():
            self._excinfo = excinfo
            self._finished.set()
            if self._notify is not None:
                self._notify()

    def _run(self, resource):
        try:
            self._result = self._task(resource)
        except Exception as err:
            self._excinfo = err
        self._finished.set()
        if self._notify is not None:
            self._notify()
            self._notify = None


class ResourceWorker(Worker):
    """
    Args:
        work_queue (``queue.Queue``): The worker calls ``get`` on this queue
            for tasks in the form of :doc:`uproot.source.futures.Future`
            objects and runs them. If it ever gets a None value, the thread
            is stopped.

    A :doc:`uproot.source.futures.Worker` that is bound to a
    :doc:`uproot.source.chunk.Resource`. This
    :ref:`uproot.source.futures.ResourceWorker.resource` is the first argument
    passed to each :doc:`uproot.source.futures.ResourceFuture` that it
    executes.
    """

    def __init__(self, work_queue: queue.Queue, resource):
        super().__init__(work_queue)
        self._resource = resource

    @property
    def resource(self):
        """
        The :doc:`uproot.source.chunk.Resource` that is bound to this worker.
        """
        return self._resource

    def run(self):
        """
        Listens to the :ref:`uproot.source.futures.ResourceWorker.work_queue`
        and executes each :doc:`uproot.source.futures.ResourceFuture` it
        receives (with :ref:`uproot.source.futures.ResourceWorker.resource` as
        its first argument) until it receives None.
        """
        future = None
        while True:
            del future  # don't hang onto a reference while waiting for more work
            future = self._work_queue.get()
            if future is None:
                break
            assert isinstance(future, ResourceFuture)
            future._run(self._resource)


class ResourceThreadPoolExecutor(ThreadPoolExecutor):
    """
    Args:
        resources (list of :doc:`uproot.source.chunk.Resource`): Resources to
            wrap as :doc:`uproot.source.futures.ResourceFuture` objects.

    A :doc:`uproot.source.futures.ThreadPoolExecutor` whose workers are bound
    to resources, such as file handles.
    """

    def __init__(self, resources):
        self._closed = False
        self._shutdown_lock = threading.Lock()

        if len(resources) < 1:
            raise ValueError("at least one worker is required")

        self._work_queue = queue.Queue()
        self._workers = []
        for resource in resources:
            self._workers.append(ResourceWorker(self._work_queue, resource))
        for worker in self._workers:
            worker.start()

    def submit(self, future: ResourceFuture) -> ResourceFuture:
        """
        Pass the ``task`` onto the workers'
        :ref:`uproot.source.futures.ResourceWorker.work_queue` as a
        :doc:`uproot.source.futures.ResourceFuture` so that it will be
        executed with its :ref:`uproot.source.futures.ResourceWorker.resource`
        when that worker is available.
        """
        assert isinstance(future, ResourceFuture)
        with self._shutdown_lock:
            if self._closed:
                raise OSError(
                    f"resource is closed for file {self._workers[0].resource.file_path}"
                )
            self._work_queue.put(future)
        return future

    def close(self):
        """
        Stops all :doc:`uproot.source.futures.ResourceWorker` threads and frees
        their :ref:`uproot.source.futures.ResourceWorker.resource`.
        """
        self.__exit__(None, None, None)

    @property
    def closed(self) -> bool:
        """
        True once the :doc:`uproot.source.futures.ResourceWorker` threads have
        started stopping; their
        :ref:`uproot.source.futures.ResourceWorker.resource` is freed by the
        time teardown returns.
        """
        return self._closed

    def __enter__(self):
        for worker in self._workers:
            worker.resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.shutdown()
        for worker in self._workers:
            worker.resource.__exit__(exception_type, exception_value, traceback)


##################### use-case 4: resources for I/O with trivial executor


class ResourceTrivialExecutor(TrivialExecutor):
    """
    Args:
        resource (:doc:`uproot.source.chunk.Resource`): Resource to
            wrap as :doc:`uproot.source.futures.ResourceFuture` object.

    A :doc:`uproot.source.futures.ResourceTrivialExecutor` is bound
    to a resource, such as a file handle.
    """

    def __init__(self, resource):
        self._resource = resource
        self._closed = False

    def submit(self, future: ResourceFuture) -> ResourceFuture:
        """
        Pass the ``task`` as a
        :doc:`uproot.source.futures.ResourceFuture` so that it will be
        executed with its :ref:`self._resource`.
        """
        assert isinstance(future, ResourceFuture)
        if self.closed:
            raise OSError(f"resource is closed for file {self._resource.file_path}")
        future._run(self._resource)
        return future

    def close(self):
        """
        Stops the :doc:`uproot.source.futures.ResourceTrivialExecutor`
        """
        self.__exit__(None, None, None)

    @property
    def num_workers(self) -> int:
        """
        Always 0: tasks run in the thread that submits them.
        """
        return 0

    @property
    def closed(self) -> bool:
        """
        True if the :doc:`uproot.source.futures.ResourceTrivialExecutor` has
        been stopped.
        """
        return self._closed

    def __enter__(self):
        self._resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.shutdown()
        self._resource.__exit__(exception_type, exception_value, traceback)
        self._closed = True
