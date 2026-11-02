# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a physical layer for remote files, accessed via HTTP(S).

Defines a :doc:`uproot.source.http.HTTPResource` (stateless) and two sources:
:doc:`uproot.source.http.MultithreadedHTTPSource` and
:doc:`uproot.source.http.HTTPSource`. The multi-threaded source only requires
the server to support byte range requests (code 206), but the general source
requires the server to support multi-part byte range requests. If the server
does not support multi-part GET, :doc:`uproot.source.http.HTTPSource`
automatically falls back to :doc:`uproot.source.http.MultithreadedHTTPSource`.

Despite the name, both sources support secure HTTPS (selected by URL scheme).
"""

from __future__ import annotations

import base64
import http.client
import queue
import re
import socket
import threading
import urllib.parse
from urllib.parse import urlparse

import uproot
import uproot.source.chunk
import uproot.source.futures


def make_connection(parsed_url: urllib.parse.ParseResult, timeout: float | None):
    """
    Args:
        parsed_url (``urllib.parse.ParseResult``): The URL to connect to, which
            may be HTTP or HTTPS.
        timeout (None or float): An optional timeout in seconds.

    Creates a ``http.client.HTTPConnection`` or a ``http.client.HTTPSConnection``,
    depending on the URL scheme.
    """
    from http.client import HTTPConnection, HTTPSConnection

    # default socket timeout is None, which can cause hangs
    if timeout is None and socket.getdefaulttimeout() is None:
        timeout = 30

    if parsed_url.scheme == "https":
        return HTTPSConnection(parsed_url.hostname, parsed_url.port, timeout=timeout)

    elif parsed_url.scheme == "http":
        return HTTPConnection(parsed_url.hostname, parsed_url.port, timeout)

    else:
        raise ValueError(
            f"unrecognized URL scheme for HTTP MultipartSource: {parsed_url.scheme}"
        )


def full_path(parsed_url) -> str:
    """
    Returns the ``parsed_url.path`` with ``"?"`` and the ``parsed_url.query``
    if it exists, just the path otherwise.
    """
    if parsed_url.query:
        return parsed_url.path + "?" + parsed_url.query
    else:
        return parsed_url.path


def basic_auth_headers(parsed_url):
    """
    Returns the headers required for basic authorization, if parsed_url contains
    a username / password pair, otherwise returns an empty dict
    """
    if parsed_url.username is None or parsed_url.password is None:
        return {}
    ret = {
        "Authorization": "Basic "
        + base64.b64encode(
            (parsed_url.username + ":" + parsed_url.password).encode("utf-8")
        ).decode("utf-8")
    }
    return ret


def get_num_bytes(file_path: str, parsed_url: urllib.parse.ParseResult, timeout) -> int:
    """
    Args:
        file_path (str): The URL to access as a raw string.
        parsed_url (``urllib.parse.ParseResult``): The URL to access.
        timeout (None or float): An optional timeout in seconds.

    Returns the number of bytes in the file by making a HEAD request.
    """
    connection = make_connection(parsed_url, timeout)
    auth_headers = basic_auth_headers(parsed_url)
    connection.request("HEAD", full_path(parsed_url), headers=auth_headers)
    response = connection.getresponse()

    while 300 <= response.status < 400:
        response.close()
        connection.close()
        for k, x in response.getheaders():
            if k.lower() == "location":
                redirect_url = urlparse(x)
                connection = make_connection(redirect_url, timeout)
                connection.request(
                    "HEAD", full_path(redirect_url), headers=auth_headers
                )
                response = connection.getresponse()
                break
        else:
            raise http.client.HTTPException(
                f"""remote server responded with status {response.status} (redirect) without a 'location'
for URL {file_path}"""
            )

    if response.status == 404:
        response.close()
        connection.close()
        raise uproot._util._file_not_found(file_path, "HTTP(S) returned 404")

    if response.status != 200:
        response.close()
        connection.close()
        raise http.client.HTTPException(
            f"""HTTP response was {response.status}, rather than 200, in attempt to get file size
in file {file_path}"""
        )

    for k, x in response.getheaders():
        if k.lower() == "content-length" and x.strip() != "0":
            response.close()
            connection.close()
            return int(x)
    else:
        response.close()
        connection.close()
        raise http.client.HTTPException(
            f"""response headers did not include content-length: {dict(response.getheaders())}
in file {file_path}"""
        )


class HTTPResource(uproot.source.chunk.Resource):
    """
    Args:
        file_path (str): A URL of the file to open.
        timeout (None or float): An optional timeout in seconds.

    A :doc:`uproot.source.chunk.Resource` for HTTP(S) connections.

    For simplicity, this resource does not manage a live
    ``http.client.HTTPConnection`` or ``http.client.HTTPSConnection``, though
    in principle, it could.
    """

    def __init__(self, file_path, timeout):
        self._file_path = file_path
        self._timeout = timeout
        self._parsed_url = urlparse(file_path)
        self._auth_headers = basic_auth_headers(self._parsed_url)

    @property
    def timeout(self) -> float | None:
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def parsed_url(self):
        """
        A ``urllib.parse.ParseResult`` version of the ``file_path``.
        """
        return self._parsed_url

    @property
    def auth_headers(self):
        """
        Returns a dict containing auth headers, if any for this resource
        """
        return self._auth_headers

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def get(self, connection, start: int, stop: int) -> bytes:
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        response = connection.getresponse()

        if response.status == 404:
            response.close()
            connection.close()
            raise uproot._util._file_not_found(self.file_path, "HTTP(S) returned 404")

        if 300 <= response.status < 400:
            response.close()
            connection.close()
            for k, x in response.getheaders():
                if k.lower() == "location":
                    redirect_url = urlparse(x)
                    redirect = make_connection(redirect_url, self._timeout)
                    redirect.request(
                        "GET",
                        full_path(redirect_url),
                        headers=dict(
                            {"Range": f"bytes={start}-{stop - 1}"}, **self.auth_headers
                        ),
                    )
                    return self.get(redirect, start, stop)

            raise http.client.HTTPException(
                f"""remote server responded with status {response.status} (redirect) without a 'location'
for URL {self._file_path}"""
            )

        if response.status != 206:
            response.close()
            connection.close()
            raise http.client.HTTPException(
                f"""remote server responded with status {response.status}, rather than 206 (range requests)
for URL {self._file_path}"""
            )
        try:
            return response.read()
        finally:
            connection.close()

    @staticmethod
    def future(source: uproot.source.chunk.Source, start: int, stop: int):
        """
        Args:
            source (:doc:`uproot.source.http.HTTPSource` or :doc:`uproot.source.http.MultithreadedHTTPSource`): The
                data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :doc:`uproot.source.futures.ResourceFuture` that calls
        :ref:`uproot.source.http.HTTPResource.get` with ``start`` and ``stop``.
        """
        # The default implementation doesn't work in Pyodide
        if uproot._util.wasm:

            def task(resource):
                import requests

                # https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
                timeout = source.timeout if source.timeout is not None else 30

                r = requests.get(
                    source._file_path,
                    headers=dict(
                        {"Range": f"bytes={start}-{stop - 1}"}, **source.auth_headers
                    ),
                    timeout=timeout,
                )
                return r.content

            return uproot.source.futures.ResourceFuture(task)

        connection = make_connection(source.parsed_url, source.timeout)
        connection.request(
            "GET",
            full_path(source.parsed_url),
            headers=dict({"Range": f"bytes={start}-{stop - 1}"}, **source.auth_headers),
        )

        def task(resource):
            return resource.get(connection, start, stop)

        return uproot.source.futures.ResourceFuture(task)

    @staticmethod
    def multifuture(
        source: uproot.source.chunk.Source,
        range_header: dict,
        ranges: list[tuple[int, int]],
        futures,
        results,
    ):
        """
        Args:
            source (:doc:`uproot.source.http.HTTPSource`): The data source.
            ranges (list of (int, int) 2-tuples): Intervals to fetch
                as (start, stop) pairs in a single request, if possible.
            futures (dict of (int, int) \u2192 :doc:`uproot.source.futures.ResourceFuture`): Mapping
                from (start, stop) to a future that is awaiting its result.
            results (dict of (int, int) \u2192 None or ``numpy.ndarray`` of ``numpy.uint8``): Mapping
                from (start, stop) to None or results.

        Returns a :doc:`uproot.source.futures.ResourceFuture` that attempts
        to perform an HTTP(S) multipart GET, filling ``results`` to satisfy
        the individual :doc:`uproot.source.chunk.Chunk`'s ``futures`` with
        its multipart response.

        If the server does not support multipart GET, that same future
        sets :ref:`uproot.source.http.HTTPSource.fallback` and retries the
        request without multipart, using a
        :doc:`uproot.source.http.MultithreadedHTTPSource` to fill the same
        ``results`` and ``futures``. Subsequent attempts would immediately
        use the :ref:`uproot.source.http.HTTPSource.fallback`.
        """
        # The default implementation doesn't work in Pyodide
        if uproot._util.wasm:

            def task(resource):
                resource.handle_no_multipart(source, ranges, futures, results)

            return uproot.source.futures.ResourceFuture(task)

        connection = make_connection(source.parsed_url, source.timeout)

        connection.request(
            "GET",
            full_path(source.parsed_url),
            headers=dict(**range_header, **source.auth_headers),
        )

        def task(resource):
            nonlocal connection
            response = None
            try:
                response = connection.getresponse()

                if 300 <= response.status < 400:
                    connection.close()

                    for k, x in response.getheaders():
                        if k.lower() == "location":
                            redirect_url = urlparse(x)
                            connection = make_connection(redirect_url, source.timeout)
                            connection.request(
                                "GET",
                                full_path(redirect_url),
                                headers={
                                    **range_header,
                                    **source.auth_headers,
                                },
                            )
                            task(resource)
                            return

                    raise http.client.HTTPException(
                        f"""remote server responded with status {response.status} (redirect) without a 'location'
for URL {source.file_path}"""
                    )

                multipart_supported = resource.is_multipart_supported(ranges, response)

                if not multipart_supported:
                    resource.handle_no_multipart(source, ranges, futures, results)
                else:
                    resource.handle_multipart(
                        source, futures, results, response, ranges
                    )

            except Exception as err:
                for future in futures.values():
                    future._set_excinfo(err)

            finally:
                if response is not None:
                    response.close()
                connection.close()

        return uproot.source.futures.ResourceFuture(task)

    _content_range_size = re.compile(
        b"Content-Range: bytes ([0-9]+-[0-9]+)/([0-9]+)", re.IGNORECASE
    )
    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)", re.IGNORECASE)

    def is_multipart_supported(
        self, ranges: list[tuple[int, int]], response: http.client.HTTPResponse
    ) -> bool:
        """
        Helper function for :ref:`uproot.source.http.HTTPResource.multifuture`
        to check for multipart GET support.
        """
        if response.status != 206:
            return False

        for k, x in response.getheaders():
            if k.lower() == "content-length":
                content_length = int(x)
                for start, stop in ranges:
                    if content_length == stop - start:
                        return False
        else:
            return True

    def handle_no_multipart(
        self,
        source: uproot.source.chunk.Source,
        ranges: list[tuple[int, int]],
        futures,
        results,
    ):
        """
        Helper function for :ref:`uproot.source.http.HTTPResource.multifuture`
        to handle a lack of multipart GET support.
        """
        source._set_fallback()

        notifications = queue.Queue()
        source.fallback.chunks(ranges, notifications)

        for _ in range(len(ranges)):
            chunk = notifications.get()
            results[chunk.start, chunk.stop] = chunk.raw_data
            futures[chunk.start, chunk.stop]._run(self)

    def handle_multipart(
        self,
        source: uproot.source.chunk.Source,
        futures,
        results,
        response: http.client.HTTPResponse,
        ranges: list[tuple[int, int]],
    ):
        """
        Helper function for :ref:`uproot.source.http.HTTPResource.multifuture`
        to handle the multipart GET response.
        """
        if hasattr(response, "readline"):
            response_buffer = response
        else:
            response_buffer = _ResponseBuffer(response)

        original_futures = dict(futures)

        while len(futures) > 0:
            range_string, _size = self.next_header(response_buffer)
            if range_string is None:
                self.handle_no_multipart(source, ranges, original_futures, results)
                return

            start, last = range_string.split(b"-")
            start, last = int(start), int(last)
            stop = last + 1

            length = stop - start
            data = response_buffer.read(length)

            if len(data) != length:
                raise http.client.HTTPException(
                    f"""wrong chunk length {len(data)} (expected {length}) for byte range {range_string.decode()!r} in HTTP multipart
for URL {self._file_path}"""
                )

            found = futures.pop((start, stop), None)

            if found is not None:
                results[start, stop] = data
                found._run(self)

            else:
                now = start
                while now < stop:
                    for (
                        future_start,
                        future_stop,  # noqa: B007 (future_stop is used after the loop)
                    ) in futures:
                        if now == future_start:
                            break
                    else:
                        range_string = range_string.decode("utf-8", "surrogateescape")
                        expecting = ", ".join(f"{a}-{b - 1}" for a, b in futures)
                        raise http.client.HTTPException(
                            f"""unrecognized byte range in headers of HTTP multipart: {range_string!r}

    expecting: {expecting}

for URL {self._file_path}"""
                        )
                    subdata = data[
                        now - start : now + future_stop - future_start - start
                    ]
                    found = futures.pop((future_start, future_stop))
                    results[future_start, future_stop] = subdata
                    found._run(self)
                    now = future_stop

    def next_header(self, response_buffer):
        """
        Helper function for :ref:`uproot.source.http.HTTPResource.multifuture`
        to return the next header from the ``response_buffer``.
        """
        range_string, size = None, None
        # Read the part headers up to (and including) the blank line that
        # separates them from the body, capturing the Content-Range wherever it
        # appears. The Content-Range is not guaranteed to be the last header, so
        # we must keep reading until the blank line is consumed; otherwise any
        # leftover header bytes would shift the part payload. Leading blank
        # lines (e.g. the CRLF trailing the previous part's data) and the part
        # boundary line are skipped before the headers begin.
        seen_header = False
        while True:
            line = response_buffer.readline()
            if len(line) == 0:
                # End of stream.
                break
            if len(line.strip()) == 0:
                if seen_header:
                    # Blank line terminating this part's headers.
                    break
                # Leading blank line before the part's headers; skip it.
                continue
            seen_header = True
            if range_string is None:
                m = self._content_range_size.match(line)
                if m is not None:
                    range_string = m.group(1)
                    size = int(m.group(2))
                else:
                    m = self._content_range.match(line)
                    if m is not None:
                        range_string = m.group(1)
                        size = None
        return range_string, size

    @staticmethod
    def partfuture(results, start: int, stop: int):
        """
        Returns a :doc:`uproot.source.futures.ResourceFuture` to simply select
        the ``(start, stop)`` item from the ``results`` dict.

        In :ref:`uproot.source.http.HTTPSource.chunks`, each chunk has a
        :ref:`uproot.source.http.HTTPResource.partfuture` that are collectively
        filled by a single :ref:`uproot.source.http.HTTPResource.multifuture`.
        """

        def task(resource):
            return results[start, stop]

        return uproot.source.futures.ResourceFuture(task)


class _ResponseBuffer:
    CHUNK = 1024

    def __init__(self, stream):
        self.already_read = b""
        self.stream = stream

    def read(self, length: int):
        if length < len(self.already_read):
            out = self.already_read[:length]
            self.already_read = self.already_read[length:]
            return out

        elif len(self.already_read) > 0:
            out = self.already_read
            self.already_read = b""
            return out + self.stream.read(length - len(out))

        else:
            return self.stream.read(length)

    def readline(self) -> bytes:
        while True:
            try:
                index = self.already_read.index(b"\n")
            except ValueError:
                data = self.stream.read(self.CHUNK)
                if len(data) == 0:
                    # End of stream: there's no newline to wait for.
                    out = self.already_read
                    self.already_read = b""
                    return out
                self.already_read = self.already_read + data
            else:
                out = self.already_read[: index + 1]
                self.already_read = self.already_read[index + 1 :]
                return out


class HTTPSource(uproot.source.chunk.Source):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"num_fallback_workers"``, ``"use_threads"``, and ``"timeout"``.

    A :doc:`uproot.source.chunk.Source` that first attempts an HTTP(S)
    multipart GET, but if the server doesn't support it, it falls back to many
    HTTP(S) connections in threads
    (:doc:`uproot.source.http.MultithreadedHTTPSource`).

    Since the multipart GET is a single request and response, it needs only one
    thread, but it is a background thread (a single
    :doc:`uproot.source.futures.ResourceWorker` in a
    :doc:`uproot.source.futures.ResourceThreadPoolExecutor`).
    """

    ResourceClass = HTTPResource

    def __init__(self, file_path: str, **options):
        options = dict(uproot.reading.open.defaults, **options)

        self._num_fallback_workers = options["num_fallback_workers"]
        self._timeout = options.get("timeout")
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._use_threads = options["use_threads"]
        self._file_path = file_path
        self._num_bytes = None

        self._fallback = None
        self._fallback_lock = threading.Lock()
        self._fallback_options = options.copy()
        self._fallback_options["num_workers"] = self._num_fallback_workers

        self._http_max_header_bytes = options["http_max_header_bytes"]

        # Parse the URL here, so that we can expose these fields
        self._parsed_url = urlparse(file_path)
        self._auth_headers = basic_auth_headers(self._parsed_url)

        self._open()

    def _open(self):
        if self._use_threads:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [HTTPResource(self._file_path, self._timeout)]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                HTTPResource(self._file_path, self._timeout)
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        state.pop("_fallback_lock")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._fallback_lock = threading.Lock()
        self._open()

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        fallback = ""
        if getattr(self, "_fallback", None) is not None:
            fallback = " with fallback"
        return f"<{type(self).__name__} {path}{fallback} at 0x{id(self):012x}>"

    def chunk(self, start: int, stop: int) -> uproot.source.chunk.Chunk:
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        future = self.ResourceClass.future(self, start, stop)
        chunk = uproot.source.chunk.Chunk(self, start, stop, future)
        self._executor.submit(future)
        return chunk

    def chunks(
        self,
        ranges: list[tuple[int, int]],
        notifications: queue.Queue,
    ) -> list[uproot.source.chunk.Chunk]:
        if self._fallback is None:
            self._num_requests += 1
            self._num_requested_chunks += len(ranges)
            self._num_requested_bytes += sum(stop - start for start, stop in ranges)
            chunks = []

            def set_futures_and_results(ranges):
                futures = {}
                results = {}

                for start, stop in ranges:
                    partfuture = self.ResourceClass.partfuture(results, start, stop)
                    futures[start, stop] = partfuture
                    results[start, stop] = None
                    chunk = uproot.source.chunk.Chunk(self, start, stop, partfuture)
                    partfuture._set_notify(
                        uproot.source.chunk.notifier(chunk, notifications)
                    )
                    chunks.append(chunk)

                return futures, results

            def submit_batch(batch, range_header):
                futures, results = set_futures_and_results(batch)
                self._executor.submit(
                    self.ResourceClass.multifuture(
                        self, range_header, batch, futures, results
                    )
                )

            batch_start = 0
            range_header = {"Range": "bytes=" + f"{ranges[0][0]}-{ranges[0][1] - 1}"}

            for i in range(1, len(ranges)):
                this_range = f"{ranges[i][0]}-{ranges[i][1] - 1}"
                new_range_to_append = ", " + this_range
                if len(range_header["Range"]) < self._http_max_header_bytes - len(
                    new_range_to_append
                ):
                    range_header["Range"] += new_range_to_append
                else:
                    submit_batch(ranges[batch_start:i], range_header)
                    batch_start = i
                    range_header = {"Range": "bytes=" + this_range}

            submit_batch(ranges[batch_start:], range_header)

            return chunks

        else:
            return self._fallback.chunks(ranges, notifications)

    @property
    def executor(self):
        """
        The :doc:`uproot.source.futures.ResourceThreadPoolExecutor` that
        manages this source's single background thread.
        """
        return self._executor

    @property
    def closed(self) -> bool:
        return self._executor.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.__exit__(exception_type, exception_value, traceback)
        if self._fallback is not None:
            self._fallback.__exit__(exception_type, exception_value, traceback)

    @property
    def timeout(self):
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def num_bytes(self) -> int:
        if self._num_bytes is None:
            self._num_bytes = get_num_bytes(
                self._file_path, self.parsed_url, self._timeout
            )
        return self._num_bytes

    @property
    def parsed_url(self):
        """
        A ``urllib.parse.ParseResult`` version of the ``file_path``.
        """
        return self._parsed_url

    @property
    def auth_headers(self):
        """
        Dict containing auth headers, if any
        """
        return self._auth_headers

    @property
    def fallback(self):
        """
        If None, the source has not encountered an unsuccessful multipart GET
        and no fallback is needed yet.

        Otherwise, this is a :doc:`uproot.source.http.MultithreadedHTTPSource`
        to which all requests are forwarded.
        """
        return self._fallback

    def _set_fallback(self):
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = MultithreadedHTTPSource(
                    self._file_path, **self._fallback_options
                )


class MultithreadedHTTPSource(uproot.source.chunk.MultithreadedSource):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"num_workers"``, ``"use_threads"``, and ``"timeout"``.

    A :doc:`uproot.source.chunk.MultithreadedSource` that manages many
    :doc:`uproot.source.http.HTTPResource` objects.
    """

    ResourceClass = HTTPResource

    def __init__(self, file_path, **options):
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0
        self._use_threads = options["use_threads"]
        self._num_workers = options["num_workers"]

        self._file_path = file_path
        self._num_bytes = None
        self._timeout = options.get("timeout")

        # Parse the URL here, so that we can expose these fields
        self._parsed_url = urlparse(file_path)
        self._auth_headers = basic_auth_headers(self._parsed_url)

        self._open()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def _open(self):
        if self._use_threads:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [
                    HTTPResource(self._file_path, self._timeout)
                    for _ in range(self._num_workers)
                ]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                HTTPResource(self._file_path, self._timeout)
            )

    @property
    def timeout(self):
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def num_bytes(self) -> int:
        if self._num_bytes is None:
            self._num_bytes = get_num_bytes(
                self._file_path, self.parsed_url, self._timeout
            )
        return self._num_bytes

    @property
    def parsed_url(self):
        """
        A ``urllib.parse.ParseResult`` version of the ``file_path``.
        """
        return self._parsed_url

    @property
    def auth_headers(self):
        """
        Dict containing auth headers, if any
        """
        return self._auth_headers
