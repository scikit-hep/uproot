import pytest
from uproot.source.coalesce import CoalesceConfig, RangeRequest, _coalesce, Future


@pytest.mark.parametrize(
    "config",
    [
        CoalesceConfig(),
        CoalesceConfig(max_range_gap=2, max_request_ranges=1),
    ],
    ids=["default", "tiny"],
)
@pytest.mark.parametrize(
    "ranges",
    [
        [(1, 3), (4, 6), (10, 20)],
        [(1, 3), (10, 20), (4, 6), (9, 10)],
        [(1, 3), (10, 20), (6, 15)],
        [(1, 3), (10, 20), (6, 25)],
    ],
    ids=["sorted", "jumbled", "overlapped", "nested"],
)
def test_coalesce(ranges, config):
    data = b"abcdefghijklmnopqurstuvwxyz"

    all_requests = [RangeRequest(start, stop, None) for start, stop in ranges]
    nreq = 0
    for merged_request in _coalesce(all_requests, config):
        future = Future()
        future.set_result([data[start:stop] for start, stop in merged_request.ranges()])
        merged_request.set_future(future, "test.root")
        nreq += 1

    if config.max_range_gap == 2:
        assert nreq > 1

    for req in all_requests:
        assert req.future
        assert req.future.result() == data[req.start : req.stop]
