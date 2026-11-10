# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata

import uproot


def test_rntuple_iterate_report():
    """
    ``RNTuple.iterate(report=True)`` must yield ``(arrays, Report)`` pairs,
    matching the contract documented in its docstring and already provided
    by ``TTree.iterate``.
    """
    path = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with uproot.open(path)["Staff"] as rntuple:
        out = list(rntuple.iterate("Age", step_size=1000, report=True))

    assert [(report.tree_entry_start, report.tree_entry_stop) for _, report in out] == [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 3354),
    ]
    assert [report.file_path for _, report in out] == [path] * 4
    assert [len(arrays) for arrays, _ in out] == [1000, 1000, 1000, 354]
