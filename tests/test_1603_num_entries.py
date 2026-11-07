# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import uproot


def test_num_entries(tmp_path):
    for i in range(5):
        filepath = os.path.join(tmp_path, f"test_{i}.root")
        with uproot.recreate(filepath) as file:
            file.mktree("tree", {"x": list(range(1 + i))})

    num_entries = uproot.num_entries(f"{tmp_path}/test_*.root:tree")
    assert sorted(num_entries, key=lambda x: x[0]) == [
        (os.path.join(tmp_path, f"test_{i}.root"), "tree", 1 + i) for i in range(5)
    ]
