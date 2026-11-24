# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import uproot


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    with uproot.recreate(filename) as file:
        file["one"] = "one"

    with uproot.update(filename) as file:
        file["two"] = "two"

    with uproot.open(filename) as file:
        assert file.file.fVersion == uproot.writing._cascade.FileHeader.class_version
