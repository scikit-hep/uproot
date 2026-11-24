# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
import os
import uproot


def test_uppercase_extension(tmp_path):
    filename1 = os.path.join(tmp_path, "test1.ROOT")
    filename2 = os.path.join(tmp_path, "test2.Root")

    with uproot.writing.recreate(filename1) as root_directory:
        root_directory.mkdir("mydir")

    with uproot.writing.recreate(filename2) as root_directory:
        root_directory.mkdir("mydir")

    assert uproot.open(f"{filename1}:mydir").path == ("mydir",)
    assert uproot.open(f"{filename2}:mydir").path == ("mydir",)
