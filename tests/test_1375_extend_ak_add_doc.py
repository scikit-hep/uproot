# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")

file = skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root")
tree = "Events"

to_test = {
    "Electron_r9": {
        "title": "R9 of the supercluster, calculated with full 5x5 region",
        "typename": "float[]",
    },
    "luminosityBlock": {"title": "luminosityBlock/i", "typename": "uint32_t"},
    "Tau_jetIdx": {
        "title": "index of the associated jet (-1 if none)",
        "typename": "int32_t[]",
    },
}


@pytest.mark.parametrize(
    "open_files, ak_add_doc_value",
    [
        # opening this file is slow, and both modes hand the value to _base_form_of
        (True, True),
        (True, {"__doc__": "title", "typename": "typename"}),
        (False, False),
        (False, True),
        (False, {"__doc__": "title"}),
        (False, {"title": "title"}),
        (False, {"typename": "typename"}),
        (False, {"__doc__": "title", "typename": "typename"}),
    ],
)
def test_extend_ak_add_doc(open_files, ak_add_doc_value):
    events = uproot.dask(
        file + ":" + tree,
        filter_name=list(to_test),
        open_files=open_files,
        ak_add_doc=ak_add_doc_value,
    )
    for branch_name in to_test.keys():
        if ak_add_doc_value is False:
            assert events[branch_name].form.parameters == {}
        elif ak_add_doc_value is True:
            assert events[branch_name].form.parameters == {
                "__doc__": to_test[branch_name]["title"]
            }
        else:
            assert events[branch_name].form.parameters == {
                key: to_test[branch_name][val] for key, val in ak_add_doc_value.items()
            }
