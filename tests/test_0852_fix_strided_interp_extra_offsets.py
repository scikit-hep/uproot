# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import uproot
import skhep_testdata


def _members_after_headers(branch):
    # every entry is 20 bytes of nested object headers, then 3 big-endian floats
    basket = branch.basket(0)
    assert basket.byte_offsets.tolist() == list(range(0, 3233, 32))
    return numpy.frombuffer(basket.data, dtype=">f4").reshape(-1, 8)[:, 5:]


def test_xyz_vector_with_headers():
    file = uproot.open(skhep_testdata.data_path("uproot-issue-513.root"))
    trkana = file["TrkAnaNeg/trkana"]
    branch = trkana["demcent/_mom"]
    array = branch.array()

    assert len(array) == 101
    assert array.fields == ["fCoordinates"]
    assert array.fCoordinates.fields == ["fX", "fY", "fZ"]

    # a Mu2e conversion electron: close to 105 MeV/c, less what it lost on its way
    first = array[0].fCoordinates
    assert (first.fX, first.fY, first.fZ) == pytest.approx(
        (-11.16814136505127, 85.2090835571289, 57.95511245727539)
    )
    assert numpy.sqrt(first.fX**2 + first.fY**2 + first.fZ**2) == pytest.approx(
        103.654, abs=0.001
    )

    expected = _members_after_headers(branch)
    numpy.testing.assert_array_equal(array.fCoordinates.fX.to_numpy(), expected[:, 0])
    numpy.testing.assert_array_equal(array.fCoordinates.fY.to_numpy(), expected[:, 1])
    numpy.testing.assert_array_equal(array.fCoordinates.fZ.to_numpy(), expected[:, 2])


def test_xyz_vector_with_headers_in_numpy():
    file = uproot.open(skhep_testdata.data_path("uproot-issue-513.root"))
    branch = file["TrkAnaNeg/trkana"]["demcent/_mom"]
    array = branch.array(entry_start=99, library="np")

    assert len(array) == 2
    expected = _members_after_headers(branch)[99:]
    for vector, (x, y, z) in zip(array, expected, strict=True):
        coordinates = vector.member("fCoordinates")
        assert coordinates.member("fX") == x
        assert coordinates.member("fY") == y
        assert coordinates.member("fZ") == z


@pytest.mark.parametrize(
    "members, header_size",
    [
        ([("x", ">f8"), ("y", ">i4"), ("z", ">f4"), ("w", ">i2")], 10),
        ([("x", ">i2"), ("y", ">f8"), ("z", ">i8"), ("w", ">f4")], 20),
    ],
)
def test_headers_keep_the_order_of_members_of_different_types(members, header_size):
    interpretation = uproot.interpretation.objects.AsStridedObjects(
        None, [(None, None)] + [(name, numpy.dtype(x)) for name, x in members]
    )
    item = numpy.dtype(members)
    values = numpy.array([(1, 2, 3, 4), (5, 6, 7, 8)], dtype=item)
    headers = numpy.frombuffer(bytes(range(header_size)), dtype=numpy.uint8)
    size = header_size + item.itemsize
    data = numpy.concatenate(
        [
            numpy.concatenate(
                [headers, numpy.frombuffer(value.tobytes(), dtype=numpy.uint8)]
            )
            for value in values
        ]
    )

    array = interpretation.basket_array(
        data, numpy.array([0, size, 2 * size]), None, None, {}, 0, None, {}
    )

    assert array.dtype.names == ("@headers", *item.names)
    assert array["@headers"].tolist() == [list(range(header_size))] * 2
    for name, _ in members:
        assert array[name].tolist() == values[name].tolist()
