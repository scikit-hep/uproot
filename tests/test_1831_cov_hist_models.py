# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
Checks of the (generated) models in ``uproot.models.TH``, ``uproot.models.TGraph``,
and ``uproot.models.TAtt``, parametrized over all of their classes. Every class is
exercised on the serialized bytes of a real object.
"""

import re
import struct
import types

import numpy
import pytest
import skhep_testdata

import uproot
import uproot._awkwardforth
import uproot.models.TAtt
import uproot.models.TGraph
import uproot.models.TH
import uproot.serialization
import uproot.writing.identify

awkward = pytest.importorskip("awkward")

CannotBeAwkward = uproot.interpretation.objects.CannotBeAwkward
CannotBeForth = uproot.interpretation.objects.CannotBeForth
CannotBeStrided = uproot.interpretation.objects.CannotBeStrided

DTYPES = {"C": "i1", "S": "i2", "I": "i4", "F": "f4", "D": "f8"}

MODELS = sorted(
    (
        cls
        for module in (uproot.models.TH, uproot.models.TGraph, uproot.models.TAtt)
        for cls in vars(module).values()
        if isinstance(cls, type) and issubclass(cls, uproot.model.VersionedModel)
    ),
    key=lambda cls: cls.__name__,
)
MEMBERWISE_MODELS = [cls for cls in MODELS if "read_member_n" in vars(cls)]
WRITABLE_MODELS = [
    cls
    for cls in MODELS
    if cls.writable
    and cls
    not in (
        uproot.models.TGraph.Model_TGraphErrors_v3,
        uproot.models.TGraph.Model_TGraphAsymmErrors_v3,
    )
]

# the first CannotBeStrided raised by each class that can't be strided on its own
STRIDED_BLOCKER = {
    uproot.models.TH.Model_TAxis_v10: "Model_TStreamerObjectPointer of type THashList* in member fLabels of class TAxis",
    uproot.models.TH.Model_TH1_v8: "Model_TStreamerBasicPointer of type double* in member fBuffer of class TH1",
    uproot.models.TGraph.Model_TGraph_v4: "Model_TStreamerBasicPointer of type double* in member fX of class TGraph",
    uproot.models.TGraph.Model_TGraphErrors_v3: "Model_TStreamerBasicPointer of type double* in member fEX of class TGraphErrors",
    uproot.models.TGraph.Model_TGraphAsymmErrors_v3: "Model_TStreamerBasicPointer of type double* in member fEXlow of class TGraphAsymmErrors",
}

# members that are pointers to other objects have no Awkward form
POINTERS = {
    uproot.models.TH.Model_TAxis_v10: ("fLabels", "fModLabs"),
    uproot.models.TGraph.Model_TGraph_v4: ("fFunctions", "fHistogram"),
}

# struct formats of the members of each class: TAtt keeps them in module globals
MODULE_FORMATS = {
    uproot.models.TAtt.Model_TAttLine_v1: uproot.models.TAtt._tattline1_format1,
    uproot.models.TAtt.Model_TAttLine_v2: uproot.models.TAtt._tattline1_format1,
    uproot.models.TAtt.Model_TAttFill_v1: uproot.models.TAtt._tattfill1_format1,
    uproot.models.TAtt.Model_TAttFill_v2: uproot.models.TAtt._tattfill2_format1,
    uproot.models.TAtt.Model_TAttMarker_v2: uproot.models.TAtt._tattmarker2_format1,
}


def name_of(cls):
    return cls.__name__


def root_name(cls):
    return uproot.model.classname_decode(cls.__name__)[0]


def scalar_dtypes(cls):
    """
    The dtypes of a class's fixed-width members, in serialized order, as declared
    by the struct formats that read_members uses.
    """
    if cls in MODULE_FORMATS:
        formats = [MODULE_FORMATS[cls]]
    else:
        formats = [
            getattr(cls, name)
            for name in sorted(vars(cls), key=lambda x: (len(x), x))
            if re.fullmatch("_format[0-9]+", name)
        ]
    return [numpy.dtype(">" + code) for fmt in formats for code in fmt.format[1:]]


def axis(name, **kwargs):
    return uproot.writing.identify.to_TAxis(
        fName=name, fTitle="", fNbins=3, fXmin=-1.0, fXmax=2.0, **kwargs
    )


def write_all_histogram_types(file):
    """
    Writes one of every TH?? and TProfile*, with content that is distinct enough for
    a misplaced member or a wrong array type to show up.
    """
    for code, dtype in DTYPES.items():
        file[f"h1{code}"] = uproot.writing.identify.to_TH1x(
            fName=None,
            fTitle="one",
            data=numpy.arange(1, 6, dtype=dtype),
            fEntries=15.0,
            fTsumw=15.0,
            fTsumw2=55.0,
            fTsumwx=1.5,
            fTsumwx2=2.5,
            fSumw2=numpy.arange(1, 6, dtype="f8"),
            fXaxis=axis("xaxis"),
            fBufferSize=3,
            fBuffer=numpy.array([1.5, 2.5, 3.5]),
        )
        file[f"h2{code}"] = uproot.writing.identify.to_TH2x(
            fName=None,
            fTitle="two",
            data=numpy.arange(1, 26, dtype=dtype),
            fEntries=325.0,
            fTsumw=325.0,
            fTsumw2=5525.0,
            fTsumwx=1.5,
            fTsumwx2=2.5,
            fTsumwy=3.5,
            fTsumwy2=4.5,
            fTsumwxy=5.5,
            fSumw2=numpy.arange(1, 26, dtype="f8"),
            fXaxis=axis("xaxis"),
            fYaxis=axis("yaxis"),
        )
        file[f"h3{code}"] = uproot.writing.identify.to_TH3x(
            fName=None,
            fTitle="three",
            data=numpy.arange(1, 126, dtype=dtype),
            fEntries=7875.0,
            fTsumw=7875.0,
            fTsumw2=1.5,
            fTsumwx=2.5,
            fTsumwx2=3.5,
            fTsumwy=4.5,
            fTsumwy2=5.5,
            fTsumwxy=6.5,
            fTsumwz=7.5,
            fTsumwz2=8.5,
            fTsumwxz=9.5,
            fTsumwyz=10.5,
            fSumw2=numpy.arange(1, 126, dtype="f8"),
            fXaxis=axis("xaxis"),
            fYaxis=axis("yaxis"),
            fZaxis=axis("zaxis"),
        )
    file["p1"] = uproot.writing.identify.to_TProfile(
        fName=None,
        fTitle="profile",
        data=numpy.arange(1, 6, dtype="f8"),
        fEntries=15.0,
        fTsumw=15.0,
        fTsumw2=1.5,
        fTsumwx=2.5,
        fTsumwx2=3.5,
        fTsumwy=4.5,
        fTsumwy2=5.5,
        fSumw2=numpy.arange(1, 6, dtype="f8"),
        fBinEntries=numpy.arange(1, 6, dtype="f8"),
        fBinSumw2=numpy.arange(11, 16, dtype="f8"),
        fXaxis=axis("xaxis"),
        fYmin=-7.5,
        fYmax=7.5,
        fErrorMode=1,
    )
    file["p2"] = uproot.writing.identify.to_TProfile2D(
        fName=None,
        fTitle="profile2d",
        data=numpy.arange(1, 26, dtype="f8"),
        fEntries=15.0,
        fTsumw=15.0,
        fTsumw2=1.5,
        fTsumwx=2.5,
        fTsumwx2=3.5,
        fTsumwy=4.5,
        fTsumwy2=5.5,
        fTsumwxy=6.5,
        fTsumwz=7.5,
        fTsumwz2=8.5,
        fSumw2=numpy.arange(1, 26, dtype="f8"),
        fBinEntries=numpy.arange(1, 26, dtype="f8"),
        fBinSumw2=numpy.arange(11, 36, dtype="f8"),
        fXaxis=axis("xaxis"),
        fYaxis=axis("yaxis"),
        fZmin=-7.5,
        fZmax=7.5,
        fErrorMode=1,
    )
    file["p3"] = uproot.writing.identify.to_TProfile3D(
        fName=None,
        fTitle="profile3d",
        data=numpy.arange(1, 126, dtype="f8"),
        fEntries=15.0,
        fTsumw=15.0,
        fTsumw2=1.5,
        fTsumwx=2.5,
        fTsumwx2=3.5,
        fTsumwy=4.5,
        fTsumwy2=5.5,
        fTsumwxy=6.5,
        fTsumwz=7.5,
        fTsumwz2=8.5,
        fTsumwxz=9.5,
        fTsumwyz=10.5,
        fTsumwt=11.5,
        fTsumwt2=12.5,
        fSumw2=numpy.arange(1, 126, dtype="f8"),
        fBinEntries=numpy.arange(1, 126, dtype="f8"),
        fBinSumw2=numpy.arange(11, 136, dtype="f8"),
        fXaxis=axis("xaxis"),
        fYaxis=axis("yaxis"),
        fZaxis=axis("zaxis"),
        fTmin=-7.5,
        fTmax=7.5,
        fErrorMode=1,
    )


class Sample:
    """
    A deserialized object with the buffer and cursor that it was read from.
    """

    def __init__(self, obj, chunk, file):
        self.obj = obj
        self.chunk = chunk
        self.file = file

    @property
    def cursor(self):
        return self.obj.cursor.copy()

    @property
    def data(self):
        start = self.obj.cursor.index
        return bytes(self.chunk.raw_data[start : start + self.obj.num_bytes])

    @classmethod
    def from_bytes(cls, model, data, file):
        chunk = uproot.source.chunk.Chunk.wrap(file.source, data)
        cursor = uproot.source.cursor.Cursor(0)
        obj = model.read(chunk, cursor, {"breadcrumbs": ()}, file, file, None)
        assert cursor.index == len(data)
        return cls(obj, chunk, file)


def collect_samples(obj, chunk, file, samples):
    if type(obj) in MODELS:
        samples.setdefault(type(obj), Sample(obj, chunk, file))
    for part in (*obj.bases, *obj.members.values()):
        if isinstance(part, uproot.model.Model):
            collect_samples(part, chunk, file, samples)


def member_spans(sample):
    """
    Reads one object with read_member_n, as memberwise deserialization would, and
    returns it with the (start, stop) range of bytes that each member occupied.
    """
    cls = type(sample.obj)
    context = {"breadcrumbs": ()}
    cursor = sample.cursor
    obj = cls.read(
        sample.chunk,
        cursor.copy(),
        {"breadcrumbs": (), "reading": False},
        sample.file,
        sample.file,
        None,
    )
    uproot.deserialization.numbytes_version(sample.chunk, cursor, context)
    spans = []
    for index in range(len(cls.base_names_versions) + len(cls.member_names)):
        start = cursor.index
        obj.read_member_n(sample.chunk, cursor, context, sample.file, index)
        spans.append((start, cursor.index))
    return obj, spans, cursor.index


def assert_same(actual, expected, ignore=()):
    assert type(actual) is type(expected)
    assert [type(x) for x in actual.bases] == [type(x) for x in expected.bases]
    assert list(actual.members) == list(expected.members)
    actual_json = actual.tojson()
    expected_json = expected.tojson()
    for name in ignore:
        del actual_json[name], expected_json[name]
    assert actual_json == expected_json


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    path = tmp_path_factory.mktemp("cov_hist") / "histograms.root"
    with uproot.recreate(path, compression=None) as file:
        write_all_histogram_types(file)

    found = {}
    with (
        uproot.open(path) as written,
        uproot.open(skhep_testdata.data_path("uproot-issue-240.root")) as graphs,
    ):
        for directory in (written, graphs):
            for name, classname in directory.classnames(
                recursive=True, cycle=False
            ).items():
                if classname == "TDirectory":
                    continue
                key = directory.key(name)
                chunk, _ = key.get_uncompressed_chunk_cursor()
                collect_samples(key.get(), chunk, directory.file, found)

        # Neither ROOT's TAttLine and TAttFill version 1 (old files) nor TGraphErrors
        # (Uproot can't write it) are in these files: they are made of the same parts.
        file = written.file
        for old, new in (
            (
                uproot.models.TAtt.Model_TAttLine_v1,
                uproot.models.TAtt.Model_TAttLine_v2,
            ),
            (
                uproot.models.TAtt.Model_TAttFill_v1,
                uproot.models.TAtt.Model_TAttFill_v2,
            ),
        ):
            found[old] = Sample.from_bytes(
                old, serialized(found[new].data[6:], 1), file
            )
        tgraph = found[uproot.models.TGraph.Model_TGraph_v4]
        errors = numpy.arange(tgraph.obj.member("fNpoints"), dtype=">f8")
        found[uproot.models.TGraph.Model_TGraphErrors_v3] = Sample.from_bytes(
            uproot.models.TGraph.Model_TGraphErrors_v3,
            serialized(
                tgraph.data
                + b"\x01"
                + (0.5 + errors).tobytes()
                + b"\x01"
                + (1.5 + errors).tobytes(),
                3,
            ),
            file,
        )
        yield found


def serialized(payload, version):
    return uproot.serialization.numbytes_version(len(payload), version) + payload


def test_every_model_has_a_sample(samples):
    assert set(samples) == set(MODELS)


@pytest.mark.parametrize("cls", MEMBERWISE_MODELS, ids=name_of)
def test_read_member_n_reproduces_read_members(samples, cls):
    sample = samples[cls]
    obj, _, stop = member_spans(sample)
    assert_same(obj, sample.obj)
    assert stop == sample.obj.cursor.index + sample.obj.num_bytes


@pytest.mark.parametrize(
    "cls, arrays",
    [
        (uproot.models.TH.Model_TH1_v8, ["fBuffer"]),
        (uproot.models.TGraph.Model_TGraph_v4, ["fX", "fY"]),
        (uproot.models.TGraph.Model_TGraphErrors_v3, ["fEX", "fEY"]),
        (
            uproot.models.TGraph.Model_TGraphAsymmErrors_v3,
            ["fEXlow", "fEXhigh", "fEYlow", "fEYhigh"],
        ),
    ],
    ids=lambda x: x.__name__ if isinstance(x, type) else "",
)
def test_no_speedbump(samples, cls, arrays):
    sample = samples[cls]
    data = without_speedbumps(sample)
    assert len(data) < sample.obj.num_bytes

    context = {"breadcrumbs": (), "speedbump": False}
    chunk = uproot.source.chunk.Chunk.wrap(sample.file.source, data)
    cursor = uproot.source.cursor.Cursor(0)
    obj = cls.read(chunk, cursor.copy(), context, sample.file, sample.file, None)
    ignore = ("fHistogram",) if "TGraph" in cls.__name__ else ()
    assert_same(obj, sample.obj, ignore)

    empty = cls.read(
        chunk, cursor.copy(), dict(context, reading=False), None, sample.file, None
    )
    uproot.deserialization.numbytes_version(chunk, cursor, context)
    num_bases = len(cls.base_names_versions)
    for index in range(num_bases + len(cls.member_names)):
        empty.read_member_n(chunk, cursor, context, sample.file, index)
    assert cursor.index == len(data)
    assert_same(empty, sample.obj, ignore)
    for name in arrays:
        assert numpy.array_equal(empty.member(name), sample.obj.member(name))


def without_speedbumps(sample):
    cls = type(sample.obj)
    _, spans, _ = member_spans(sample)
    num_bases = len(cls.base_names_versions)
    payload = []
    for index, (start, stop) in enumerate(spans):
        if index < num_bases:
            base = sample.obj.bases[index]
            if isinstance(
                base,
                (uproot.models.TH.Model_TH1_v8, uproot.models.TGraph.Model_TGraph_v4),
            ):
                payload.append(
                    without_speedbumps(Sample(base, sample.chunk, sample.file))
                )
                continue
        elif cls.member_names[index - num_bases] == "fHistogram":
            # a TH1F whose own speedbump would need removing: make it a null pointer
            payload.append(b"\x00\x00\x00\x00")
            continue
        elif isinstance(
            sample.obj.member(cls.member_names[index - num_bases]), numpy.ndarray
        ):
            start += 1
        payload.append(bytes(sample.chunk.raw_data[start:stop]))
    return serialized(b"".join(payload), sample.obj.instance_version)


@pytest.mark.parametrize("cls", MODELS, ids=name_of)
def test_memberwise_bit_is_not_implemented(samples, cls):
    sample = samples[cls]
    data = bytearray(sample.chunk.raw_data.tobytes())
    data[sample.obj.cursor.index + 4] |= uproot.const.kStreamedMemberWise >> 8
    chunk = uproot.source.chunk.Chunk.wrap(sample.file.source, bytes(data))

    message = f"memberwise serialization of {cls.__name__}\nin file {re.escape(sample.file.file_path)}"
    with pytest.raises(NotImplementedError, match=message):
        cls.read(
            chunk, sample.cursor, {"breadcrumbs": ()}, sample.file, sample.file, None
        )


@pytest.mark.parametrize("cls", MODELS, ids=name_of)
def test_forth_is_refused(samples, cls):
    sample = samples[cls]
    generator = uproot._awkwardforth.ForthGenerator(None)
    context = {"breadcrumbs": (), "forth": types.SimpleNamespace(gen=generator)}
    with pytest.raises(CannotBeForth):
        sample.obj.read_members(sample.chunk, sample.cursor, context, sample.file)


def test_histogram_in_a_ttree():
    path = skhep_testdata.data_path("uproot-issue-tbranch-of-th2.root")
    with uproot.open(path) as file:
        branch = file["g4SimHits/tree/histogram"]
        interpretation = branch.interpretation
        assert isinstance(interpretation, uproot.interpretation.objects.AsObjects)

        hist = branch.array(library="np", entry_stop=1)[0]
        assert isinstance(hist, uproot.models.TH.Model_TH2F_v4)
        assert hist.values().sum() == pytest.approx(39210.28)

        with pytest.raises(ValueError, match="cannot produce Awkward Arrays"):
            branch.array(library="ak", entry_stop=1)


class StandIn(uproot.model.Model):
    """
    A class that can be strided and has an Awkward form, to replace the classes that
    a generated class is composed of. In a real file, TNamed, TString, and TList stop
    ``strided_interpretation`` and ``awkward_form`` from getting past them, and they
    can't be replaced with ``custom_classes``.
    """

    calls = []

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        cls.calls.append((cls.root_name, header, tobject_header, breadcrumbs))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, [(None, None), (cls.part, numpy.dtype(">i4"))]
        )

    @classmethod
    def awkward_form(cls, file, context):
        cls.calls.append(
            (
                cls.root_name,
                context["header"],
                context["tobject_header"],
                context["breadcrumbs"],
            )
        )
        return awkward.forms.RecordForm(
            [uproot._util.awkward_form(numpy.dtype("i4"), file, context)],
            [cls.part],
        )


STAND_INS = {
    name: type(f"Model_{name}", (StandIn,), {"root_name": name, "part": f"{name}_part"})
    for name in (
        "TNamed",
        "TString",
        "TList",
        "TAxis",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAtt3D",
        "TAttAxis",
        "TArrayC",
        "TArrayS",
        "TArrayI",
        "TArrayF",
        "TArrayD",
        "TH1",
        "TH2",
        "TH3",
        "TH1D",
        "TH2D",
        "TH3D",
        "TGraph",
    )
}


class StandInFile:
    """
    The only part of a file that the methods of a generated class use.
    """

    def __init__(self):
        StandIn.calls.clear()

    def class_named(self, classname, version=None):
        return STAND_INS[classname]


@pytest.fixture
def stand_in_file():
    return StandInFile()


def expected_names(cls, header):
    names = ["@num_bytes", "@instance_version"] if header else []
    names.extend(f"{name}_part" for name, _ in cls.base_names_versions)
    return names


def assert_calls(cls, header, breadcrumbs_of):
    called = [call[0] for call in StandIn.calls]
    bases = [name for name, _ in cls.base_names_versions]
    assert called[: len(bases)] == bases
    for _, call_header, tobject_header, breadcrumbs in StandIn.calls:
        assert call_header is header
        assert tobject_header is True
        assert breadcrumbs_of(breadcrumbs)[-1] is cls


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize(
    "cls", [cls for cls in MODELS if cls not in STRIDED_BLOCKER], ids=name_of
)
def test_strided_interpretation_layout(stand_in_file, cls, header):
    interpretation = cls.strided_interpretation(
        stand_in_file, header=header, original=cls
    )
    assert interpretation.model is cls
    assert interpretation.original is cls
    assert [name for name, _ in interpretation.members] == [
        *expected_names(cls, header),
        *cls.member_names,
    ]
    assert_calls(cls, header, lambda breadcrumbs: breadcrumbs)

    scalars = [
        member
        for name, member in interpretation.members
        if name in cls.member_names and isinstance(member, numpy.dtype)
    ]
    assert scalars == scalar_dtypes(cls)
    if header:
        assert interpretation.members[:2] == [
            ("@num_bytes", numpy.dtype(">u4")),
            ("@instance_version", numpy.dtype(">u2")),
        ]


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("cls", list(STRIDED_BLOCKER), ids=name_of)
def test_strided_interpretation_stops_at_unstridable_member(stand_in_file, cls, header):
    with pytest.raises(CannotBeStrided, match=re.escape(STRIDED_BLOCKER[cls])):
        cls.strided_interpretation(stand_in_file, header=header)
    assert_calls(cls, header, lambda breadcrumbs: breadcrumbs)


@pytest.mark.parametrize(
    "cls", [cls for cls in MODELS if cls.base_names_versions], ids=name_of
)
def test_strided_interpretation_of_the_real_classes_stops_at_tnamed(samples, cls):
    with pytest.raises(CannotBeStrided, match="^TNamed$"):
        cls.strided_interpretation(samples[cls].file)


@pytest.mark.parametrize(
    "cls", [cls for cls in MODELS if cls not in MODULE_FORMATS], ids=name_of
)
def test_recursive_models_are_refused(samples, cls):
    file = samples[cls].file
    with pytest.raises(CannotBeStrided, match="the depth of instances is unbounded"):
        cls.strided_interpretation(file, breadcrumbs=(cls,))
    context = {
        "index_format": "i64",
        "header": False,
        "tobject_header": True,
        "breadcrumbs": (cls,),
    }
    with pytest.raises(CannotBeAwkward, match="the depth of instances is unbounded"):
        cls.awkward_form(file, context)


@pytest.mark.parametrize(
    "cls",
    [
        *MODULE_FORMATS,
        uproot.models.TAtt.Model_TAttAxis_v4,
        uproot.models.TAtt.Model_TAtt3D_v1,
    ],
    ids=name_of,
)
def test_strided_dtype_decodes_serialized_bytes(samples, cls):
    sample = samples[cls]
    interpretation = cls.strided_interpretation(sample.file, header=True)
    assert interpretation.from_dtype.itemsize == sample.obj.num_bytes

    array = numpy.frombuffer(sample.data, dtype=interpretation.from_dtype)
    assert (
        array["@num_bytes"][0]
        == (sample.obj.num_bytes - 4) | uproot.const.kByteCountMask
    )
    assert array["@instance_version"][0] == uproot.model.classname_version(cls.__name__)
    for name in cls.member_names:
        assert array[name][0] == pytest.approx(sample.obj.member(name))


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("cls", MODELS, ids=name_of)
def test_awkward_form_layout(stand_in_file, cls, header):
    context = {
        "index_format": "i64",
        "header": header,
        "tobject_header": True,
        "breadcrumbs": (),
    }
    form = cls.awkward_form(stand_in_file, context)
    assert form.parameters["__record__"] == root_name(cls)
    assert form.fields == [
        *expected_names(cls, header),
        *(x for x in cls.member_names if x not in POINTERS.get(cls, ())),
    ]
    assert context["breadcrumbs"] == ()
    assert_calls(cls, header, lambda breadcrumbs: breadcrumbs)

    if header:
        assert form.content("@num_bytes").primitive == "uint32"
        assert form.content("@instance_version").primitive == "uint16"

    scalars = [
        numpy.dtype(form.content(name).primitive)
        for name in cls.member_names
        if name in form.fields
        and isinstance(form.content(name), awkward.forms.NumpyForm)
    ]
    assert [x.itemsize for x in scalars] == [x.itemsize for x in scalar_dtypes(cls)]
    assert [x.kind for x in scalars] == [x.kind for x in scalar_dtypes(cls)]


@pytest.mark.parametrize(
    "cls, arrays",
    [
        (uproot.models.TH.Model_TH1_v8, ["fBuffer"]),
        (uproot.models.TGraph.Model_TGraph_v4, ["fX", "fY"]),
        (uproot.models.TGraph.Model_TGraphErrors_v3, ["fEX", "fEY"]),
        (
            uproot.models.TGraph.Model_TGraphAsymmErrors_v3,
            ["fEXlow", "fEXhigh", "fEYlow", "fEYhigh"],
        ),
    ],
    ids=lambda x: x.__name__ if isinstance(x, type) else "",
)
def test_awkward_form_of_dynamic_arrays(stand_in_file, cls, arrays):
    context = {
        "index_format": "i32",
        "header": False,
        "tobject_header": True,
        "breadcrumbs": (),
    }
    form = cls.awkward_form(stand_in_file, context)
    for name in arrays:
        assert isinstance(form.content(name), awkward.forms.ListOffsetForm)
        assert form.content(name).offsets == "i32"
        assert form.content(name).content.primitive == "float64"


def test_awkward_form_of_the_real_classes(samples):
    file = samples[uproot.models.TH.Model_TAxis_v10].file
    context = {
        "index_format": "i64",
        "header": False,
        "tobject_header": False,
        "breadcrumbs": (),
    }

    # a histogram's fFunctions has no form
    with pytest.raises(CannotBeAwkward, match="^TList$"):
        uproot.models.TH.Model_TH1F_v3.awkward_form(file, context)

    # but an axis has one, with all of the fields of its bases
    form = uproot.models.TH.Model_TAxis_v10.awkward_form(file, context)
    assert form.fields[:2] == ["fName", "fTitle"]
    assert form.fields[2:13] == [
        "fNdivisions",
        "fAxisColor",
        "fLabelColor",
        "fLabelFont",
        "fLabelOffset",
        "fLabelSize",
        "fTickLength",
        "fTitleOffset",
        "fTitleSize",
        "fTitleColor",
        "fTitleFont",
    ]
    assert form.content("fXbins").content.primitive == "float64"
    assert form.content("fTimeFormat").parameters["__array__"] == "string"


@pytest.mark.parametrize("cls", WRITABLE_MODELS, ids=name_of)
def test_serialize_round_trip(samples, cls):
    sample = samples[cls]
    writable = sample.obj.to_writable()
    out = []
    writable._serialize(out, True, None, numpy.uint32(0))
    # the TGraph from ROOT has a fHistogram, which writing doesn't preserve
    ignore = ("fBits", "fHistogram") if cls.__name__.startswith("Model_TGraph") else ()
    copy = Sample.from_bytes(cls, b"".join(out), sample.file).obj
    assert_same(copy, sample.obj, ignore)


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in WRITABLE_MODELS
        if cls is not uproot.models.TH.Model_TH1_v8  # passes header down to its axes
    ],
    ids=name_of,
)
def test_serialize_header_is_num_bytes_and_version(samples, cls):
    writable = samples[cls].obj.to_writable()
    with_header, without_header = [], []
    writable._serialize(with_header, True, None, numpy.uint32(0))
    writable._serialize(without_header, False, None, numpy.uint32(0))
    data = b"".join(with_header)
    assert data[6:] == b"".join(without_header)
    assert data[:6] == struct.pack(
        ">IH",
        (len(data) - 4) | uproot.const.kByteCountMask,
        uproot.model.classname_version(cls.__name__),
    )


@pytest.mark.parametrize(
    "cls",
    [
        uproot.models.TGraph.Model_TGraphErrors_v3,
        uproot.models.TGraph.Model_TGraphAsymmErrors_v3,
    ],
    ids=name_of,
)
def test_tgraph_with_errors_can_not_be_written(samples, cls, tmp_path):
    # documented in uproot.as_TGraph
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(NotImplementedError, match="FIXME"):
            file["graph"] = samples[cls].obj


def test_objects_in_a_ttree_written_by_root(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = str(tmp_path / "objects.root")
    file = ROOT.TFile(path, "recreate")
    tree = ROOT.TTree("tree", "tree")
    x = numpy.array([1.0, 2.0, 3.0])
    y = numpy.array([4.0, 5.0, 6.0])
    objects = {
        "line": ROOT.TAttLine(2, 3, 4),
        "fill": ROOT.TAttFill(5, 1001),
        "taxis": ROOT.TAxis(3, 0.0, 3.0),
        "errors": ROOT.TGraphErrors(3, x, y, y, x),
        "asymm": ROOT.TGraphAsymmErrors(3, x, y, x, x, y, y),
    }
    for name, value in objects.items():
        tree.Branch(name, value, 32000, 0)
    for color in (10, 11, 12):
        objects["line"].SetLineColor(color)
        tree.Fill()
    tree.Write()
    file.Close()

    with uproot.open(path) as opened:
        tree = opened["tree"]

        # strided: their members have a fixed size
        line = tree["line"]
        assert isinstance(
            line.interpretation, uproot.interpretation.objects.AsStridedObjects
        )
        assert line.array(library="ak")["fLineColor"].tolist() == [10, 11, 12]
        assert line.array(library="ak")["fLineWidth"].tolist() == [4, 4, 4]
        assert line.array(library="np")[1].member("fLineColor") == 11
        fill = tree["fill"].array(library="ak")
        assert fill["fFillColor"].tolist() == [5, 5, 5]
        assert fill["fFillStyle"].tolist() == [1001, 1001, 1001]

        # not strided: objects with an Awkward form, but no AwkwardForth
        taxis = tree["taxis"]
        assert taxis.array(library="ak")["fXmax"].tolist() == [3.0, 3.0, 3.0]
        assert taxis.interpretation._forth is False

        errors = tree["errors"].array(library="ak")
        assert errors["fEX"].tolist() == [[4.0, 5.0, 6.0]] * 3
        assert errors["fEY"].tolist() == [[1.0, 2.0, 3.0]] * 3
        asymm = tree["asymm"].array(library="np")[0]
        assert asymm.values("y").tolist() == [4.0, 5.0, 6.0]
        assert asymm.errors("high", "y").tolist() == [4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    "name, writable_class",
    [
        ("hpx", uproot.models.TH.Model_TH1F_v3),
        ("hpxpy", uproot.models.TH.Model_TH2F_v4),
    ],
)
def test_old_histogram_versions_are_updated_for_writing(name, writable_class):
    path = skhep_testdata.data_path("uproot-hepdata-example.root")
    with uproot.open(path) as file:
        old = file[name]
        writable = old.to_writable()

    assert type(writable) is writable_class
    assert "fModLabs" not in old.member("fXaxis").members
    assert writable.member("fXaxis").member("fModLabs") is None
    assert writable.member("fStatOverflows") == 0


def test_old_histogram_round_trip(tmp_path):
    path = skhep_testdata.data_path("uproot-hepdata-example.root")
    with uproot.open(path) as file:
        old = file["hpxpy"]
        values = old.values(flow=True)
        edges = old.axis(0).edges()

    with uproot.recreate(tmp_path / "updated.root") as file:
        file["hpxpy"] = old

    with uproot.open(tmp_path / "updated.root") as file:
        new = file["hpxpy"]
        assert type(new) is uproot.models.TH.Model_TH2F_v4
        assert numpy.array_equal(new.values(flow=True), values)
        assert numpy.array_equal(new.axis(0).edges(), edges)


def test_th1_serialize_passes_its_header_flag_to_its_members(samples):
    # This pins what the generated code does: no writer serializes a TH1 without
    # its header, and the TH1 reader always expects the members' headers.
    writable = samples[uproot.models.TH.Model_TH1_v8].obj.to_writable()
    with_header, without_header = [], []
    writable._serialize(with_header, True, None, numpy.uint32(0))
    writable._serialize(without_header, False, None, numpy.uint32(0))
    # its own, the three axes, and fFunctions
    assert len(b"".join(with_header)) - len(b"".join(without_header)) == 5 * 6


def test_th1_without_speedbump_is_written_without_it(samples):
    sample = samples[uproot.models.TH.Model_TH1_v8]
    context = {"breadcrumbs": (), "speedbump": False}

    def read(data):
        chunk = uproot.source.chunk.Chunk.wrap(sample.file.source, data)
        cursor = uproot.source.cursor.Cursor(0)
        return uproot.models.TH.Model_TH1_v8.read(
            chunk, cursor, dict(context), sample.file, sample.file, None
        )

    out = []
    read(without_speedbumps(sample)).to_writable()._serialize(
        out, True, None, numpy.uint32(0)
    )
    assert_same(read(b"".join(out)), sample.obj)


def test_read_member_n_of_a_class_without_members_reads_nothing(samples):
    sample = samples[uproot.models.TAtt.Model_TAtt3D_v1]
    cursor = sample.cursor
    start = cursor.index
    sample.obj.read_member_n(sample.chunk, cursor, {}, sample.file, 0)
    assert cursor.index == start
    assert sample.obj.members == {}
