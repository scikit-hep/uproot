# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a series of rules for identifying Python objects that can be written
to ROOT files and preparing them for writing.

The :doc:`uproot.writing.identify.add_to_directory` function is the most general in that
it recognizes data that could be converted into a TTree or into a normal object. It also
adds the data to a :doc:`uproot.writing.writable.WritableDirectory`.

The :doc:`uproot.writing.identify.to_writable` function recognizes all static objects
(everything but TTrees) as a series of rules, returning a :doc:`uproot.model.Model`
but not adding it to any :doc:`uproot.writing.writable.WritableDirectory`.

The (many) other functions in this module construct writable :doc:`uproot.model.Model`
objects from Python builtins and other writable models.
"""

from __future__ import annotations

from collections.abc import Mapping

import awkward
import numpy

import uproot.compression
import uproot.extras
import uproot.pyroot
import uproot.writing
import uproot.writing._cascade
import uproot.writing._cascadentuple
import uproot.writing.writable


def add_to_directory(obj, name, directory, streamers):
    """
    Args:
        obj: Object to attempt to recognize as something that can be written to a
            ROOT file.
        name (str or None): Name to assign to the writable object.
        directory (:doc:`uproot.writing.writable.WritableDirectory`): Directory to
            add the object to, if successful.
        streamers (list of :doc:`uproot.streamers.Model_TStreamerInfo`, :doc:`uproot.writable._cascade.RawStreamerInfo`, or constructor arguments for the latter): Collects
            streamers to add to the output file so that all objects in it can be read
            in any version of ROOT.

    This function performs two tasks: it attempts to recognize ``obj`` as a writable
    object and, if successful, writes it to a ``directory``.

    It can recognize dynamic TTrees and static objects such as histograms. For only
    static objects, without the additional concern of writing the resulting object
    in a ``directory``, see :doc:`uproot.writing.identify.to_writable`.

    Raises ``TypeError`` if ``obj`` is not recognized as writable data.
    """
    obj = uproot.writing._cascadentuple._regularize_input_type_to_awkward(obj)

    if isinstance(obj, Mapping) and all(isinstance(x, str) for x in obj):
        metadata, data = uproot.writing.writable._unpack_metadata_and_arrays(obj)
        rntuple = directory.mkrntuple(name, metadata)
        rntuple.extend(data)
    elif isinstance(obj, (awkward.Array, awkward.contents.Content, awkward.forms.Form)):
        directory.mkrntuple(name, obj)
    else:
        writable = to_writable(obj)

        for rawstreamer in writable.class_rawstreamers:
            if isinstance(rawstreamer, tuple):
                streamers.append(uproot.writing._cascade.RawStreamerInfo(*rawstreamer))
            else:
                streamers.append(rawstreamer)

        uncompressed_data = writable.serialize(name=name)
        compressed_data = uproot.compression.compress(
            uncompressed_data, directory.file.compression
        )

        if hasattr(writable, "fTitle"):
            title = writable.fTitle
        elif writable.has_member("fTitle"):
            title = writable.member("fTitle")
        else:
            title = ""

        directory._cascading.add_object(
            directory.file.sink,
            writable.classname,
            name,
            title,
            compressed_data,
            len(uncompressed_data),
        )


def to_writable(obj):
    """
    Converts arbitrary Python object ``obj`` to a writable :doc:`uproot.model.Model`
    if possible; raises ``TypeError`` otherwise.

    This function is a series of rules that defines what Python data can or cannot be
    written to ROOT files. For instance, a 2-tuple of NumPy arrays with the appropriate
    dimensions is recognized as a histogram, since NumPy's ``np.histogram`` function
    produces such objects.

    This series of rules is expected to grow with time.
    """
    # This is turns histogramdd-style into histogram2d-style.
    if (
        isinstance(obj, (tuple, list))
        and 2 <= len(obj) <= 3  # might have a histogram title as the last item
        and isinstance(obj[0], numpy.ndarray)
        and isinstance(obj[1], (tuple, list))
        and 2 <= len(obj[1]) <= 3  # 2D or 3D
        and isinstance(obj[1][0], numpy.ndarray)
        and isinstance(obj[1][1], numpy.ndarray)
        and (len(obj[1]) == 2 or isinstance(obj[1][2], numpy.ndarray))
        and all(len(x.shape) == 1 for x in obj[1])
        and len(obj[0].shape) == len(obj[1])
    ):
        obj = (obj[0], *tuple(obj[1]), *tuple(obj[2:]))

    # This is the big if-elif-else chain of rules
    if isinstance(obj, uproot.model.Model):
        return obj.to_writable()

    elif type(obj).__module__ == "cppyy.gbl":
        import ROOT

        if isinstance(obj, ROOT.TObject):
            return uproot.pyroot._PyROOTWritable(obj)
        else:
            raise TypeError(
                f"only instances of TObject can be written to files, not {type(obj).__name__}"
            )

    elif isinstance(obj, str):
        return to_TObjString(obj)

    elif (
        hasattr(obj, "axes")
        and hasattr(obj, "kind")
        and hasattr(obj, "values")
        and hasattr(obj, "variances")
        and hasattr(obj, "counts")
    ):
        # boost_histogram is used in _fXbins_maybe_regular *if* this is such a type
        boost_histogram = None
        if (
            type(obj).__module__ == "boost_histogram"
            or type(obj).__module__.startswith("boost_histogram.")
            or type(obj).__module__ == "hist"
            or type(obj).__module__.startswith("hist.")
        ):
            import boost_histogram

        if obj.kind != "COUNT" and obj.kind != "MEAN":
            raise ValueError(
                "PlottableHistogram can only be converted to ROOT TH* if kind='COUNT' or 'MEAN'"
            )

        ndim = len(obj.axes)
        if not 1 <= ndim <= 3:
            raise ValueError(
                "PlottableHistogram can only be converted to ROOT TH* if it has between 1 and 3 axes (TH1, TH2, TH3)"
            )

        title = getattr(obj, "title", getattr(obj, "name", ""))
        if title is None:
            title = ""

        try:
            # using flow=True if supported
            data = obj.values(flow=True)
            fSumw2 = (
                obj.variances(flow=True)
                if boost_histogram is None
                or obj.storage_type == boost_histogram.storage.Weight
                else None
            )

            # and flow=True is different from flow=False (obj actually has flow bins)
            data_noflow = obj.values(flow=False)
            for flow, noflow in zip(data.shape, data_noflow.shape, strict=True):
                if flow != noflow + 2:
                    raise TypeError

        except TypeError:
            # flow=True is not supported, fallback to allocate-and-fill

            tmp = data_noflow = obj.values()
            s = tmp.shape
            d = tmp.dtype.newbyteorder(">")
            if ndim == 1:
                data = numpy.zeros(s[0] + 2, dtype=d)
                data[1:-1] = tmp
            elif ndim == 2:
                data = numpy.zeros((s[0] + 2, s[1] + 2), dtype=d)
                data[1:-1, 1:-1] = tmp
            elif ndim == 3:
                data = numpy.zeros((s[0] + 2, s[1] + 2, s[2] + 2), dtype=d)
                data[1:-1, 1:-1, 1:-1] = tmp

            tmp = (
                obj.variances()
                if boost_histogram is None
                or obj.storage_type == boost_histogram.storage.Weight
                else None
            )
            fSumw2 = None
            if tmp is not None:
                s = tmp.shape
                if ndim == 1:
                    fSumw2 = numpy.zeros(s[0] + 2, dtype=">f8")
                    fSumw2[1:-1] = tmp
                elif ndim == 2:
                    fSumw2 = numpy.zeros((s[0] + 2, s[1] + 2), dtype=">f8")
                    fSumw2[1:-1, 1:-1] = tmp
                elif ndim == 3:
                    fSumw2 = numpy.zeros((s[0] + 2, s[1] + 2, s[2] + 2), dtype=">f8")
                    fSumw2[1:-1, 1:-1, 1:-1] = tmp

        else:
            # continuing to use flow=True, because it is supported
            data = data.astype(data.dtype.newbyteorder(">"))
            if fSumw2 is not None:
                fSumw2 = fSumw2.astype(">f8")

        # we're assuming the PlottableHistogram ensures data.shape == weights.shape
        if fSumw2 is not None:
            assert data.shape == fSumw2.shape

        # data are stored in transposed order for 2D and 3D
        data = data.T.reshape(-1)
        if fSumw2 is not None:
            fSumw2 = fSumw2.T.reshape(-1)

        # ROOT has fEntries = sum *without* weights, *with* flow bins
        fEntries = data.sum()

        # convert all axes in one list comprehension
        axes = [
            to_TAxis(
                fName=default_name,
                fTitle=getattr(axis, "label", getattr(obj, "name", "")),
                fNbins=len(axis),
                fXmin=axis.edges[0],
                fXmax=axis.edges[-1],
                fXbins=_fXbins_maybe_regular(axis, boost_histogram),
                fLabels=_fLabels_maybe_categorical(axis, boost_histogram),
            )
            for axis, default_name in zip(
                obj.axes, ["xaxis", "yaxis", "zaxis"], strict=False
            )
        ]

        # make TH1, TH2, TH3 types independently
        if len(axes) == 1:
            if obj.kind == "MEAN":
                if hasattr(obj, "storage_type"):
                    obj_sum = obj.sum()
                    _view_flow = obj.view(flow=True)

                    if obj.storage_type is boost_histogram.storage.WeightedMean:
                        _sumw = numpy.asarray(
                            _view_flow["sum_of_weights"], dtype=numpy.float64
                        )
                        _sumw2 = numpy.asarray(
                            _view_flow["sum_of_weights_squared"], dtype=numpy.float64
                        )
                        _value = numpy.asarray(_view_flow["value"], dtype=numpy.float64)
                        _sum_sq_dev = numpy.asarray(
                            _view_flow["_sum_of_weighted_deltas_squared"],
                            dtype=numpy.float64,
                        )
                        fTsumw = obj_sum["sum_of_weights"]
                        fTsumw2 = obj_sum["sum_of_weights_squared"]
                        fTsumwy = obj_sum["value"] * obj_sum["sum_of_weights"]
                        fTsumwy2 = obj_sum["_sum_of_weighted_deltas_squared"] + (
                            obj_sum["value"] ** 2 * obj_sum["sum_of_weights"]
                            if obj_sum["sum_of_weights"] > 0
                            else 0
                        )
                    else:
                        _sumw = numpy.asarray(_view_flow["count"], dtype=numpy.float64)
                        _sumw2 = _sumw  # unweighted: sum(w^2) = sum(1^2) = count
                        _value = numpy.asarray(_view_flow["value"], dtype=numpy.float64)
                        _sum_sq_dev = numpy.asarray(
                            _view_flow["_sum_of_deltas_squared"], dtype=numpy.float64
                        )
                        fTsumw = obj_sum["count"]
                        fTsumw2 = obj_sum["count"]
                        fTsumwy = obj_sum["value"] * obj_sum["count"]
                        fTsumwy2 = obj_sum["_sum_of_deltas_squared"] + (
                            obj_sum["value"] ** 2 * obj_sum["count"]
                            if obj_sum["count"] > 0
                            else 0
                        )

                    # ROOT TProfile convention: fSumw2 = sum(w*y^2) = sum_sq_dev + sum(w)*mean^2
                    fSumw2 = _sum_sq_dev + _sumw * _value**2

                    fEntries = (
                        obj.metadata.get("fEntries", fTsumw)
                        if ("metadata" in obj.__dir__() and obj.metadata is not None)
                        else fTsumw
                    )

                    _data = _sumw * _value
                    _fBinEntries = _sumw
                    _fBinSumw2 = _sumw2

                    # Categorical axes in boost-histogram do not have underflow.
                    # ROOT histograms always have underflow (bin 0) AND overflow.
                    # boost-histogram category axis view already includes overflow if flow=True.
                    # We prepend a zero to all arrays to align boost-histogram bin 0 with ROOT bin 1.
                    # We also append a zero for the ROOT overflow bin if needed.
                    if hasattr(obj.axes[0], "traits") and getattr(
                        obj.axes[0].traits, "discrete", False
                    ):
                        # boost-histogram categorical axis with flow=True has nbins + 1 (data + overflow)
                        # ROOT expects nbins + 2 (underflow + data + overflow)
                        _data = numpy.insert(_data, 0, 0)
                        _fBinEntries = numpy.insert(_fBinEntries, 0, 0)
                        _fBinSumw2 = numpy.insert(_fBinSumw2, 0, 0)
                        fSumw2 = numpy.insert(fSumw2, 0, 0)

                        if len(_fBinEntries) == len(obj.axes[0]) + 1:
                            _data = numpy.append(_data, 0)
                            _fBinEntries = numpy.append(_fBinEntries, 0)
                            _fBinSumw2 = numpy.append(_fBinSumw2, 0)
                            fSumw2 = numpy.append(fSumw2, 0)

                        # Now _fBinEntries is nbins + 2.
                        # Edges should be nbins + 3 to have nbins + 2 centers.
                        _edges = numpy.arange(
                            -1, len(obj.axes[0]) + 2, dtype=numpy.float64
                        )
                    else:
                        # Regular axis: prepend and append a bin width to edges for flow bins.
                        _edges = obj.axes[0].edges
                        width = _edges[1] - _edges[0]
                        _edges = numpy.insert(_edges, 0, _edges[0] - width)
                        _edges = numpy.append(_edges, _edges[-1] + width)

                    _fTsumw, _fTsumw2, fTsumwx, fTsumwx2 = _root_stats_1d(
                        _fBinEntries, _edges
                    )

                    return to_TProfile(
                        fName=None,
                        fTitle=title,
                        data=_data,
                        fEntries=fEntries,
                        fTsumw=fTsumw,
                        fTsumw2=fTsumw2,
                        fTsumwx=fTsumwx,
                        fTsumwx2=fTsumwx2,
                        fTsumwy=fTsumwy,
                        fTsumwy2=fTsumwy2,
                        fSumw2=fSumw2,
                        fBinEntries=_fBinEntries,
                        fBinSumw2=_fBinSumw2,
                        fXaxis=axes[0],
                    )
                else:
                    return to_TProfile(
                        fName=None,
                        fTitle=title,
                        data=obj._bases[0]._bases[-1],
                        fEntries=obj.member("fEntries"),
                        fTsumw=obj.member("fTsumw"),
                        fTsumw2=obj.member("fTsumw2"),
                        fTsumwx=obj.member("fTsumwx"),
                        fTsumwx2=obj.member("fTsumwx2"),
                        fTsumwy=obj.member("fTsumwy"),
                        fTsumwy2=obj.member("fTsumwy2"),
                        fSumw2=obj.member("fSumw2"),
                        fBinEntries=obj.member("fBinEntries"),
                        fBinSumw2=obj.member("fBinSumw2"),
                        fXaxis=axes[0],
                    )
            else:
                fTsumw, fTsumw2, fTsumwx, fTsumwx2 = _root_stats_1d(
                    data_noflow, obj.axes[0].edges
                )
                return to_TH1x(
                    fName=None,
                    fTitle=title,
                    data=data,
                    fEntries=fEntries,
                    fTsumw=fTsumw,
                    fTsumw2=fTsumw2,
                    fTsumwx=fTsumwx,
                    fTsumwx2=fTsumwx2,
                    fSumw2=fSumw2,
                    fXaxis=axes[0],
                )

        elif len(axes) == 2:
            if obj.kind == "MEAN":
                raise NotImplementedError(
                    "2D PlottableHistogram with kind='MEAN' (i.e. 2D profile plots) not supported yet"
                )
            else:
                (
                    fTsumw,
                    fTsumw2,
                    fTsumwx,
                    fTsumwx2,
                    fTsumwy,
                    fTsumwy2,
                    fTsumwxy,
                ) = _root_stats_2d(data_noflow, obj.axes[0].edges, obj.axes[1].edges)
                return to_TH2x(
                    fName=None,
                    fTitle=title,
                    data=data,
                    fEntries=fEntries,
                    fTsumw=fTsumw,
                    fTsumw2=fTsumw2,
                    fTsumwx=fTsumwx,
                    fTsumwx2=fTsumwx2,
                    fTsumwy=fTsumwy,
                    fTsumwy2=fTsumwy2,
                    fTsumwxy=fTsumwxy,
                    fSumw2=fSumw2,
                    fXaxis=axes[0],
                    fYaxis=axes[1],
                )

        elif len(axes) == 3:
            if obj.kind == "MEAN":
                raise NotImplementedError(
                    "3D PlottableHistogram with kind='MEAN' (i.e. 3D profile plots) not supported yet"
                )
            else:
                (
                    fTsumw,
                    fTsumw2,
                    fTsumwx,
                    fTsumwx2,
                    fTsumwy,
                    fTsumwy2,
                    fTsumwxy,
                    fTsumwz,
                    fTsumwz2,
                    fTsumwxz,
                    fTsumwyz,
                ) = _root_stats_3d(
                    data_noflow,
                    obj.axes[0].edges,
                    obj.axes[1].edges,
                    obj.axes[2].edges,
                )
                return to_TH3x(
                    fName=None,
                    fTitle=title,
                    data=data,
                    fEntries=fEntries,
                    fTsumw=fTsumw,
                    fTsumw2=fTsumw2,
                    fTsumwx=fTsumwx,
                    fTsumwx2=fTsumwx2,
                    fTsumwy=fTsumwy,
                    fTsumwy2=fTsumwy2,
                    fTsumwxy=fTsumwxy,
                    fTsumwz=fTsumwz,
                    fTsumwz2=fTsumwz2,
                    fTsumwxz=fTsumwxz,
                    fTsumwyz=fTsumwyz,
                    fSumw2=fSumw2,
                    fXaxis=axes[0],
                    fYaxis=axes[1],
                    fZaxis=axes[2],
                )

        else:
            raise AssertionError(f"ndim is 1, 2, or 3 by now, but it is {len(axes)}")

    elif (
        isinstance(obj, (tuple, list))
        and 2 <= len(obj) <= 5  # might have a histogram title as the last item
        and all(isinstance(x, numpy.ndarray) for x in obj[:-1])
        and isinstance(obj[-1], (numpy.ndarray, str))
        and 1 <= len(obj[0].shape) <= 3
        and len(obj[0].shape) == sum(int(isinstance(x, numpy.ndarray)) for x in obj[1:])
        and all(len(x.shape) == 1 for x in obj[1:] if isinstance(x, numpy.ndarray))
    ):
        if isinstance(obj[-1], str):
            obj, title = obj[:-1], obj[-1]
        else:
            title = ""

        if len(obj) == 2:
            entries, edges = obj

            with_flow = numpy.empty(len(entries) + 2, dtype=">f8")
            with_flow[1:-1] = entries
            with_flow[0] = 0
            with_flow[-1] = 0

            fEntries = entries.sum()
            fTsumw, fTsumw2, fTsumwx, fTsumwx2 = _root_stats_1d(entries, edges)

            fNbins = len(edges) - 1
            fXmin, fXmax = edges[0], edges[-1]
            if numpy.allclose(edges, numpy.linspace(fXmin, fXmax, len(edges))):
                edges = numpy.array([], dtype=">f8")
            else:
                edges = edges.astype(">f8")

            return to_TH1x(
                fName=None,
                fTitle=title,
                data=with_flow,
                fEntries=fEntries,
                fTsumw=fTsumw,
                fTsumw2=fTsumw2,
                fTsumwx=fTsumwx,
                fTsumwx2=fTsumwx2,
                fSumw2=None,
                fXaxis=to_TAxis(
                    fName="xaxis",
                    fTitle="",
                    fNbins=fNbins,
                    fXmin=fXmin,
                    fXmax=fXmax,
                    fXbins=edges,
                ),
            )

        elif len(obj) == 3:
            entries, xedges, yedges = obj

            fEntries = entries.sum()
            (
                fTsumw,
                fTsumw2,
                fTsumwx,
                fTsumwx2,
                fTsumwy,
                fTsumwy2,
                fTsumwxy,
            ) = _root_stats_2d(entries, xedges, yedges)

            with_flow = numpy.zeros(
                (entries.shape[0] + 2, entries.shape[1] + 2), dtype=">f8"
            )
            with_flow[1:-1, 1:-1] = entries
            with_flow = with_flow.T.reshape(-1)

            fXaxis_fNbins = len(xedges) - 1
            fXmin, fXmax = xedges[0], xedges[-1]
            if numpy.allclose(xedges, numpy.linspace(fXmin, fXmax, len(xedges))):
                xedges = numpy.array([], dtype=">f8")
            else:
                xedges = xedges.astype(">f8")

            fYaxis_fNbins = len(yedges) - 1
            fYmin, fYmax = yedges[0], yedges[-1]
            if numpy.allclose(yedges, numpy.linspace(fYmin, fYmax, len(yedges))):
                yedges = numpy.array([], dtype=">f8")
            else:
                yedges = yedges.astype(">f8")

            return to_TH2x(
                fName=None,
                fTitle=title,
                data=with_flow,
                fEntries=fEntries,
                fTsumw=fTsumw,
                fTsumw2=fTsumw2,
                fTsumwx=fTsumwx,
                fTsumwx2=fTsumwx2,
                fTsumwy=fTsumwy,
                fTsumwy2=fTsumwy2,
                fTsumwxy=fTsumwxy,
                fSumw2=None,
                fXaxis=to_TAxis(
                    fName="xaxis",
                    fTitle="",
                    fNbins=fXaxis_fNbins,
                    fXmin=fXmin,
                    fXmax=fXmax,
                    fXbins=xedges,
                ),
                fYaxis=to_TAxis(
                    fName="yaxis",
                    fTitle="",
                    fNbins=fYaxis_fNbins,
                    fXmin=fYmin,
                    fXmax=fYmax,
                    fXbins=yedges,
                ),
            )

        elif len(obj) == 4:
            entries, xedges, yedges, zedges = obj

            fEntries = entries.sum()
            (
                fTsumw,
                fTsumw2,
                fTsumwx,
                fTsumwx2,
                fTsumwy,
                fTsumwy2,
                fTsumwxy,
                fTsumwz,
                fTsumwz2,
                fTsumwxz,
                fTsumwyz,
            ) = _root_stats_3d(entries, xedges, yedges, zedges)

            with_flow = numpy.zeros(
                (entries.shape[0] + 2, entries.shape[1] + 2, entries.shape[2] + 2),
                dtype=">f8",
            )
            with_flow[1:-1, 1:-1, 1:-1] = entries
            with_flow = with_flow.T.reshape(-1)

            fXaxis_fNbins = len(xedges) - 1
            fXmin, fXmax = xedges[0], xedges[-1]
            if numpy.allclose(xedges, numpy.linspace(fXmin, fXmax, len(xedges))):
                xedges = numpy.array([], dtype=">f8")
            else:
                xedges = xedges.astype(">f8")

            fYaxis_fNbins = len(yedges) - 1
            fYmin, fYmax = yedges[0], yedges[-1]
            if numpy.allclose(yedges, numpy.linspace(fYmin, fYmax, len(yedges))):
                yedges = numpy.array([], dtype=">f8")
            else:
                yedges = yedges.astype(">f8")

            fZaxis_fNbins = len(zedges) - 1
            fZmin, fZmax = zedges[0], zedges[-1]
            if numpy.allclose(zedges, numpy.linspace(fZmin, fZmax, len(zedges))):
                zedges = numpy.array([], dtype=">f8")
            else:
                zedges = zedges.astype(">f8")

            return to_TH3x(
                fName=None,
                fTitle=title,
                data=with_flow,
                fEntries=fEntries,
                fTsumw=fTsumw,
                fTsumw2=fTsumw2,
                fTsumwx=fTsumwx,
                fTsumwx2=fTsumwx2,
                fTsumwy=fTsumwy,
                fTsumwy2=fTsumwy2,
                fTsumwxy=fTsumwxy,
                fTsumwz=fTsumwz,
                fTsumwz2=fTsumwz2,
                fTsumwxz=fTsumwxz,
                fTsumwyz=fTsumwyz,
                fSumw2=None,
                fXaxis=to_TAxis(
                    fName="xaxis",
                    fTitle="",
                    fNbins=fXaxis_fNbins,
                    fXmin=fXmin,
                    fXmax=fXmax,
                    fXbins=xedges,
                ),
                fYaxis=to_TAxis(
                    fName="yaxis",
                    fTitle="",
                    fNbins=fYaxis_fNbins,
                    fXmin=fYmin,
                    fXmax=fYmax,
                    fXbins=yedges,
                ),
                fZaxis=to_TAxis(
                    fName="zaxis",
                    fTitle="",
                    fNbins=fZaxis_fNbins,
                    fXmin=fZmin,
                    fXmax=fZmax,
                    fXbins=zedges,
                ),
            )

    else:
        raise TypeError(
            "unrecognized type cannot be written to a ROOT file: " + type(obj).__name__
        )


def _fXbins_maybe_regular(axis, boost_histogram):
    if boost_histogram is None:
        edges = axis.edges
        fXmin, fXmax = edges[0], edges[-1]
        if numpy.allclose(edges, numpy.linspace(fXmin, fXmax, len(edges))):
            return numpy.array([], dtype=">f8")
        else:
            return edges.astype(">f8")
    else:
        if (
            isinstance(axis, boost_histogram.axis.Regular)
            and getattr(axis, "transform", None) is None
        ):
            return numpy.array([], dtype=">f8")
        else:
            return axis.edges


def _fLabels_maybe_categorical(axis, boost_histogram):
    if boost_histogram is None:
        return None

    if not isinstance(
        axis, (boost_histogram.axis.IntCategory, boost_histogram.axis.StrCategory)
    ):
        return None

    labels = [str(label) for label in axis]
    if isinstance(axis, boost_histogram.axis.IntCategory):
        # Check labels are valid integers (this may be redundant)
        for label in labels:
            try:
                int(label)
            except ValueError:
                raise ValueError(
                    f"IntCategory labels must be valid integers. Found {label!r} on axis {axis!r}"
                ) from None

    labels = to_THashList([to_TObjString(label) for label in labels])
    # we need to set the TObject.fUniqueID to the index of the bin as done by TAxis::SetBinLabel
    for i, label in enumerate(labels):
        label._bases[0]._members["@fUniqueID"] = i + 1

    return labels


def _root_stats_1d(entries, edges):
    centers = (edges[:-1] + edges[1:]) / 2.0

    fTsumw = fTsumw2 = entries.sum()
    fTsumwx = (entries * centers).sum()
    fTsumwx2 = (entries * centers**2).sum()

    return fTsumw, fTsumw2, fTsumwx, fTsumwx2


def _root_stats_2d(entries, xedges, yedges):
    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    fTsumw = fTsumw2 = entries.sum()
    fTsumwx = (entries.T * xcenters).sum()
    fTsumwx2 = (entries.T * xcenters**2).sum()
    fTsumwy = (entries * ycenters).sum()
    fTsumwy2 = (entries * ycenters**2).sum()
    fTsumwxy = ((entries * ycenters).T * xcenters).sum()
    return fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwy, fTsumwy2, fTsumwxy


def _root_stats_3d(entries, xedges, yedges, zedges):
    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    zcenters = (zedges[:-1] + zedges[1:]) / 2.0
    fTsumw = fTsumw2 = entries.sum()
    fTsumwx = (numpy.transpose(entries, (1, 2, 0)) * xcenters).sum()
    fTsumwx2 = (numpy.transpose(entries, (1, 2, 0)) * xcenters**2).sum()
    fTsumwy = (numpy.transpose(entries, (2, 0, 1)) * ycenters).sum()
    fTsumwy2 = (numpy.transpose(entries, (2, 0, 1)) * ycenters**2).sum()
    fTsumwz = (entries * zcenters).sum()
    fTsumwz2 = (entries * zcenters**2).sum()
    fTsumwxy = (
        numpy.transpose(numpy.transpose(entries, (2, 0, 1)) * ycenters, (2, 0, 1))
        * xcenters
    ).sum()
    fTsumwxz = (numpy.transpose(entries * zcenters, (1, 2, 0)) * xcenters).sum()
    fTsumwyz = (numpy.transpose(entries * zcenters, (2, 0, 1)) * ycenters).sum()
    return (
        fTsumw,
        fTsumw2,
        fTsumwx,
        fTsumwx2,
        fTsumwy,
        fTsumwy2,
        fTsumwxy,
        fTsumwz,
        fTsumwz2,
        fTsumwxz,
        fTsumwyz,
    )


def to_TString(string):
    """
    This function is for developers to create TString objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tstring = uproot.models.TString.Model_TString(str(string))
    tstring._deeply_writable = True
    tstring._cursor = None
    tstring._file = None
    tstring._parent = None
    tstring._members = {}
    tstring._bases = []
    tstring._num_bytes = None
    tstring._instance_version = None
    return tstring


def to_TObjString(string):
    """
    This function is for developers to create TObjString objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tobject = uproot.models.TObject.Model_TObject.empty()

    tobjstring = uproot.models.TObjString.Model_TObjString(str(string))
    tobjstring._deeply_writable = True
    tobjstring._cursor = None
    tobjstring._parent = None
    tobjstring._members = {}
    tobjstring._bases = (tobject,)
    tobjstring._num_bytes = len(string) + (1 if len(string) < 255 else 5) + 16
    tobjstring._instance_version = 1
    return tobjstring


def to_TList(data, name=""):
    """
    Args:
        data (:doc:`uproot.model.Model`): Python iterable to convert into a TList.
        name (str): Name of the list (usually empty: ``""``).

    This function is for developers to create TList objects that can be
    written to ROOT files, to implement conversion routines.
    """
    if not all(isinstance(x, uproot.model.Model) for x in data):
        raise TypeError(
            "list to convert to TList must only contain ROOT objects (uproot.Model)"
        )

    tobject = uproot.models.TObject.Model_TObject.empty()

    tlist = uproot.models.TList.Model_TList.empty()
    tlist._bases.append(tobject)
    tlist._members["fName"] = name
    tlist._data = list(data)
    tlist._members["fSize"] = len(tlist._data)
    tlist._options = [b""] * len(tlist._data)

    if all(x._deeply_writable for x in tlist._data):
        tlist._deeply_writable = True

    return tlist


def to_THashList(data, name=""):
    """
    Args:
        data (:doc:`uproot.model.Model`): Python iterable to convert into a THashList.
        name (str): Name of the list (usually empty: ``""``).

    This function is for developers to create THashList objects that can be
    written to ROOT files, to implement conversion routines.
    """

    if not all(isinstance(x, uproot.model.Model) for x in data):
        raise TypeError(
            "list to convert to THashList must only contain ROOT objects (uproot.Model)"
        )

    tlist = to_TList(data, name)

    thashlist = uproot.models.THashList.Model_THashList.empty()

    thashlist._bases.append(tlist)

    return thashlist


def to_TArray(data):
    """
    Args:
        data (numpy.ndarray): The array to convert to big-endian and wrap as
            TArrayC, TArrayS, TArrayI, TArrayL, TArrayF, or TArrayD, depending
            on its dtype.

    This function is for developers to create TArray objects that can be
    written to ROOT files, to implement conversion routines.
    """
    if data.ndim != 1:
        raise ValueError("data to convert to TArray must be one-dimensional")

    # dispatch on kind and size, since distinct C types of the same size (such as
    # long and long long) are distinct numpy scalar types
    cls = {
        ("i", 1): uproot.models.TArray.Model_TArrayC,
        ("i", 2): uproot.models.TArray.Model_TArrayS,
        ("i", 4): uproot.models.TArray.Model_TArrayI,
        ("i", 8): uproot.models.TArray.Model_TArrayL,
        ("f", 4): uproot.models.TArray.Model_TArrayF,
        ("f", 8): uproot.models.TArray.Model_TArrayD,
    }.get((data.dtype.kind, data.dtype.itemsize))
    if cls is None:
        raise ValueError(
            f"data to convert to TArray must have signed integer or floating-point type, not {data.dtype!r}"
        )

    tarray = cls.empty()
    tarray._deeply_writable = True
    tarray._members["fN"] = len(data)
    tarray._data = data.astype(data.dtype.newbyteorder(">"))
    return tarray


def to_TAxis(
    fName,
    fTitle,
    fNbins,
    fXmin,
    fXmax,
    fXbins=None,
    fFirst=0,
    fLast=0,
    fBits2=0,
    fTimeDisplay=False,
    fTimeFormat="",
    fLabels=None,
    fModLabs=None,
    fNdivisions=510,
    fAxisColor=1,
    fLabelColor=1,
    fLabelFont=42,
    fLabelOffset=0.005,
    fLabelSize=0.035,
    fTickLength=0.03,
    fTitleOffset=1.0,
    fTitleSize=0.035,
    fTitleColor=1,
    fTitleFont=42,
):
    """
    Args:
        fName (str): Internal name of axis, usually ``"xaxis"``, ``"yaxis"``, ``"zaxis"``.
        fTitle (str): Internal title of axis, usually empty: ``""``.
        fNbins (int): Number of bins. (https://root.cern.ch/doc/master/classTAxis.html)
        fXmin (float): Low edge of first bin.
        fXmax (float): Upper edge of last bin.
        fXbins (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Bin
            edges array in X. None generates an empty array.
        fFirst (int): First bin to display. 1 if no range defined NOTE: in some cases a zero is returned (see TAxis::SetRange)
        fLast (int): Last bin to display. fNbins if no range defined NOTE: in some cases a zero is returned (see TAxis::SetRange)
        fBits2 (int): Second bit status word.
        fTimeDisplay (bool): On/off displaying time values instead of numerics.
        fTimeFormat (str or :doc:`uproot.models.TString.Model_TString`): Date&time format, ex: 09/12/99 12:34:00.
        fLabels (None or :doc:`uproot.models.THashList.Model_THashList`): List of labels.
        fModLabs (None or :doc:`uproot.models.TList.Model_TList`): List of modified labels.
        fNdivisions (int): Number of divisions(10000*n3 + 100*n2 + n1). (https://root.cern.ch/doc/master/classTAttAxis.html)
        fAxisColor (int): Color of the line axis.
        fLabelColor (int): Color of labels.
        fLabelFont (int): Font for labels.
        fLabelOffset (float): Offset of labels.
        fLabelSize (float): Size of labels.
        fTickLength (float): Length of tick marks.
        fTitleOffset (float): Offset of axis title.
        fTitleSize (float): Size of axis title.
        fTitleColor (int): Color of axis title.
        fTitleFont (int): Font for axis title.

    This function is for developers to create TAxis objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tobject = uproot.models.TObject.Model_TObject.empty()

    tnamed = uproot.models.TNamed.Model_TNamed.empty()
    tnamed._deeply_writable = True
    tnamed._bases.append(tobject)
    tnamed._members["fName"] = fName
    tnamed._members["fTitle"] = fTitle

    tattaxis = uproot.models.TAtt.Model_TAttAxis_v4.empty()
    tattaxis._deeply_writable = True
    tattaxis._members["fNdivisions"] = fNdivisions
    tattaxis._members["fAxisColor"] = fAxisColor
    tattaxis._members["fLabelColor"] = fLabelColor
    tattaxis._members["fLabelFont"] = fLabelFont
    tattaxis._members["fLabelOffset"] = fLabelOffset
    tattaxis._members["fLabelSize"] = fLabelSize
    tattaxis._members["fTickLength"] = fTickLength
    tattaxis._members["fTitleOffset"] = fTitleOffset
    tattaxis._members["fTitleSize"] = fTitleSize
    tattaxis._members["fTitleColor"] = fTitleColor
    tattaxis._members["fTitleFont"] = fTitleFont

    if fXbins is None:
        fXbins = numpy.array([], dtype=numpy.float64)

    if isinstance(fXbins, uproot.models.TArray.Model_TArrayD):
        tarray_fXbins = fXbins
    else:
        tarray_fXbins = to_TArray(fXbins)

    if isinstance(fTimeFormat, uproot.models.TString.Model_TString):
        tstring_fTimeFormat = fTimeFormat
    else:
        tstring_fTimeFormat = to_TString(fTimeFormat)

    taxis = uproot.models.TH.Model_TAxis_v10.empty()
    taxis._deeply_writable = True
    taxis._bases.append(tnamed)
    taxis._bases.append(tattaxis)
    taxis._members["fNbins"] = fNbins
    taxis._members["fXmin"] = fXmin
    taxis._members["fXmax"] = fXmax
    taxis._members["fXbins"] = tarray_fXbins
    taxis._members["fFirst"] = fFirst
    taxis._members["fLast"] = fLast
    taxis._members["fBits2"] = fBits2
    taxis._members["fTimeDisplay"] = fTimeDisplay
    taxis._members["fTimeFormat"] = tstring_fTimeFormat
    taxis._members["fLabels"] = fLabels
    taxis._members["fModLabs"] = fModLabs

    return taxis


def to_TH1x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fSumw2,
    fXaxis,
    fYaxis=None,
    fZaxis=None,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH1C, TH1D, TH1F, TH1I, or TH1S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D histograms.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH1* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH1C, TH1D, TH1F, TH1I, or TH1S depends on the dtype of the ``data`` array.
    """
    tobject = uproot.models.TObject.Model_TObject.empty()

    tnamed = uproot.models.TNamed.Model_TNamed.empty()
    tnamed._deeply_writable = True
    tnamed._bases.append(tobject)
    tnamed._members["fName"] = fName
    tnamed._members["fTitle"] = fTitle

    tattline = uproot.models.TAtt.Model_TAttLine_v2.empty()
    tattline._deeply_writable = True
    tattline._members["fLineColor"] = fLineColor
    tattline._members["fLineStyle"] = fLineStyle
    tattline._members["fLineWidth"] = fLineWidth

    tattfill = uproot.models.TAtt.Model_TAttFill_v2.empty()
    tattfill._deeply_writable = True
    tattfill._members["fFillColor"] = fFillColor
    tattfill._members["fFillStyle"] = fFillStyle

    tattmarker = uproot.models.TAtt.Model_TAttMarker_v2.empty()
    tattmarker._deeply_writable = True
    tattmarker._members["fMarkerColor"] = fMarkerColor
    tattmarker._members["fMarkerStyle"] = fMarkerStyle
    tattmarker._members["fMarkerSize"] = fMarkerSize

    th1 = uproot.models.TH.Model_TH1_v8.empty()

    th1._bases.append(tnamed)
    th1._bases.append(tattline)
    th1._bases.append(tattfill)
    th1._bases.append(tattmarker)

    if fYaxis is None:
        fYaxis = to_TAxis(fName="yaxis", fTitle="", fNbins=1, fXmin=0.0, fXmax=1.0)
    if fZaxis is None:
        fZaxis = to_TAxis(fName="zaxis", fTitle="", fNbins=1, fXmin=0.0, fXmax=1.0)
    if fContour is None:
        fContour = numpy.array([], dtype=numpy.float64)
    if fFunctions is None:
        fFunctions = []
    if fBuffer is None:
        fBuffer = numpy.array([], dtype=numpy.float64)

    if isinstance(data, uproot.models.TArray.Model_TArray):
        tarray_data = data
    else:
        tarray_data = to_TArray(data)

    if fSumw2 is None:
        tarray_fSumw2 = to_TArray(numpy.array([], dtype=numpy.float64))
    elif isinstance(fSumw2, uproot.models.TArray.Model_TArray):
        tarray_fSumw2 = fSumw2
    else:
        tarray_fSumw2 = to_TArray(fSumw2)
    if not isinstance(tarray_fSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fSumw2 must be an array of float64 (TArrayD)")

    if isinstance(fContour, uproot.models.TArray.Model_TArray):
        tarray_fContour = fContour
    else:
        tarray_fContour = to_TArray(fContour)
    if not isinstance(tarray_fContour, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fContour must be an array of float64 (TArrayD)")

    if isinstance(fOption, uproot.models.TString.Model_TString):
        tstring_fOption = fOption
    else:
        tstring_fOption = to_TString(fOption)

    if isinstance(fFunctions, uproot.models.TList.Model_TList):
        tlist_fFunctions = fFunctions
    else:
        tlist_fFunctions = to_TList(fFunctions, name="")
    # FIXME: require all list items to be the appropriate class (TFunction?)

    th1._members["fNcells"] = len(tarray_data) if fNcells is None else fNcells
    th1._members["fXaxis"] = fXaxis
    th1._members["fYaxis"] = fYaxis
    th1._members["fZaxis"] = fZaxis
    th1._members["fBarOffset"] = fBarOffset
    th1._members["fBarWidth"] = fBarWidth
    th1._members["fEntries"] = fEntries
    th1._members["fTsumw"] = fTsumw
    th1._members["fTsumw2"] = fTsumw2
    th1._members["fTsumwx"] = fTsumwx
    th1._members["fTsumwx2"] = fTsumwx2
    th1._members["fMaximum"] = fMaximum
    th1._members["fMinimum"] = fMinimum
    th1._members["fNormFactor"] = fNormFactor
    th1._members["fContour"] = tarray_fContour
    th1._members["fSumw2"] = tarray_fSumw2
    th1._members["fOption"] = tstring_fOption
    th1._members["fFunctions"] = tlist_fFunctions
    th1._members["fBufferSize"] = len(fBuffer) if fBufferSize is None else fBufferSize
    th1._members["fBuffer"] = fBuffer
    th1._members["fBinStatErrOpt"] = fBinStatErrOpt
    th1._members["fStatOverflows"] = fStatOverflows

    th1._speedbump1 = b"\x00"

    th1._deeply_writable = tlist_fFunctions._deeply_writable

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH1C_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH1S_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH1I_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH1F_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH1D_v3
    else:
        raise TypeError(f"no TH1* subclasses correspond to {tarray_data.classname}")

    th1x = cls.empty()
    th1x._bases.append(th1)
    th1x._bases.append(tarray_data)

    th1x._deeply_writable = th1._deeply_writable

    return th1x


def to_TH2x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fSumw2,
    fXaxis,
    fYaxis,
    fZaxis=None,
    fScalefactor=1.0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH2C, TH2D, TH2F, TH2I, or TH2S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH2 only: https://root.cern.ch/doc/master/classTH2.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH2 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH2 only.)
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fScalefactor (float): Scale factor. (TH2 only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH2* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH2C, TH2D, TH2F, TH2I, or TH2S depends on the dtype of the ``data`` array.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )

    th1 = th1x._bases[0]
    tarray_data = th1x._bases[1]

    th2 = uproot.models.TH.Model_TH2_v5.empty()
    th2._bases.append(th1)
    th2._members["fScalefactor"] = fScalefactor
    th2._members["fTsumwy"] = fTsumwy
    th2._members["fTsumwy2"] = fTsumwy2
    th2._members["fTsumwxy"] = fTsumwxy

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH2C_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH2S_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH2I_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH2F_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH2D_v4
    else:
        raise TypeError(f"no TH2* subclasses correspond to {tarray_data.classname}")

    th2x = cls.empty()
    th2x._bases.append(th2)
    th2x._bases.append(tarray_data)

    th2._deeply_writable = th1._deeply_writable
    th2x._deeply_writable = th2._deeply_writable

    return th2x


def to_TH3x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fTsumwz,
    fTsumwz2,
    fTsumwxz,
    fTsumwyz,
    fSumw2,
    fXaxis,
    fYaxis,
    fZaxis,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH3C, TH3D, TH3F, TH3I, or TH3S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH3 only: https://root.cern.ch/doc/master/classTH3.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH3 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH3 only.)
        fTsumwz (float): Total Sum of weight*Z. (TH3 only.)
        fTsumwz2 (float): Total Sum of weight*Z*Z. (TH3 only.)
        fTsumwxz (float): Total Sum of weight*X*Z. (TH3 only.)
        fTsumwyz (float): Total Sum of weight*Y*Z. (TH3 only.)
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="zaxis"`` and ``fTitle=""``.
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH3* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH3C, TH3D, TH3F, TH3I, or TH3S depends on the dtype of the ``data`` array.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )

    th1 = th1x._bases[0]
    tarray_data = th1x._bases[1]

    tatt3d = uproot.models.TAtt.Model_TAtt3D_v1.empty()
    tatt3d._deeply_writable = True

    th3 = uproot.models.TH.Model_TH3_v6.empty()
    th3._bases.append(th1)
    th3._bases.append(tatt3d)
    th3._members["fTsumwy"] = fTsumwy
    th3._members["fTsumwy2"] = fTsumwy2
    th3._members["fTsumwxy"] = fTsumwxy
    th3._members["fTsumwz"] = fTsumwz
    th3._members["fTsumwz2"] = fTsumwz2
    th3._members["fTsumwxz"] = fTsumwxz
    th3._members["fTsumwyz"] = fTsumwyz

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH3C_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH3S_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH3I_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH3F_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH3D_v4
    else:
        raise TypeError(f"no TH3* subclasses correspond to {tarray_data.classname}")

    th3x = cls.empty()
    th3x._bases.append(th3)
    th3x._bases.append(tarray_data)

    th3._deeply_writable = th1._deeply_writable
    th3x._deeply_writable = th3._deeply_writable

    return th3x


def to_TProfile(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fSumw2,
    fBinEntries,
    fBinSumw2,
    fXaxis,
    fYaxis=None,
    fZaxis=None,
    fYmin=0.0,
    fYmax=0.0,
    fErrorMode=0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Bin
            contents with first bin as underflow, last bin as overflow. The dtype of this array
            must be float64.
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TProfile only: https://root.cern.ch/doc/master/classTProfile.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TProfile only.)
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fBinEntries (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Number
            of entries per bin. (TProfile only.)
        fBinSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights per bin. (TProfile only.)
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D histograms.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fYmin (float): Lower limit in Y (if set). (TProfile only.)
        fYmax (float): Upper limit in Y (if set). (TProfile only.)
        fErrorMode (int): Option to compute errors. (TProfile only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TProfile objects that can be
    written to ROOT files, to implement conversion routines.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )
    if not isinstance(th1x, uproot.models.TH.Model_TH1D_v3):
        raise TypeError("TProfile requires an array of float64 (TArrayD)")

    if isinstance(fBinEntries, uproot.models.TArray.Model_TArray):
        tarray_fBinEntries = fBinEntries
    else:
        tarray_fBinEntries = to_TArray(fBinEntries)
    if not isinstance(tarray_fBinEntries, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinEntries must be an array of float64 (TArrayD)")

    if isinstance(fBinSumw2, uproot.models.TArray.Model_TArray):
        tarray_fBinSumw2 = fBinSumw2
    else:
        tarray_fBinSumw2 = to_TArray(fBinSumw2)
    if not isinstance(tarray_fBinSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinSumw2 must be an array of float64 (TArrayD)")

    tprofile = uproot.models.TH.Model_TProfile_v7.empty()
    tprofile._bases.append(th1x)
    tprofile._members["fBinEntries"] = tarray_fBinEntries
    tprofile._members["fErrorMode"] = fErrorMode
    tprofile._members["fYmin"] = fYmin
    tprofile._members["fYmax"] = fYmax
    tprofile._members["fTsumwy"] = fTsumwy
    tprofile._members["fTsumwy2"] = fTsumwy2
    tprofile._members["fBinSumw2"] = tarray_fBinSumw2

    tprofile._deeply_writable = th1x._deeply_writable

    return tprofile


def to_TProfile2D(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fTsumwz,
    fTsumwz2,
    fSumw2,
    fBinEntries,
    fBinSumw2,
    fXaxis,
    fYaxis,
    fZaxis=None,
    fScalefactor=1.0,
    fZmin=0.0,
    fZmax=0.0,
    fErrorMode=0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            must be float64.
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH2 only: https://root.cern.ch/doc/master/classTH2.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH2 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH2 only.)
        fTsumwz (float): Total Sum of weight*Z. (TProfile2D only: https://root.cern.ch/doc/master/classTProfile2D.html)
        fTsumwz2 (float): Total Sum of weight*Z*Z. (TProfile2D only.)
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fBinEntries (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Number
            of entries per bin. (TProfile2D only.)
        fBinSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights per bin. (TProfile2D only.)
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fScalefactor (float): Scale factor. (TH2 only.)
        fZmin (float): Lower limit in Z (if set). (TProfile2D only.)
        fZmax (float): Upper limit in Z (if set). (TProfile2D only.)
        fErrorMode (int): Option to compute errors. (TProfile2D only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TProfile2D objects that can be
    written to ROOT files, to implement conversion routines.
    """
    th2x = to_TH2x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fTsumwy=fTsumwy,
        fTsumwy2=fTsumwy2,
        fTsumwxy=fTsumwxy,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fScalefactor=fScalefactor,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )
    if not isinstance(th2x, uproot.models.TH.Model_TH2D_v4):
        raise TypeError("TProfile2D requires an array of float64 (TArrayD)")

    if isinstance(fBinEntries, uproot.models.TArray.Model_TArray):
        tarray_fBinEntries = fBinEntries
    else:
        tarray_fBinEntries = to_TArray(fBinEntries)
    if not isinstance(tarray_fBinEntries, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinEntries must be an array of float64 (TArrayD)")

    if isinstance(fBinSumw2, uproot.models.TArray.Model_TArray):
        tarray_fBinSumw2 = fBinSumw2
    else:
        tarray_fBinSumw2 = to_TArray(fBinSumw2)
    if not isinstance(tarray_fBinSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinSumw2 must be an array of float64 (TArrayD)")

    tprofile2d = uproot.models.TH.Model_TProfile2D_v8.empty()
    tprofile2d._bases.append(th2x)
    tprofile2d._members["fBinEntries"] = tarray_fBinEntries
    tprofile2d._members["fErrorMode"] = fErrorMode
    tprofile2d._members["fZmin"] = fZmin
    tprofile2d._members["fZmax"] = fZmax
    tprofile2d._members["fTsumwz"] = fTsumwz
    tprofile2d._members["fTsumwz2"] = fTsumwz2
    tprofile2d._members["fBinSumw2"] = tarray_fBinSumw2

    tprofile2d._deeply_writable = th2x._deeply_writable

    return tprofile2d


def to_TProfile3D(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fTsumwz,
    fTsumwz2,
    fTsumwxz,
    fTsumwyz,
    fTsumwt,
    fTsumwt2,
    fSumw2,
    fBinEntries,
    fBinSumw2,
    fXaxis,
    fYaxis,
    fZaxis,
    fTmin=0.0,
    fTmax=0.0,
    fErrorMode=0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            must be float64.
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH3 only: https://root.cern.ch/doc/master/classTH3.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH3 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH3 only.)
        fTsumwz (float): Total Sum of weight*Z. (TH3 only.)
        fTsumwz2 (float): Total Sum of weight*Z*Z. (TH3 only.)
        fTsumwxz (float): Total Sum of weight*X*Z. (TH3 only.)
        fTsumwyz (float): Total Sum of weight*Y*Z. (TH3 only.)
        fTsumwt (float): Total Sum of weight*T. (TProfile3D only: https://root.cern.ch/doc/master/classTProfile3D.html)
        fTsumwt2 (float): Total Sum of weight*T*T. (TProfile3D only.)
        fSumw2 (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights. If None, a zero-length :doc:`uproot.models.TArray.Model_TArrayD`
            is created in its place.
        fBinEntries (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Number
            of entries per bin. (TProfile3D only.)
        fBinSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights per bin. (TProfile3D only.)
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.identify.to_TAxis`
            with ``fName="zaxis"`` and ``fTitle=""``.
        fTmin (float): Lower limit in T (if set). (TProfile3D only.)
        fTmax (float): Upper limit in T (if set). (TProfile3D only.)
        fErrorMode (int): Option to compute errors. (TProfile3D only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TProfile3D objects that can be
    written to ROOT files, to implement conversion routines.
    """
    th3x = to_TH3x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fTsumwy=fTsumwy,
        fTsumwy2=fTsumwy2,
        fTsumwxy=fTsumwxy,
        fTsumwz=fTsumwz,
        fTsumwz2=fTsumwz2,
        fTsumwxz=fTsumwxz,
        fTsumwyz=fTsumwyz,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )
    if not isinstance(th3x, uproot.models.TH.Model_TH3D_v4):
        raise TypeError("TProfile3D requires an array of float64 (TArrayD)")

    if isinstance(fBinEntries, uproot.models.TArray.Model_TArray):
        tarray_fBinEntries = fBinEntries
    else:
        tarray_fBinEntries = to_TArray(fBinEntries)
    if not isinstance(tarray_fBinEntries, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinEntries must be an array of float64 (TArrayD)")

    if isinstance(fBinSumw2, uproot.models.TArray.Model_TArray):
        tarray_fBinSumw2 = fBinSumw2
    else:
        tarray_fBinSumw2 = to_TArray(fBinSumw2)
    if not isinstance(tarray_fBinSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinSumw2 must be an array of float64 (TArrayD)")

    tprofile3d = uproot.models.TH.Model_TProfile3D_v8.empty()
    tprofile3d._bases.append(th3x)
    tprofile3d._members["fBinEntries"] = tarray_fBinEntries
    tprofile3d._members["fErrorMode"] = fErrorMode
    tprofile3d._members["fTmin"] = fTmin
    tprofile3d._members["fTmax"] = fTmax
    tprofile3d._members["fTsumwt"] = fTsumwt
    tprofile3d._members["fTsumwt2"] = fTsumwt2
    tprofile3d._members["fBinSumw2"] = tarray_fBinSumw2

    tprofile3d._deeply_writable = th3x._deeply_writable

    return tprofile3d
