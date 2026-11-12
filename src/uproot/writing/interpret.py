from __future__ import annotations

import numbers

import numpy as np

import uproot


def _as_TGraph(
    x,
    y,
    x_errors=None,
    y_errors=None,
    x_errors_low=None,
    x_errors_high=None,
    y_errors_low=None,
    y_errors_high=None,
    title="",
    xAxisLabel="",
    yAxisLabel="",
    minY=None,
    maxY=None,
    lineColor: int = 602,
    lineStyle: int = 1,
    lineWidth: int = 1,
    fillColor: int = 0,
    fillStyle: int = 1001,
    markerColor: int = 1,
    markerStyle: int = 1,
    markerSize: float = 1.0,
):
    """
    Args:
        x (1D numpy.ndarray): x values of TGraph (length of x and y has to be the same).
        y (1D numpy.ndarray): y values of TGraph (length of x and y has to be the same).
        x_errors(None or 1D numpy.ndarray): Symethrical values of errors for corresponding x value (length of x_errors has to be the same as x and y)
        y_errors(None or 1D numpy.ndarray): Symethrical values of errors for corresponding y value (length of y_errors has to be the same as x and y)
        x_errors_low(None or 1D numpy.ndarray): Asymmetrical lower values of errors for corresponding x value (length of x_errors_low has to be the same as x and y)
        x_errors_high(None or 1D numpy.ndarray): Asymmetrical upper values of errors for corresponding x value (length of x_errors_high has to be the same as x and y)
        y_errors_low(None or 1D numpy.ndarray): Asymmetrical lower values of errors for corresponding y value (length of y_errors_low has to be the same as x and y)
        y_errors_high(None or 1D numpy.ndarray): Asymmetrical upper values of errors for corresponding y value (length of y_errors_high has to be the same as x and y)
        title (str): Title of the histogram.
        xAxisLabel (str): Label of the X axis.
        yAxisLabel (str): Label of the Y axis.
        minY (None or float): Minimum value on the Y axis to be shown, if set to None then minY=min(y)
        maxY (None or float): Maximum value on the Y axis to be shown, if set to None then maxY=max(y)
        lineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        lineStyle (int): Line style.
        lineWidth (int): Line width.
        fillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fillStyle (int): Fill area style.
        markerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        markerStyle (int): Marker style.
        markerSize (float): Marker size.

    WARNING! This function only works for TGraph, because serialization of TGraphErrors and TGraphAsymmErrors is not implemented yet.

    Function that converts arguments into TGraph, TGraphErrors or TGraphAsymmErros based on the given arguments.
    When all errors are unspecified, detected object is TGraph.
    When x_errors, y_errors are specified, detected object is TGraphErrors.
    When x_errors_low, x_errors_high, y_errors_low, y_errors_high are specified, detected object is TGraphAsymmErrors.
    Note that both x_errors, y_errors need to be specified or set to None.
    The same rule applies to x_errors_low, x_errors_high, y_errors_low, y_errors_high.
    Also can't specify x_errors, y_errors and x_errors_low, x_errors_high, y_errors_low, y_errors_high at the same time.
    All rules are designed to remove any ambiguity.
    """

    sym_errors = [x_errors, y_errors]
    sym_errors_bool = [err is not None for err in sym_errors]

    asym_errors = [x_errors_low, x_errors_high, y_errors_low, y_errors_high]
    asym_errors_bool = [err is not None for err in asym_errors]

    tgraph_type = "TGraph"

    # Detecting which type of TGraph to chose
    if any(sym_errors_bool):
        if not all(sym_errors_bool):
            raise ValueError("uproot.as_TGraph requires both x_errors and y_errors")
        if any(asym_errors_bool):
            raise ValueError(
                "uproot.as_TGraph can accept symmetrical errors OR asymmetrical errors, but not both"
            )
        tgraph_type = "TGraphErrors"

    elif any(asym_errors_bool):
        if not all(asym_errors_bool):
            raise ValueError(
                "uproot.as_TGraph requires all of the following: x_errors_low, x_errors_high, y_errors_low, y_errors_high"
            )
        tgraph_type = "TGraphAsymmErrors"

    tobject = uproot.models.TObject.Model_TObject.empty()

    tnamed = uproot.models.TNamed.Model_TNamed.empty()
    tnamed._deeply_writable = True
    tnamed._bases.append(tobject)
    tnamed._members["fName"] = (
        ""  #  Temporary name, will be overwritten by the writing process because Uproot's write syntax is ``file[name] = histogram``
    )
    # Constraint so user won't break TGraph naming
    if ";" in title or ";" in xAxisLabel or ";" in yAxisLabel:
        raise ValueError("title and xAxisLabel and yAxisLabel can't contain ';'!")
    fTitle = f"{title};{xAxisLabel};{yAxisLabel}"
    tnamed._members["fTitle"] = fTitle

    # setting line styling
    tattline = uproot.models.TAtt.Model_TAttLine_v2.empty()
    tattline._deeply_writable = True
    tattline._members["fLineColor"] = lineColor
    tattline._members["fLineStyle"] = lineStyle
    tattline._members["fLineWidth"] = lineWidth

    # setting filling styling, does not do anything to TGraph
    tattfill = uproot.models.TAtt.Model_TAttFill_v2.empty()
    tattfill._deeply_writable = True
    tattfill._members["fFillColor"] = fillColor
    tattfill._members["fFillStyle"] = fillStyle

    # setting marker styling, those are points on graph
    tattmarker = uproot.models.TAtt.Model_TAttMarker_v2.empty()
    tattmarker._deeply_writable = True
    tattmarker._members["fMarkerColor"] = markerColor
    tattmarker._members["fMarkerStyle"] = markerStyle
    tattmarker._members["fMarkerSize"] = markerSize

    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length!")
    if len(x) == 0:
        raise ValueError("uproot.as_TGraph x and y arrays can't be empty")
    if len(x.shape) != 1:
        raise ValueError(f"x has to be 1D, but is {len(x.shape)}D!")
    if len(y.shape) != 1:
        raise ValueError(f"y has to be 1D, but is {len(y.shape)}D!")

    if minY is None:
        new_minY = np.min(y)
    elif not isinstance(minY, numbers.Real):
        raise ValueError(
            f"uproot.as_TGraph minY has to be None or a number, not {type(minY)}"
        )
    else:
        new_minY = minY

    if maxY is None:
        new_maxY = np.max(y)
    elif not isinstance(maxY, numbers.Real):
        raise ValueError(
            f"uproot.as_TGraph maxY has to be None or a number, not {type(maxY)}"
        )
    else:
        new_maxY = maxY

    tGraph = uproot.models.TGraph.Model_TGraph_v4.empty()

    tGraph._bases.append(tnamed)
    tGraph._bases.append(tattline)
    tGraph._bases.append(tattfill)
    tGraph._bases.append(tattmarker)

    tGraph._members["fNpoints"] = len(x)
    tGraph._members["fX"] = x
    tGraph._members["fY"] = y
    tGraph._members["fMinimum"] = (
        minY if minY is not None else new_minY - 0.1 * (new_maxY - new_minY)
    )  # by default graph line wont touch the edge of the chart
    tGraph._members["fMaximum"] = (
        maxY if maxY is not None else new_maxY + 0.1 * (new_maxY - new_minY)
    )  # by default graph line wont touch the edge of the chart

    returned_TGraph = tGraph

    if tgraph_type == "TGraphErrors":
        if not (len(x_errors) == len(y_errors) == len(x)):
            raise ValueError(
                "Length of all error arrays has to be the same as length of arrays X and Y"
            )
        tGraphErrors = uproot.models.TGraph.Model_TGraphErrors_v3.empty()
        tGraphErrors._bases.append(tGraph)
        tGraphErrors._members["fEX"] = x_errors
        tGraphErrors._members["fEY"] = y_errors

        returned_TGraph = tGraphErrors
    elif tgraph_type == "TGraphAsymmErrors":
        if not (
            len(x_errors_low)
            == len(x_errors_high)
            == len(y_errors_low)
            == len(y_errors_high)
            == len(x)
        ):
            raise ValueError(
                "Length of errors all error arrays has to be the same as length of arrays X and Y"
            )
        tGraphAsymmErrors = uproot.models.TGraph.Model_TGraphAsymmErrors_v3.empty()
        tGraphAsymmErrors._bases.append(tGraph)
        tGraphAsymmErrors._members["fEXlow"] = x_errors_low
        tGraphAsymmErrors._members["fEXhigh"] = x_errors_high
        tGraphAsymmErrors._members["fEYlow"] = y_errors_low
        tGraphAsymmErrors._members["fEYhigh"] = y_errors_high

        returned_TGraph = tGraphAsymmErrors

    return returned_TGraph


def as_TGraph(
    df,
    title="",
    xAxisLabel="",
    yAxisLabel="",
    minY=None,
    maxY=None,
    lineColor: int = 602,
    lineStyle: int = 1,
    lineWidth: int = 1,
    markerColor: int = 1,
    markerStyle: int = 1,
    markerSize: float = 1.0,
):
    """
    Args:
        df (DataFrame or and dict like object): DataFrame object with column names as follows:
            x (float): x values of TGraph.
            y (float): y values of TGraph.
            x_errors (float or left unspecified): Symethrical error values for corresponding x value
            y_errors (float or left unspecified): Symethrical error values for corresponding y value
            x_errors_low (float or left unspecified): Asymmetrical lower error values for corresponding x value
            x_errors_high (float or left unspecified): Asymmetrical upper error values for corresponding x value
            y_errors_low (float or left unspecified): Asymmetrical lower error values for corresponding y value
            y_errors_high (float or left unspecified): Asymmetrical upper error values for corresponding y value
            (other column names will be ignored!)
        title (str): Title of the histogram.
        xAxisLabel (str): Label of the X axis.
        yAxisLabel (str): Label of the Y axis.
        minY (None or float): Minimum value on the Y axis to be shown, if set to None then minY=min(y)
        maxY (None or float): Maximum value on the Y axis to be shown, if set to None then maxY=max(y)
        lineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        lineStyle (int): Line style.
        lineWidth (int): Line width.
        markerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        markerStyle (int): Marker style.
        markerSize (float): Marker size.

    WARNING! This function only works for TGraph, because serialization of TGraphErrors and TGraphAsymmErrors is not implemented yet.

    Function that converts DataFrame into TGraph, TGraphErrors or TGraphAsymmErros based on the specified DataFrame columns.
    When all error columns are unspecified, detected object is TGraph.
    When x_errors, y_errors are specified, detected object is TGraphErrors.
    When x_errors_low, x_errors_high, y_errors_low, y_errors_high are specified, detected object is TGraphAsymmErrors.
    Note that both {x_errors, x_errors} need to be specified or set to None.
    The same rule applies {to x_errors_low, x_errors_high, x_errors_low, x_errors_high}.
    Also can't specify {x_errors, y_errors} and {x_errors_low, x_errors_high, y_errors_low, y_errors_high} at the same time.
    """

    x = np.array(df["x"]) if df.get("x", None) is not None else None
    y = np.array(df["y"]) if df.get("y", None) is not None else None
    x_errors = (
        np.array(df["x_errors"]) if df.get("x_errors", None) is not None else None
    )
    y_errors = (
        np.array(df["y_errors"]) if df.get("y_errors", None) is not None else None
    )
    x_errors_low = (
        np.array(df["x_errors_low"])
        if df.get("x_errors_low", None) is not None
        else None
    )
    x_errors_high = (
        np.array(df["x_errors_high"])
        if df.get("x_errors_high", None) is not None
        else None
    )
    y_errors_low = (
        np.array(df["y_errors_low"])
        if df.get("y_errors_low", None) is not None
        else None
    )
    y_errors_high = (
        np.array(df["y_errors_high"])
        if df.get("y_errors_high", None) is not None
        else None
    )

    return _as_TGraph(
        x,
        y,
        x_errors,
        y_errors,
        x_errors_low,
        x_errors_high,
        y_errors_low,
        y_errors_high,
        title,
        xAxisLabel,
        yAxisLabel,
        minY,
        maxY,
        lineColor,
        lineStyle,
        lineWidth,
        0,
        1001,
        markerColor,
        markerStyle,
        markerSize,
    )
