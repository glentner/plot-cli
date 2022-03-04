# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Plotting interface implementations."""


# type annotations
from __future__ import annotations
from typing import Tuple, Optional

# standard libs
from abc import ABC, abstractmethod

# external libs
import tplot
from pandas import Series

# internal libs
from plot.data import DataSet

# public interface
__all__ = ['PlotInterface', 'TPlot', 'TPlotLine', ]


class PlotInterface(ABC):
    """Abstract plotting interface for all implementations."""

    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    size: Optional[Tuple[float, float]]
    legendloc: Optional[str]

    def __init__(self: PlotInterface, title: str = None, xlabel: str = None, ylabel: str = None,
                 size: Tuple[float, float] = None, legendloc: str = None) -> None:
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.size = size
        self.legendloc = legendloc

    @abstractmethod
    def setup(self: PlotInterface) -> None:
        """Initialize figure object."""

    @abstractmethod
    def add(self: PlotInterface, data: DataSet, column: str, **options) -> None:
        """Add data to the plot."""

    @abstractmethod
    def draw(self: PlotInterface) -> None:
        """Render the plot."""


class TPlot(PlotInterface, ABC):
    """Terminal based plotting backend using `tplot`."""

    figure: tplot.Figure

    def setup(self: TPlotLine) -> None:
        """Create tplot figure."""
        width = height = None
        if self.size:
            width, height = map(int, self.size)
        self.figure = tplot.Figure(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel,
                                   width=width, height=height, legendloc=self.legendloc)


class TPlotLine(TPlot):
    """Line plotting with `tplot`."""

    def add(self: TPlotLine, data: DataSet, column: str, **options) -> None:
        """Add data to plot."""
        self.figure.line(x=data.index, y=data[column], **options)

    def draw(self: TPlotLine) -> None:
        """Render the plot."""
        self.figure.show()


class TPlotHist(TPlot):
    """Histogram plotting with `tplot`."""

    def add(self: TPlotLine, x: Series, y: Series, **options) -> None:
        """Add data to plot."""
        self.figure.line(x=x, y=y, **options)

    def draw(self: TPlotLine) -> None:
        """Render the plot."""
        self.figure.show()
