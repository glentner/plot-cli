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
from numpy import histogram

# internal libs
from plot_cli.data import DataSet

# public interface
__all__ = ['PlotInterface', 'TPlot', 'TPlotLine', 'TPlotHist', ]


class PlotInterface(ABC):
    """Abstract plotting interface for all implementations."""

    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    size: Optional[Tuple[float, float]]
    legend: Optional[str]

    def __init__(self: PlotInterface,
                 title: str = None, xlabel: str = None, ylabel: str = None,
                 size: Tuple[float, float] = None, legend: str = None) -> None:
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.size = size
        self.legend = legend

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
                                   width=width, height=height, legendloc=self.legend)

    def draw(self: TPlot) -> None:
        """Render the plot."""
        print()  # leave a space!!
        self.figure.show()


class TPlotLine(TPlot):
    """Line plotting with `tplot`."""

    def add(self: TPlotLine, data: DataSet, column: str, **options) -> None:
        """Add data to plot."""
        self.figure.line(x=data.index, y=data[column], **options)


class TPlotHist(TPlot):
    """Histogram plotting with `tplot`."""

    def add(self: TPlotLine, data: DataSet, column: str, **options) -> None:
        """Add data to plot_cli."""
        hist, bin_edges = histogram(data[column],
                                    bins=options.pop('bins', 10),
                                    density=options.pop('density', None))
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.figure.bar(x=x, y=hist, **options)
