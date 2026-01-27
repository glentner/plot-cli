# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Plotting interface using tplot."""


# Type annotations
from __future__ import annotations
from typing import Tuple, Optional, List, Callable

# Standard libs
import logging
from enum import Enum, auto

# External libs
import tplot
import tplot.utils as tplot_utils
from numpy import histogram
from pandas import DataFrame

# Public interface
__all__ = [
    'Figure',
    'TimeSeriesFigure',
    'TimeGranularity',
    'generate_time_ticks',
]

# Module level logger
log = logging.getLogger(__name__)


class Figure:
    """
    Wrapper around tplot.Figure for terminal plotting.

    Provides a simplified interface for line plots, histograms, and bar charts.
    """

    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    size: Optional[Tuple[int, int]]
    legend: Optional[str]
    _figure: tplot.Figure

    def __init__(
        self: Figure,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        legend: str = 'bottomright',
    ) -> None:
        """Initialize figure with formatting options."""
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.size = size
        self.legend = legend
        self._setup()

    def _setup(self: Figure) -> None:
        """Create underlying tplot figure."""
        width = height = None
        if self.size:
            width, height = self.size
        self._figure = tplot.Figure(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            width=width,
            height=height,
            legendloc=self.legend,
        )

    def line(
        self: Figure,
        x: List[float],
        y: List[float],
        color: str = 'blue',
        label: Optional[str] = None,
    ) -> None:
        """Add a line plot to the figure."""
        self._figure.line(x=x, y=y, color=color, label=label)

    def bar(
        self: Figure,
        x: List[float],
        y: List[float],
        color: str = 'blue',
        label: Optional[str] = None,
    ) -> None:
        """Add a bar chart to the figure."""
        self._figure.bar(x=x, y=y, color=color, label=label)

    def hist(
        self: Figure,
        data: List[float],
        bins: int = 10,
        density: bool = False,
        color: str = 'blue',
        label: Optional[str] = None,
    ) -> None:
        """Add a histogram to the figure."""
        hist_vals, bin_edges = histogram(data, bins=bins, density=density)
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self._figure.bar(x=x, y=hist_vals, color=color, label=label)

    def draw(self: Figure) -> None:
        """Render the plot to terminal."""
        print()  # Leave a space
        self._figure.show()


class TimeGranularity(Enum):
    """Detected granularity of time range for axis labeling."""

    SECONDS = auto()    # Range < 2 minutes
    MINUTES = auto()    # Range < 2 hours
    HOURS = auto()      # Range < 2 days
    DAYS = auto()       # Range < 2 months
    MONTHS = auto()     # Range < 2 years
    YEARS = auto()      # Range >= 2 years


# TODO: Implement TimeTickResult dataclass
# TODO: Implement detect_time_granularity()
# TODO: Implement generate_time_ticks()
# TODO: Implement _nice_step() helper


def generate_time_ticks(
    min_epoch: float,
    max_epoch: float,
    max_ticks: int = 10,
) -> dict:
    """
    Generate smart time-series ticks based on the data range.

    TODO: Port full implementation from tplot/test_tplot.py

    Returns dict with:
        - tick_epochs: List[float] - epoch seconds for tick positions
        - tick_labels: List[str] - primary labels for each tick
        - secondary_labels: List[Tuple[float, str]] - (position, label) for date markers
        - granularity: TimeGranularity
        - xlabel_suffix: str - e.g., "(hours)" or "(HH:MM)"
    """
    # TODO: Implement full adaptive tick generation
    # For now, return minimal stub
    return {
        'tick_epochs': [],
        'tick_labels': [],
        'secondary_labels': [],
        'granularity': TimeGranularity.HOURS,
        'xlabel_suffix': '',
    }


class TimeSeriesFigure(Figure):
    """
    Extended Figure with smart time-series tick handling.

    Provides custom x-axis tick label formatting based on data range
    granularity. Supports secondary label row for hierarchical time
    display (e.g., hours with date markers below).

    TODO: Port full implementation from tplot/test_tplot.py
    """

    _x_tick_formatter: Optional[Callable[[float], str]]
    _custom_xtick_values: Optional[List[float]]
    _secondary_xlabel: Optional[str]

    def __init__(
        self: TimeSeriesFigure,
        *args,
        x_tick_formatter: Optional[Callable[[float], str]] = None,
        x_tick_values: Optional[List[float]] = None,
        secondary_xlabel: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize with optional custom tick formatting."""
        self._x_tick_formatter = x_tick_formatter
        self._custom_xtick_values = x_tick_values
        self._secondary_xlabel = secondary_xlabel
        super().__init__(*args, **kwargs)

    def _setup(self: TimeSeriesFigure) -> None:
        """
        Create underlying tplot figure with custom tick support.

        TODO: Override tplot.Figure methods for custom axis drawing
        """
        # For now, use base Figure setup
        # TODO: Implement custom TimeSeriesFigure class that overrides
        # _xax_height(), _fmt_x(), and _draw_x_axis()
        super()._setup()
