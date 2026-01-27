# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Plotting interface using tplot."""


# Type annotations
from __future__ import annotations
from typing import Tuple, Optional, List, Callable

# Standard libs
import math
import logging
from enum import Enum, auto
from datetime import datetime
from dataclasses import dataclass

# External libs
import tplot
import tplot.utils as tplot_utils
from numpy import histogram

# Public interface
__all__ = [
    'Figure',
    'TimeSeriesFigure',
    'TimeGranularity',
    'TimeTickResult',
    'generate_time_ticks',
    'detect_time_granularity',
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


@dataclass
class TimeTickResult:
    """Result of time tick generation."""

    tick_epochs: List[float]                    # Epoch seconds for tick positions
    tick_labels: List[str]                      # Primary labels for each tick
    secondary_labels: List[Tuple[float, str]]   # (position, label) for year/date markers
    granularity: TimeGranularity
    xlabel_suffix: str                          # e.g., "(hours)" or "(HH:MM)"


def detect_time_granularity(min_epoch: float, max_epoch: float) -> TimeGranularity:
    """Detect appropriate granularity based on time range."""
    range_seconds = max_epoch - min_epoch

    if range_seconds < 120:  # < 2 minutes
        return TimeGranularity.SECONDS
    elif range_seconds < 7200:  # < 2 hours
        return TimeGranularity.MINUTES
    elif range_seconds < 172800:  # < 2 days
        return TimeGranularity.HOURS
    elif range_seconds < 5184000:  # < 60 days (~2 months)
        return TimeGranularity.DAYS
    elif range_seconds < 63072000:  # < 2 years
        return TimeGranularity.MONTHS
    else:
        return TimeGranularity.YEARS


def _nice_step(raw_step: float, nice_values: List[float]) -> float:
    """Round step to a 'nice' value."""
    if raw_step <= 0:
        return nice_values[0]
    for nv in nice_values:
        if nv >= raw_step:
            return nv
    return nice_values[-1]


def _epoch_to_datetime(epoch: float, utc: bool = True) -> datetime:
    """Convert epoch seconds to datetime, optionally using UTC."""
    if utc:
        return datetime.utcfromtimestamp(epoch)
    return datetime.fromtimestamp(epoch)


def _datetime_to_epoch(dt: datetime, utc: bool = True) -> float:
    """Convert datetime to epoch seconds."""
    # Note: For UTC datetimes without tzinfo, we need to calculate manually
    if utc:
        from calendar import timegm
        return float(timegm(dt.timetuple()))
    return dt.timestamp()


def generate_time_ticks(
    min_epoch: float,
    max_epoch: float,
    max_ticks: int = 10,
    utc: bool = True,
) -> TimeTickResult:
    """
    Generate smart time-series ticks based on the data range.

    Args:
        min_epoch: Minimum epoch timestamp (seconds)
        max_epoch: Maximum epoch timestamp (seconds)
        max_ticks: Maximum number of ticks to generate
        utc: If True, treat epochs as UTC (default for pandas timestamps)

    Returns TimeTickResult with tick positions, labels, and formatting info.
    """
    granularity = detect_time_granularity(min_epoch, max_epoch)

    # Convert to datetime using consistent UTC handling
    min_dt = _epoch_to_datetime(min_epoch, utc)
    max_dt = _epoch_to_datetime(max_epoch, utc)

    tick_epochs: List[float] = []
    tick_labels: List[str] = []
    secondary_labels: List[Tuple[float, str]] = []
    xlabel_suffix = ""

    if granularity == TimeGranularity.SECONDS:
        # Ticks every N seconds, labels as :SS or MM:SS
        step = _nice_step((max_epoch - min_epoch) / max_ticks, [1, 2, 5, 10, 15, 30])
        start = math.ceil(min_epoch / step) * step
        t = start
        while t <= max_epoch:
            tick_epochs.append(t)
            dt = _epoch_to_datetime(t, utc)
            tick_labels.append(f":{dt.second:02d}")
            t += step
        xlabel_suffix = "(MM:SS)"

    elif granularity == TimeGranularity.MINUTES:
        # Ticks every N minutes, labels as HH:MM
        step = _nice_step((max_epoch - min_epoch) / max_ticks / 60, [1, 2, 5, 10, 15, 30]) * 60
        start = math.ceil(min_epoch / step) * step
        t = start
        while t <= max_epoch:
            tick_epochs.append(t)
            dt = _epoch_to_datetime(t, utc)
            tick_labels.append(f"{dt.hour:02d}:{dt.minute:02d}")
            t += step
        xlabel_suffix = "(HH:MM)"

    elif granularity == TimeGranularity.HOURS:
        # Ticks every N hours, labels as HH:00
        step = _nice_step((max_epoch - min_epoch) / max_ticks / 3600, [1, 2, 3, 4, 6, 12]) * 3600
        # Align to hour boundaries
        start_dt = datetime(min_dt.year, min_dt.month, min_dt.day, min_dt.hour)
        start = _datetime_to_epoch(start_dt, utc)
        if start < min_epoch:
            start += step
        t = start
        prev_date = None
        while t <= max_epoch + step * 0.1:
            tick_epochs.append(t)
            dt = _epoch_to_datetime(t, utc)
            tick_labels.append(f"{dt.hour:02d}:00")
            # Add date markers when date changes
            curr_date = dt.date()
            if prev_date is not None and curr_date != prev_date:
                secondary_labels.append((t, dt.strftime("%Y-%m-%d")))
            prev_date = curr_date
            t += step
        xlabel_suffix = "(HH:MM)"
        # Add start date as secondary label if not already there
        if not secondary_labels:
            secondary_labels.append((min_epoch, min_dt.strftime("%Y-%m-%d")))

    elif granularity == TimeGranularity.DAYS:
        # Ticks every N days, labels as MM-DD
        step = _nice_step((max_epoch - min_epoch) / max_ticks / 86400, [1, 2, 7, 14]) * 86400
        # Align to day boundaries
        start_dt = datetime(min_dt.year, min_dt.month, min_dt.day)
        start = _datetime_to_epoch(start_dt, utc)
        if start < min_epoch:
            start += step
        t = start
        prev_month = None
        while t <= max_epoch + step * 0.1:
            tick_epochs.append(t)
            dt = _epoch_to_datetime(t, utc)
            tick_labels.append(f"{dt.month:02d}-{dt.day:02d}")
            # Add year markers when year changes
            if prev_month is not None and dt.month != prev_month and dt.month == 1:
                secondary_labels.append((t, str(dt.year)))
            prev_month = dt.month
            t += step
        xlabel_suffix = "(MM-DD)"
        if not secondary_labels:
            secondary_labels.append((min_epoch, str(min_dt.year)))

    elif granularity == TimeGranularity.MONTHS:
        # Ticks every N months, labels as month name
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        range_months = (max_dt.year - min_dt.year) * 12 + (max_dt.month - min_dt.month)
        step_months = int(_nice_step(range_months / max_ticks, [1, 2, 3, 6]))

        # Start at first of month
        curr = datetime(min_dt.year, min_dt.month, 1)
        prev_year = None
        curr_epoch = _datetime_to_epoch(curr, utc)
        while curr_epoch <= max_epoch:
            if curr_epoch >= min_epoch:
                tick_epochs.append(curr_epoch)
                tick_labels.append(month_names[curr.month - 1])
                # Add year markers
                if prev_year is not None and curr.year != prev_year:
                    secondary_labels.append((curr_epoch, str(curr.year)))
                prev_year = curr.year
            # Advance by step_months
            new_month = curr.month + step_months
            new_year = curr.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            curr = datetime(new_year, new_month, 1)
            curr_epoch = _datetime_to_epoch(curr, utc)
        xlabel_suffix = ""
        if not secondary_labels and tick_epochs:
            secondary_labels.append((tick_epochs[0], str(min_dt.year)))

    elif granularity == TimeGranularity.YEARS:
        # Ticks every N years, labels as year
        range_years = max_dt.year - min_dt.year
        step_years = int(_nice_step(range_years / max_ticks, [1, 2, 5, 10, 20, 50, 100]))
        step_years = max(1, step_years)

        # Start at round year
        start_year = (min_dt.year // step_years) * step_years
        if start_year < min_dt.year:
            start_year += step_years

        year = start_year
        while year <= max_dt.year:
            dt = datetime(year, 1, 1)
            tick_epochs.append(_datetime_to_epoch(dt, utc))
            tick_labels.append(str(year))
            year += step_years
        xlabel_suffix = ""

    return TimeTickResult(
        tick_epochs=tick_epochs,
        tick_labels=tick_labels,
        secondary_labels=secondary_labels,
        granularity=granularity,
        xlabel_suffix=xlabel_suffix,
    )


class _CustomTplotFigure(tplot.Figure):
    """
    Extended tplot.Figure with custom x-axis tick formatting.

    Overrides internal methods to support custom tick labels and
    secondary label rows for hierarchical time display.
    """

    def __init__(
        self,
        *args,
        x_tick_formatter: Optional[Callable[[float], str]] = None,
        x_tick_values: Optional[List[float]] = None,
        secondary_xlabel: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._x_tick_formatter = x_tick_formatter
        self._custom_xtick_values = x_tick_values
        self._secondary_xlabel = secondary_xlabel

    def _xax_height(self) -> int:
        """Account for secondary label row."""
        base = 2 + bool(self._xlabel)
        if self._secondary_xlabel:
            base += 1
        return base

    def _fmt_x(self, value) -> str:
        """Format x-axis tick value."""
        if self._x_tick_formatter:
            return self._x_tick_formatter(value)
        # Smart default formatting - prefer integers
        if isinstance(value, (int, float)) and value == int(value):
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    def _draw_x_axis(self) -> None:
        """Override to support custom x-axis formatting and secondary label row."""
        # Use custom tick values if provided, filtered to data range
        if self._custom_xtick_values:
            xmin, xmax = self._xtick_values[0], self._xtick_values[-1]
            tick_values = [v for v in self._custom_xtick_values if xmin <= v <= xmax]
            if not tick_values:
                tick_values = self._xtick_values
        else:
            tick_values = self._xtick_values

        tick_positions = [round(v) for v in self._xscale.transform(tick_values)]
        labels = [self._fmt_x(v) for v in tick_values]

        # Draw axis line
        axis_start = round(self._xscale.transform(self._xtick_values[0]))
        axis_end = round(self._xscale.transform(self._xtick_values[-1]))
        axis_row = -self._xax_height()
        self._canvas[axis_row, axis_start:axis_end] = "─"

        # Draw ticks
        for tick_pos in tick_positions:
            self._canvas[axis_row, tick_pos] = "┬"

        # Draw primary labels
        anchors = tplot_utils._optimize_xticklabel_anchors(
            tick_positions=tick_positions, labels=labels, width=self.width
        )
        for (start, end), label in zip(anchors, labels):
            label = label[: end - start]
            self._canvas[axis_row + 1, start:end] = list(label)

        # Draw secondary label row (e.g., date below hours)
        if self._secondary_xlabel:
            self._center_draw(self._secondary_xlabel, self._canvas[axis_row + 2, axis_start:axis_end])

        # Draw axis label
        if self._xlabel:
            xlabel = self._xlabel[: axis_end - axis_start]
            self._center_draw(xlabel, self._canvas[-1, axis_start:axis_end])


class TimeSeriesFigure(Figure):
    """
    Extended Figure with smart time-series tick handling.

    Provides custom x-axis tick label formatting based on data range
    granularity. Supports secondary label row for hierarchical time
    display (e.g., hours with date markers below).
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
        """Create custom tplot figure with extended tick support."""
        width = height = None
        if self.size:
            width, height = self.size
        self._figure = _CustomTplotFigure(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            width=width,
            height=height,
            legendloc=self.legend,
            x_tick_formatter=self._x_tick_formatter,
            x_tick_values=self._custom_xtick_values,
            secondary_xlabel=self._secondary_xlabel,
        )
