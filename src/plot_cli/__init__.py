# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point."""


# Type annotations
from __future__ import annotations
from typing import Tuple, List, Optional, Any

# Standard libs
import os
import sys
from importlib.metadata import version as get_version
from itertools import cycle
from functools import partial, cached_property
from shutil import get_terminal_size

# External libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import ConfigurationError
from cmdkit.logging import Logger
from pandas import DataFrame

# Internal libs
from plot_cli.config import write_traceback
from plot_cli.query import QueryBuilder
from plot_cli.plot import Figure, TimeSeriesFigure, generate_time_ticks

# Public interface
__all__ = ['main', 'PlotApp', '__version__', ]

# Metadata
__version__ = get_version(__name__)

# Application logger
log = Logger.with_name(__name__)


APP_NAME = os.path.basename(sys.argv[0])
APP_USAGE = f"""\
Usage:
  {APP_NAME} [-h] [-v] [FILE] [-x NAME] [-y NAME] [--line | --hist] ...
  Simple command-line plotting tool.\
"""

APP_HELP = f"""\
{APP_USAGE}

Arguments:
  FILE                        Path to data file (default: <stdin>).

Options:
  -x, --xdata        NAME     Field to use for x-axis.
  -y, --ydata        NAME...  Field(s) for y-axis.
      --format       NAME     Input format: csv, json, ndjson, parquet (default: auto).
      --plot-type    NAME     Type of plot to create (default: line).
      --line                  Alias for --plot-type=line
      --hist                  Alias for --plot-type=hist

Filtering:
      --where        EXPR     SQL WHERE clause for filtering.
      --after        TIME     Filter rows after timestamp (timeseries only).
      --before       TIME     Filter rows before timestamp (timeseries only).

Formatting:
      --xlabel       NAME     Content for x-axis label.
      --ylabel       NAME     Content for y-axis label.
  -t, --title        NAME     Content for plot title.
  -c, --color        SEQ      Comma-separated color names (e.g., 'blue,red,green').
  -s, --size         SHAPE    Width and height in characters (default: terminal size).
  -l, --legend       POS      Legend position (default: 'bottomright').

Histogram:
  -b, --bins         NUM      Number of bins for histogram (default: 10).
  -d, --density               Display frequency as percentage.

Timeseries:
  -T, --timeseries            Treat x-axis as datetime.
  -S, --scale        SCALE    Re-scale datetime axis (e.g., +hours, -days).
  -B, --bucket       INTERVAL Time bucket interval (e.g., '1min', '15min', '1h').
  -F, --resample     FREQ     [DEPRECATED] Use -B/--bucket instead.
  -A, --agg-method   NAME     Aggregation method for bucketing.
      --mean                  Alias for --agg-method=mean.
      --sum                   Alias for --agg-method=sum.
      --count                 Alias for --agg-method=count.
      --max                   Alias for --agg-method=max.
      --min                   Alias for --agg-method=min.
      --first                 Alias for --agg-method=first.
      --last                  Alias for --agg-method=last.

Output:
      --json                  Output processed data as JSON (instead of plotting).
      --csv                   Output processed data as CSV (instead of plotting).

  -h, --help                  Show this message and exit.
  -v, --version               Show the version and exit.\
"""


# Cycle through colors
DEFAULT_COLORS = ['blue', 'yellow', 'green', 'red', 'magenta', 'cyan', ]


def color_list(spec: str, sep: str = ',') -> List[str]:
    """Split list of colors on separator."""
    return spec.strip().split(sep)


def split_size(spec: str, sep: str = ',') -> Tuple[int, int]:
    """Split size value (e.g., '100,20') into integers."""
    width, height = map(int, spec.strip().split(sep))
    return width, height


class PlotApp(Application):
    """Main application class."""

    interface = Interface(APP_NAME, APP_USAGE, APP_HELP)
    interface.add_argument('-v', '--version', action='version', version=__version__)

    source: str = '-'
    interface.add_argument('source', nargs='?', default=source)

    input_format: Optional[str] = None
    interface.add_argument('--format', default=None, dest='input_format',
                           choices=['csv', 'json', 'ndjson', 'parquet'])

    plot_type: str = 'line'
    plot_type_interface = interface.add_mutually_exclusive_group()
    plot_type_interface.add_argument('--plot-type', default=plot_type, choices=['line', 'hist'])
    plot_type_interface.add_argument('--line', action='store_const', const='line', dest='plot_type')
    plot_type_interface.add_argument('--hist', action='store_const', const='hist', dest='plot_type')

    xdata: Optional[str] = None
    interface.add_argument('-x', '--xdata', default=None)

    ydata: List[str] = []
    interface.add_argument('-y', '--ydata', nargs='*', default=[])

    # Filtering options
    where_clause: Optional[str] = None
    interface.add_argument('--where', default=None, dest='where_clause')

    after_datetime: Optional[str] = None
    interface.add_argument('--after', default=None, dest='after_datetime')

    before_datetime: Optional[str] = None
    interface.add_argument('--before', default=None, dest='before_datetime')

    # Timeseries options
    timeseries: bool = False
    interface.add_argument('-T', '--timeseries', action='store_true')

    timeseries_scale: Optional[str] = None
    interface.add_argument('-S', '--scale', default=None, dest='timeseries_scale')

    bucket_interval: Optional[str] = None
    interface.add_argument('-B', '--bucket', default=None, dest='bucket_interval')

    resample_freq: Optional[str] = None
    interface.add_argument('-F', '--resample', default=None, dest='resample_freq')

    agg_method: Optional[str] = None
    agg_interface = interface.add_mutually_exclusive_group()
    agg_interface.add_argument('-A', '--agg-method', default=None,
                               choices=['mean', 'sum', 'count', 'max', 'min', 'first', 'last'])
    agg_interface.add_argument('--mean', action='store_const', const='mean', dest='agg_method')
    agg_interface.add_argument('--sum', action='store_const', const='sum', dest='agg_method')
    agg_interface.add_argument('--count', action='store_const', const='count', dest='agg_method')
    agg_interface.add_argument('--min', action='store_const', const='min', dest='agg_method')
    agg_interface.add_argument('--max', action='store_const', const='max', dest='agg_method')
    agg_interface.add_argument('--first', action='store_const', const='first', dest='agg_method')
    agg_interface.add_argument('--last', action='store_const', const='last', dest='agg_method')

    # Histogram options
    hist_bins: int = 10
    interface.add_argument('-b', '--bins', type=int, default=hist_bins, dest='hist_bins')

    hist_density: bool = False
    interface.add_argument('-d', '--density', action='store_true', dest='hist_density')

    # Formatting options
    colors: List[str] = DEFAULT_COLORS
    interface.add_argument('-c', '--color', type=color_list, default=colors, dest='colors')

    title: Optional[str] = None
    interface.add_argument('-t', '--title', default=None)

    xlabel: Optional[str] = None
    interface.add_argument('-X', '--xlabel', default=None)

    ylabel: Optional[str] = None
    interface.add_argument('-Y', '--ylabel', default=None)

    size: Optional[Tuple[int, int]] = None
    interface.add_argument('-s', '--size', default=None, type=split_size)

    legend: str = 'bottomright'
    interface.add_argument('-l', '--legend', default=legend,
                           choices=['bottomleft', 'bottomright', 'topleft', 'topright'])

    # Output mode options
    output_json: bool = False
    interface.add_argument('--json', action='store_true', dest='output_json')

    output_csv: bool = False
    interface.add_argument('--csv', action='store_true', dest='output_csv')

    log_critical = log.critical
    log_exception = log.exception

    exceptions = {
        ConfigurationError: partial(write_traceback, logger=log, status=exit_status.bad_config),
        Exception: partial(write_traceback, logger=log)
    }

    # Data loaded via QueryBuilder
    _dataframe: DataFrame | None = None
    _x_column: str | None = None

    def run(self: PlotApp) -> None:
        """Run the program."""
        self._handle_deprecated_options()
        self._load_data()
        if self.output_json:
            self._output_json()
        elif self.output_csv:
            self._output_csv()
        else:
            self._render_plot()

    def _handle_deprecated_options(self: PlotApp) -> None:
        """Handle deprecated CLI options with warnings."""
        if self.resample_freq:
            log.warning('-F/--resample is deprecated, use -B/--bucket instead')
            if not self.bucket_interval:
                self.bucket_interval = self.resample_freq

    def _load_data(self: PlotApp) -> None:
        """Load and transform data using DuckDB QueryBuilder."""
        log.info(f'Loading data from {self.source}')
        query = QueryBuilder(
            source=self.source,
            format=self.input_format,
            x_column=self.xdata,
            y_columns=self.ydata if self.ydata else None,
            where_clause=self.where_clause,
            after_datetime=self.after_datetime,
            before_datetime=self.before_datetime,
            bucket_interval=self.bucket_interval,
            agg_method=self.agg_method,
            timeseries=self.timeseries,
            scale=self.timeseries_scale,
        )
        self._dataframe = query.execute()
        self._x_column = self._dataframe.columns[0]
        log.debug(f'Loaded {len(self._dataframe)} rows')

    def _output_json(self: PlotApp) -> None:
        """Output data as JSON."""
        # TODO: Implement JSON output
        print(self._dataframe.to_json(orient='records', indent=2))

    def _output_csv(self: PlotApp) -> None:
        """Output data as CSV."""
        # TODO: Implement CSV output
        print(self._dataframe.to_csv(index=False))

    def _render_plot(self: PlotApp) -> None:
        """Render the plot to terminal."""
        y_columns = list(self._dataframe.columns[1:])

        # Get x values - convert to epoch if using TimeSeriesFigure
        if self._use_timeseries_figure:
            x_values = [v.timestamp() for v in self._dataframe[self._x_column]]
        else:
            x_values = self._dataframe[self._x_column].tolist()

        for column, color in zip(y_columns, cycle(self.colors)):
            if self.plot_type == 'line':
                self.figure.line(
                    x=x_values,
                    y=self._dataframe[column].tolist(),
                    color=color,
                    label=column,
                )
            else:
                self.figure.hist(
                    data=self._dataframe[column].tolist(),
                    bins=self.hist_bins,
                    density=self.hist_density,
                    color=color,
                    label=column,
                )
        self.figure.draw()

    @cached_property
    def figure(self: PlotApp) -> Figure:
        """Create and return the Figure instance."""
        # Use TimeSeriesFigure for datetime x-axis without scale conversion
        if self._use_timeseries_figure:
            x_values = self._dataframe[self._x_column].tolist()
            # Convert timestamps to epoch for tick generation
            min_epoch = x_values[0].timestamp()
            max_epoch = x_values[-1].timestamp()
            ticks = generate_time_ticks(min_epoch, max_epoch)

            # Build formatter based on tick labels
            tick_map = dict(zip(ticks.tick_epochs, ticks.tick_labels))

            def formatter(epoch: float) -> str:
                # Find closest tick label or format generically
                if epoch in tick_map:
                    return tick_map[epoch]
                from datetime import datetime
                # Use UTC to match pandas timestamp behavior
                dt = datetime.utcfromtimestamp(epoch)
                if ticks.granularity.name == 'HOURS':
                    return f"{dt.hour:02d}:{dt.minute:02d}"
                elif ticks.granularity.name == 'MINUTES':
                    return f"{dt.hour:02d}:{dt.minute:02d}"
                elif ticks.granularity.name == 'DAYS':
                    return f"{dt.month:02d}-{dt.day:02d}"
                return str(epoch)

            secondary_label = None
            if ticks.secondary_labels:
                secondary_label = ticks.secondary_labels[0][1]

            return TimeSeriesFigure(
                title=self.plot_title,
                xlabel=self.plot_xlabel,
                ylabel=self.plot_ylabel,
                size=self.plot_size,
                legend=self.legend,
                x_tick_formatter=formatter,
                x_tick_values=ticks.tick_epochs,
                secondary_xlabel=secondary_label,
            )

        return Figure(
            title=self.plot_title,
            xlabel=self.plot_xlabel,
            ylabel=self.plot_ylabel,
            size=self.plot_size,
            legend=self.legend,
        )

    @property
    def _use_timeseries_figure(self: PlotApp) -> bool:
        """Whether to use TimeSeriesFigure for smart datetime axis labels."""
        # Use TimeSeriesFigure when:
        # - timeseries flag is set
        # - scale is NOT set (scale converts to numeric, so no datetime formatting needed)
        # - x-axis data is datetime type
        if not self.timeseries or self.timeseries_scale:
            return False
        # Check if x-column is datetime
        x_dtype = str(self._dataframe[self._x_column].dtype)
        return 'datetime' in x_dtype

    @cached_property
    def plot_title(self: PlotApp) -> str:
        """Plot title."""
        if self.title:
            return self.title
        return os.path.basename(self.source) if self.source != '-' else 'stdin'

    @cached_property
    def plot_xlabel(self: PlotApp) -> str:
        """X-axis label."""
        if self.xlabel:
            return self.xlabel
        if self._x_column:
            if self.timeseries and self.timeseries_scale:
                return f'{self._x_column} ({self.timeseries_scale})'
            return self._x_column
        return ''

    @cached_property
    def plot_ylabel(self: PlotApp) -> str:
        """Y-axis label."""
        if self.ylabel:
            return self.ylabel
        if self.plot_type == 'hist' and self.hist_density:
            return 'percent'
        return 'value'

    @cached_property
    def plot_size(self: PlotApp) -> Tuple[int, int]:
        """Plot size in width, height."""
        if self.size:
            return self.size
        width, height = get_terminal_size()
        return width - 10, height - 8


def main() -> int:
    """Entry-point for console application."""
    return PlotApp.main(sys.argv[1:])
