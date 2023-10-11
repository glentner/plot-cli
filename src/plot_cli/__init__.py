# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point."""


# type annotations
from __future__ import annotations
from typing import Tuple, Dict, List, Type, Optional, Any

# standard libs
import os
import sys
import re
from importlib.metadata import version as get_version
from itertools import cycle
from functools import partial, cached_property
from shutil import get_terminal_size

# external libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import ConfigurationError
from cmdkit.logging import Logger
from cmdkit.ansi import italic, bold, COLOR_STDOUT, colorize_usage as default_colorize_usage

# internal libs
from plot_cli.config import write_traceback
from plot_cli.provider import PlotInterface, TPlot, TPlotLine, TPlotHist
from plot_cli.data import DataSet

# public interface
__all__ = ['main', 'PlotApp', '__version__', ]

# metadata
__version__ = get_version(__name__)

# application logger
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
      --backend      NAME     Method for plotting data (default: tplot).
      --plot-type    NAME     Type of plot_cli to create (default: line).
      --line                  Alias for --plot-type=line
      --hist                  Alias for --plot-type=hist

Formatting:
      --xlabel       NAME     Content for x-axis label.
      --ylabel       NAME     Content for y-axis label.
  -t, --title        NAME     Content for plot_cli title.
  -c, --color        SEQ      Comma-separated color names (e.g., 'blue,red,green').
  -s, --size         SHAPE    Width and height in pixels (default: 100,20).
  -l, --legend       POS      Legend position (default: 'bottomright').

Histogram:
  -b, --bins         NUM      Number of bins for histogram (default: 10).
  -d, --density               Display frequency as percentage.

Timeseries:
  -T, --timeseries  [NAME]    Fields to parse as date and/or time.
  -S, --scale        SCALE    Re-scale datetime axis (e.g., +/- hours).
  -F, --resample     FREQ     Re-sample on some frequency (e.g., '1min').
  -A, --agg-method   NAME     Aggregation method (required for --resample).
      --mean                  Alias for --agg-method=mean.
      --sum                   Alias for --agg-method=sum.
      --count                 Alias for --agg-method=count.
      --max                   Alias for --agg-method=max.
      --min                   Alias for --agg-method=min.

  -h, --help                  Show this message and exit.
  -v, --version               Show the version and exit.\
"""


# Mapping of provider and plot_cli type interfaces
plot_interface: Dict[str, Dict[str, Type[PlotInterface]]] = {
    'tplot': {
        'line': TPlotLine,
        'hist': TPlotHist,
    },
}


# Cycle through colors
default_colors = ['blue', 'yellow', 'green', 'red', 'magenta', 'cyan', ]


def color_list(spec: str, sep: str = ',') -> List[str]:
    """Split list of colors on separator."""
    return spec.strip().split(sep)


def split_size(spec: str, sep: str = ',') -> Tuple[float, float]:
    """Split size value (e.g., '6,10') into floats."""
    width, height = map(float, spec.strip().split(sep))
    return width, height


# Look-around pattern to negate matches within quotation marks
# Whole quotations are formatted together
NOT_QUOTED = (
    r'(?=([^"]*"[^"]*")*[^"]*$)' +
    r"(?=([^']*'[^']*')*[^']*$)" +
    r'(?=([^`]*`[^`]*`)*[^`]*$)'
)


def format_special_args(text: str) -> str:
    """Formatting special arguments."""
    metavars = ['SEQ', 'SHAPE', 'POS', 'SCALE', 'FREQ']
    metavars_pattern = r'\b(?P<arg>' + '|'.join(metavars) + r')\b'
    return re.sub(metavars_pattern + NOT_QUOTED, italic(r'\g<arg>'), text)


def format_headers(text: str) -> str:
    """Formatting section headers."""
    names = ['Formatting', 'Histogram', 'Timeseries']
    return re.sub(r'(?P<name>' + '|'.join(names) + r'):' + NOT_QUOTED, bold(r'\g<name>:'), text)


def colorize_usage(text: str) -> str:
    """Apply additional formatting to usage/help text."""
    if not COLOR_STDOUT:  # NOTE: usage is on stdout not stderr
        return text
    else:
        return default_colorize_usage(format_headers(format_special_args(text)))


class PlotApp(Application):
    """Main application class."""

    interface = Interface(APP_NAME, APP_USAGE, APP_HELP, formatter=colorize_usage)
    interface.add_argument('-v', '--version', action='version', version=__version__)

    source: str = '-'
    interface.add_argument('source', nargs='?', default=source)

    backend: str = 'tplot'
    backend_interface = interface.add_mutually_exclusive_group()
    backend_interface.add_argument('--backend', default=backend, choices=list(plot_interface))
    backend_interface.add_argument('--terminal', '--tplot', action='store_const', const=backend, dest='backend')

    plot_type: str = 'line'
    plot_type_interface = interface.add_mutually_exclusive_group()
    plot_type_interface.add_argument('--plot-type', default=plot_type, choices=['line', ])
    plot_type_interface.add_argument('--line', action='store_const', const='line', dest='plot_type')
    plot_type_interface.add_argument('--hist', action='store_const', const='hist', dest='plot_type')

    xdata: Optional[str] = None
    interface.add_argument('-x', '--xdata', default=None)

    ydata: Optional[List[str]] = []
    interface.add_argument('-y', '--ydata', nargs='*', default=[])

    timeseries: bool = False
    interface.add_argument('-T', '--timeseries', action='store_true')

    timeseries_scale: Optional[str] = None
    interface.add_argument('-S', '--scale', default=None, dest='timeseries_scale')

    resample_freq: str = None
    interface.add_argument('-F', '--resample', default=None, dest='resample_freq')

    agg_method: str = None
    agg_interface = interface.add_mutually_exclusive_group()
    agg_interface.add_argument('-A', '--agg-method', default=None, choices=['mean', 'sum', 'count', 'max', 'min'])
    agg_interface.add_argument('--mean', action='store_const', const='mean', dest='agg_method')
    agg_interface.add_argument('--sum', action='store_const', const='sum', dest='agg_method')
    agg_interface.add_argument('--count', action='store_const', const='count', dest='agg_method')
    agg_interface.add_argument('--min', action='store_const', const='min', dest='agg_method')
    agg_interface.add_argument('--max', action='store_const', const='max', dest='agg_method')

    hist_bins: int = 10
    interface.add_argument('-b', '--bins', type=int, default=hist_bins, dest='hist_bins')

    hist_density: bool = False
    interface.add_argument('-d', '--density', action='store_true', dest='hist_density')

    drop_missing: bool = False
    interface.add_argument('--drop-missing', action='store_true')

    colors: List[str] = default_colors
    interface.add_argument('-c', '--color', type=color_list, default=colors)

    title: Optional[str] = None
    interface.add_argument('-t', '--title', default=None)

    xlabel: Optional[str] = None
    interface.add_argument('-X', '--xlabel', default=None)

    ylabel: Optional[str] = None
    interface.add_argument('-Y', '--ylabel', default=None)

    size: Optional[Tuple[float, float]] = None
    interface.add_argument('-s', '--size', default=None, type=split_size)

    legend: str = 'bottomright'
    interface.add_argument('-l', '--legend', default=legend,
                           choices=['bottomleft', 'bottomright', 'topleft', 'topright'])

    log_critical = log.critical
    log_exception = log.exception

    exceptions = {
        ConfigurationError: partial(write_traceback, logger=log, status=exit_status.bad_config),
        Exception: partial(write_traceback, logger=log)
    }

    dataset: DataSet = None

    def run(self: PlotApp) -> None:
        """Run the program."""
        self.load_dataset()
        self.add_all(self.ydata or self.dataset.columns)
        self.draw()

    def draw(self: PlotApp) -> None:
        """Issue final render command to the finished plot_cli."""
        self.plotter.draw()

    @cached_property
    def plotter(self: PlotApp) -> PlotInterface:
        """Prepared plotting interface."""
        plot_type = plot_interface.get(self.backend).get(self.plot_type)
        plotter = plot_type(title=self.title or os.path.basename(self.source),
                            xlabel=self.plot_xlabel, ylabel=self.plot_ylabel,
                            size=self.plot_size, legend=self.legend)
        plotter.setup()
        return plotter

    @cached_property
    def plot_xlabel(self: PlotApp) -> str:
        """X-axis label."""
        if self.xlabel:
            return self.xlabel
        if self.timeseries and self.timeseries_scale:
            return f'{self.dataset.index.name} ({self.timeseries_scale})'
        if self.plot_type == 'hist':
            return ''
        else:
            return f'{self.dataset.index.name}'

    @cached_property
    def plot_ylabel(self: PlotApp) -> str:
        """Y-axis label."""
        if self.ylabel:
            return self.ylabel
        if self.plot_type == 'hist' and self.hist_density:
            return 'percent'
        else:
            return 'value'

    @cached_property
    def plot_size(self: PlotApp) -> Optional[Tuple[float, float]]:
        """Plot size in width, height."""
        if not self.size and self.backend != 'tplot':
            return None
        if self.size:
            return self.size
        else:
            width, height = get_terminal_size()
            return width - 10, height - 8

    def add_all(self: PlotApp, columns: List[str]) -> None:
        """Add each `column` to the plot_cli."""
        for column, color in zip(columns, cycle(self.colors)):
            self.plotter.add(self.dataset, **self.prepare_plot_options(column=column, color=color, label=column))

    def prepare_plot_options(self: PlotApp, **options: Any) -> Dict[str, Any]:
        """Prepare plot_cli-type specify parameters, forward `options`."""
        if self.plot_type == 'line':
            return options
        if self.plot_type == 'hist':
            return {'bins': self.hist_bins, 'density': self.hist_density, **options}

    def load_dataset(self: PlotApp) -> None:
        """Load the dataset."""
        if self.source == '-':
            log.info('Reading from <stdin>')
            self.dataset = DataSet.from_io(sys.stdin)
        else:
            log.info(f'Reading from {self.source}')
            self.dataset = DataSet.from_local(self.source)
        xcol = self.xdata or self.dataset.columns[0]
        if self.timeseries:
            ts_column = self.timeseries if self.timeseries != '-' else self.dataset.columns[0]
            log.info(f'Parsing timeseries ({ts_column})')
            self.dataset.set_type(xcol, 'datetime64[ns]')
        scale_for_index = self.timeseries_scale if not self.resample_freq else None
        self.dataset.set_index(xcol, timeseries_scale=scale_for_index)
        if self.resample_freq:
            self.dataset.resample(self.resample_freq, agg=self.agg_method, scale=self.timeseries_scale)
        if self.drop_missing:
            self.dataset.drop_missing()


def main() -> int:
    """Entry-point for console application."""
    return PlotApp.main(sys.argv[1:])
