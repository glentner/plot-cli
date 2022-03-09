# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point."""


# type annotations
from __future__ import annotations
from typing import Tuple, Dict, List, Type, Optional, Any

# standard libs
import os
import sys
import logging
from itertools import cycle
from functools import partial, cached_property
from shutil import get_terminal_size

# external libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import ConfigurationError

# internal libs
from plot.core.exceptions import write_traceback
from plot.core.logging import initialize_logging
from plot.provider import PlotInterface, TPlot, TPlotLine, TPlotHist
from plot.data import DataSet

# public interface
__all__ = ['main', 'PlotApp', '__version__', ]

# metadata
__version__ = '0.1.0'

# application logger
log = logging.getLogger('plot')


APP_NAME = 'plot'
APP_USAGE = f"""\
usage: {APP_NAME} [-h] [-v] FILE ...
Simple command-line plotting tool.\
"""

APP_HELP = f"""\
{APP_USAGE}

arguments:
FILE                              Path to data file (default: <stdin>).

options:
-x, --xdata             NAME      Field to use for x-axis.
-y, --ydata             NAME...   Field(s) for y-axis.
-b, --backend           NAME      Method for plotting data (default: tplot).
    --plot-type         NAME      Type of plot to create (default: line).
    --line                        Alias for --plot-type=line
    --hist                        Alias for --plot-type=hist

formatting:
    --xlabel            STR       Content for x-axis label.
    --ylabel            STR       Content for y-axis label.
-t, --title             STR       Content for plot title.
-c, --color             SEQ       Comma-separated color names (e.g., blue,red,green).
-s, --size              SHAPE     Width and height in pixels (default: 100,20).

histogram:
    --bins              NUM       Number of bins for histogram (default: 10).
    --density                     Display frequency as percentage.

timeseries:
    --timeseries        [NAME]    Fields to parse as date and/or time.
    --timeseries-scale  SCALE     Re-scale datetime axis (e.g., +/- hours).
    --resample          FREQ      Re-sample on some frequency (e.g., '1min').
    --agg-method        NAME      Aggregation method (required for --resample).
    --mean                        Alias for --agg-method=mean.
    --sum                         Alias for --agg-method=sum.
    --count                       Alias for --agg-method=count.
    --max                         Alias for --agg-method=max.
    --min                         Alias for --agg-method=min.

-h, --help                        Show this message and exit.
-v, --version                     Show the version and exit.\
"""


# Mapping of provider and plot type interfaces
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


class PlotApp(Application):
    """Main application class."""

    interface = Interface(APP_NAME, APP_USAGE, APP_HELP)
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

    timeseries: str = None
    interface.add_argument('--timeseries', action='store_const', default=None, const='-')

    timeseries_scale: Optional[str] = None
    interface.add_argument('--timeseries-scale', default=None)

    resample_freq: str = None
    interface.add_argument('--resample', default=None, dest='resample_freq')

    agg_method: str = None
    agg_interface = interface.add_mutually_exclusive_group()
    agg_interface.add_argument('--agg-method', default=None, choices=['mean', 'sum', 'count', 'max', 'min'])
    agg_interface.add_argument('--mean', action='store_const', const='mean', dest='agg_method')
    agg_interface.add_argument('--sum', action='store_const', const='sum', dest='agg_method')
    agg_interface.add_argument('--count', action='store_const', const='count', dest='agg_method')
    agg_interface.add_argument('--min', action='store_const', const='min', dest='agg_method')
    agg_interface.add_argument('--max', action='store_const', const='max', dest='agg_method')

    hist_bins: int = 10
    interface.add_argument('-b', '--bins', type=int, default=hist_bins, dest='hist_bins')

    hist_density: bool = False
    interface.add_argument('--density', action='store_true', dest='hist_density')

    drop_missing: bool = False
    interface.add_argument('--drop-missing', action='store_true')

    colors: List[str] = default_colors
    interface.add_argument('-c', '--color', type=color_list, default=colors)

    title: Optional[str] = None
    interface.add_argument('-t', '--title', default=None)

    xlabel: Optional[str] = None
    interface.add_argument('--xlabel', default=None)

    ylabel: Optional[str] = None
    interface.add_argument('--ylabel', default=None)

    size: Optional[Tuple[float, float]] = None
    interface.add_argument('-s', '--size', default=None, type=split_size)

    legend: str = 'bottomleft'
    interface.add_argument('--legend', default='bottomleft')

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
        """Issue final render command to the finished plot."""
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
        else:
            return self.dataset.index.name

    @cached_property
    def plot_ylabel(self: PlotApp) -> str:
        """Y-axis label."""
        if self.ylabel:
            return self.ylabel
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
        """Add each `column` to the plot."""
        for column, color in zip(columns, cycle(self.colors)):
            self.plotter.add(self.dataset, **self.prepare_plot_options(column=column, color=color, label=column))

    def prepare_plot_options(self: PlotApp, **options: Any) -> Dict[str, Any]:
        """Prepare plot-type specify parameters, forward `options`."""
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
        if self.timeseries:
            ts_column = self.timeseries if self.timeseries != '-' else self.dataset.columns[0]
            log.info(f'Parsing timeseries ({ts_column})')
            self.dataset.set_type(ts_column, 'datetime64[ns]')
        scale_for_index = self.timeseries_scale if not self.resample_freq else None
        self.dataset.set_index(self.xdata or self.dataset.columns[0], timeseries_scale=scale_for_index)
        if self.resample_freq:
            self.dataset.resample(self.resample_freq, agg=self.agg_method, scale=self.timeseries_scale)
        if self.drop_missing:
            self.dataset.drop_missing()


def main() -> int:
    """Entry-point for console application."""
    initialize_logging()
    return PlotApp.main(sys.argv[1:])
