# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point."""


# type annotations
from __future__ import annotations
from typing import Tuple, Dict, List, Type, Optional, Union

# standard libs
import sys
import logging
from itertools import cycle
from functools import partial

# external libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import ConfigurationError

# internal libs
from plot.core.exceptions import write_traceback
from plot.core.logging import initialize_logging
from plot.provider import PlotInterface, TPlot, TPlotLine
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
FILE                   Path to data file (default: <stdin>).

options:
-h, --help             Show this message and exit.
-v, --version          Show the version and exit.\
"""


# Mapping of provider and plot type interfaces
plot_interface: Dict[str, Dict[str, Type[PlotInterface]]] = {
    'tplot': {
        'line': TPlotLine,
    },
}


# Cycle through colors
default_colors = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', ]


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
    backend_interface.add_argument('-b', '--backend', default=backend, choices=list(plot_interface))
    backend_interface.add_argument('--terminal', '--tplot', action='store_const', const=backend, dest='backend')

    plot_type: str = 'line'
    plot_type_interface = interface.add_mutually_exclusive_group()
    plot_type_interface.add_argument('--plot-type', default=plot_type, choices=['line', ])
    plot_type_interface.add_argument('--line', action='store_const', const='line', dest='plot_type')

    xdata: Optional[str] = None
    interface.add_argument('-x', '--xdata', default=None)

    ydata: Optional[List[str]] = []
    interface.add_argument('-y', '--ydata', nargs='*', default=[])

    parse_dates: Optional[Union[List[str], bool]] = None
    interface.add_argument('--parse-dates', nargs='*', default=None)

    datetime_offset: Optional[str] = None
    interface.add_argument('--offset', default=None, dest='datetime_offset')

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

    legend_loc: Optional[str] = 'bottomleft'
    interface.add_argument('--legend-loc', default=legend_loc)

    log_critical = log.critical
    log_exception = log.exception

    exceptions = {
        ConfigurationError: partial(write_traceback, logger=log, status=exit_status.bad_config),
        Exception: partial(write_traceback, logger=log)
    }

    def run(self: PlotApp) -> None:
        """Run the program."""
        dataset = self.load_dataset()
        plot_type = plot_interface.get(self.backend).get(self.plot_type)
        plotter = plot_type(title=self.title,
                            xlabel=self.xlabel or dataset.index.name,
                            ylabel=self.ylabel or 'value',
                            size=self.size, legendloc=self.legend_loc)
        plotter.setup()
        columns = self.ydata or dataset.columns
        for column, color in zip(columns, cycle(self.colors)):
            plotter.add(dataset, column=column, color=color, label=column)
        plotter.draw()

    def load_dataset(self: PlotApp) -> DataSet:
        """Load the dataset."""
        if self.parse_dates is None:
            parse_dates = False
        elif not self.parse_dates:
            parse_dates = True
        else:
            parse_dates = self.parse_dates
        if self.source == '-':
            log.info('Reading data from <stdin>')
            dataset = DataSet.from_io(sys.stdin, parse_dates=parse_dates)
        else:
            log.info(f'Reading data from {self.source}')
            dataset = DataSet.from_local(self.source, parse_dates=parse_dates)
        if self.xdata:
            dataset.set_index(self.xdata, datetime_offset=self.datetime_offset)
        else:
            dataset.set_index(dataset.columns[0], datetime_offset=self.datetime_offset)
        return dataset


def main() -> int:
    """Entry-point for console application."""
    initialize_logging()
    return PlotApp.main(sys.argv[1:])
