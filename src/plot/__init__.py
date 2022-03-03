# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point."""


# type annotations
from __future__ import annotations

# standard libs
import sys
import logging
from functools import partial

# external libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import ConfigurationError

# internal libs
from plot.core.exceptions import write_traceback
from plot.core.logging import initialize_logging

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


class PlotApp(Application):
    """Main application class."""

    interface = Interface(APP_NAME, APP_USAGE, APP_HELP)
    interface.add_argument('-v', '--version', action='version', version=__version__)

    source: str
    interface.add_argument('source')

    log_critical = log.critical
    log_exception = log.exception

    exceptions = {
        ConfigurationError: partial(write_traceback, logger=log, status=exit_status.bad_config),
        Exception: partial(write_traceback, logger=log)
    }

    def run(self: PlotApp) -> None:
        """Run the program."""
        log.info('Started')


def main() -> int:
    """Entry-point for console application."""
    initialize_logging()
    return PlotApp.main(sys.argv[1:])
