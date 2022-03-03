# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Core exception handling."""


# type annotations
from typing import Union

# standard libraries
import os
import sys
import logging
import functools
import traceback
from datetime import datetime

# external libs
from cmdkit.app import exit_status
from cmdkit.config import Namespace

# internal libs
from plot.core.platform import path as default_path

# public interface
__all__ = ['display_critical', 'traceback_filepath', 'write_traceback', ]


def display_critical(error: Union[Exception, str], module: str = None) -> None:
    """Apply basic formatting to exceptions (i.e., without logging)."""
    text = error if isinstance(error, str) else f'{error.__class__.__name__}: {error}'
    name = '' if not module else f'[{module}]'
    print(f'CRITICAL {name} {text}', file=sys.stderr)


def traceback_filepath(path: Namespace = None) -> str:
    """Construct filepath for writing traceback."""
    path = path or default_path
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(path.log, f'exception-{time}.log')


def write_traceback(exc: Exception, site: Namespace = None, logger: logging.Logger = None,
                    status: int = exit_status.uncaught_exception, module: str = None) -> int:
    """Write exception to file and return exit code."""
    write = functools.partial(display_critical, module=module) if not logger else logger.critical
    path = traceback_filepath(site)
    with open(path, mode='w') as stream:
        print(traceback.format_exc(), file=stream)
    write(f'{exc.__class__.__name__}: ' + str(exc).replace('\n', ' - '))
    write(f'Exception traceback written to {path}')
    return status
