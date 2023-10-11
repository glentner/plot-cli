# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Configuration management."""


# type annotations
from typing import Union

# standard libs
import os
import sys
import functools
import traceback
from datetime import datetime

# external libs
from cmdkit.config import Configuration
from cmdkit.namespace import Namespace
from cmdkit.logging import Logger, level_by_name, logging_styles
from cmdkit.ansi import magenta, bold, faint
from cmdkit.app import exit_status


# public interface
__all__ = ['default_config', 'config', 'context', 'write_traceback']


default_config = {
    'logging': {
        'level': 'warning',
        **logging_styles.get('default'),
    },
}


def display_critical(error: Union[Exception, str], module: str = None) -> None:
    """Apply basic formatting to exceptions (i.e., without logging)."""
    text = error if isinstance(error, str) else f'{error.__class__.__name__}: {error}'
    name = faint(f'[{module or "plot"}]')
    print(f'{bold(magenta("CRITICAL"))} {name} {text}', file=sys.stderr)


def traceback_filepath(path: Namespace = None) -> str:
    """Construct filepath for writing traceback."""
    path = path or context.default_path
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(path.log, f'exception-{time}.log')


def write_traceback(exc: Exception, site: Namespace = None, logger: Logger = None,
                    status: int = exit_status.uncaught_exception, module: str = None) -> int:
    """Write exception to file and return exit code."""
    write = functools.partial(display_critical, module=module) if not logger else logger.critical
    path = traceback_filepath(site)
    with open(path, mode='w') as stream:
        print(traceback.format_exc(), file=stream)
    write(f'{exc.__class__.__name__}: ' + str(exc).replace('\n', ' - '))
    write(f'Exception traceback written to {path}')
    return status


try:
    context, config = Configuration.from_context('plot', create_dirs=True,
                                                 config_format='toml', default_config=default_config)
    Logger.default(__name__,
                   format=config.logging.format,
                   level=level_by_name.get(config.logging.level.upper()))
except Exception as err:
    write_traceback(err, module=__name__)
    sys.exit(exit_status.bad_config)
