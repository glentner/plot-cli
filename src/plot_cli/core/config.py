# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Configuration management."""


# type annotations
from __future__ import annotations
from typing import Optional

# standard libs
import os
import sys
import functools

# external libs
from cmdkit.config import Namespace, Configuration, Environ, ConfigurationError
from cmdkit.app import exit_status

# internal libs
from plot_cli.core.platform import path
from plot_cli.core.exceptions import write_traceback

# public interface
__all__ = ['default', 'config', 'blame', ]


DEFAULT_LOGGING_STYLE = 'default'
LOGGING_STYLES = {
    'default': {
        'format': ('%(ansi_bold)s%(ansi_level)s%(levelname)8s%(ansi_reset)s %(ansi_faint)s[%(name)s]%(ansi_reset)s'
                   ' %(message)s'),
    },
    'system': {
        'format': '%(asctime)s.%(msecs)03d %(hostname)s %(levelname)8s [%(app_id)s] [%(name)s] %(message)s',
    },
    'detailed': {
        'format': ('%(ansi_faint)s%(asctime)s.%(msecs)03d %(hostname)s %(ansi_reset)s'
                   '%(ansi_level)s%(ansi_bold)s%(levelname)8s%(ansi_reset)s '
                   '%(ansi_faint)s[%(name)s]%(ansi_reset)s %(message)s'),
    }
}


# environment variables and configuration files are automatically
# depth-first merged with defaults
default = Namespace({
    'logging': {
        'level': 'warning',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'style': DEFAULT_LOGGING_STYLE,
        **LOGGING_STYLES.get(DEFAULT_LOGGING_STYLE),
    },
})


@functools.lru_cache(maxsize=None)
def load_file(filepath: str) -> Namespace:
    """Load configuration file manually."""
    try:
        if not os.path.exists(filepath):
            return Namespace({})
        else:
            return Namespace.from_toml(filepath)
    except Exception as err:
        raise ConfigurationError(f'(from file: {filepath}) {err.__class__.__name__}: {err}')


@functools.lru_cache(maxsize=None)
def load_env() -> Environ:
    """Load environment variables and expand hierarchy as namespace."""
    return Environ(prefix='PLOT').expand()


def load(**preload: Namespace) -> Configuration:
    """Load configuration from files and merge environment variables."""
    return Configuration(**{
        'default': default, **preload,  # NOTE: preloads _after_ defaults
        'user': load_file(path.config),
        'env': load_env(),
    })


try:
    config = load()
except Exception as error:
    write_traceback(error, module=__name__)
    sys.exit(exit_status.bad_config)


def blame(*varpath: str) -> Optional[str]:
    """Construct filename or variable assignment string based on precedent of `varpath`"""
    source = config.which(*varpath)
    if not source:
        return None
    if source in ('user', ):
        return f'from: {path.get(source).config}'
    elif source == 'env':
        return 'from: PLOT_' + '_'.join([node.upper() for node in varpath])
    else:
        return f'from: <{source}>'
