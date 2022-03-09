# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Platform specific file paths and initialization."""


# standard libs
import os

# external libs
from cmdkit.config import Namespace

# public interface
__all__ = ['home', 'site', 'path']


home = os.path.expanduser('~')
site = os.path.join(home, '.plot')
path = Namespace({
    'lib': os.path.join(site, 'lib'),
    'log': os.path.join(site, 'log'),
    'config': os.path.join(site, 'config.toml')
})


# automatically initialize default site directories
os.makedirs(path.lib, exist_ok=True)
os.makedirs(path.log, exist_ok=True)
