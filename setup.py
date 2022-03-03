# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Build and installation script."""


# standard libs
import re
from setuptools import setup, find_packages


# Long description from README.rst
with open('README.md', mode='r') as readme:
    long_description = readme.read()


# Package metadata by parsing __init__ module
with open('src/plot/__init__.py', mode='r') as source:
    content = source.read().strip()
    version = re.search('__version__' + r'\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)


# Core dependencies
DEPENDENCIES = ['cmdkit', 'toml', 'pandas', 'tplot', 'matplotlib', ]


setup(
    name             = 'plot',
    version          = version,
    author           = 'Geoffrey Lentner <glentner@purdue.edu>',
    description      = 'Simple Plotting Command-Line Tool',
    url              = 'https://github.com/glentner/plot-cli',
    license          = 'Apache Software License',
    packages         = find_packages('src'),
    package_dir      = {'': 'src', },
    long_description = long_description,
    classifiers      = ['Development Status :: 3 - Alpha',
                        'Programming Language :: Python :: 3.9',
                        'Programming Language :: Python :: 3.10', ],
    entry_points     = {'console_scripts': ['plot-cli=plot:main', ]},
    install_requires = DEPENDENCIES,
)
