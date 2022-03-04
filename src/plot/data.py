# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Data loading and transformations."""


# type annotations
from __future__ import annotations
from typing import List, IO, Union, Type

# standard libs
import re
import logging
from io import StringIO

# external libs
from pandas import DataFrame, Series, Index, read_csv

# public interface
__all__ = ['DataSet', ]

# module level logger
log = logging.getLogger(__name__)


DAY_SCALE = 86400
HOUR_SCALE = 3600
MINUTE_SCALE = 60
SECOND_SCALE = 1

OFFSET_PATTERN = re.compile(r'([+-]?)(d|day|days|h|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)')
DATETIME_SCALE = {
    'd': DAY_SCALE, 'day': DAY_SCALE, 'days': DAY_SCALE,
    'h': HOUR_SCALE, 'hour': HOUR_SCALE, 'hours': HOUR_SCALE,
    'm': MINUTE_SCALE, 'min': MINUTE_SCALE, 'mins': MINUTE_SCALE, 'minute': MINUTE_SCALE, 'minutes': MINUTE_SCALE,
    's': SECOND_SCALE, 'sec': SECOND_SCALE, 'secs': SECOND_SCALE, 'second': SECOND_SCALE, 'seconds': SECOND_SCALE,
}


def apply_datetime_offset(values: Index, offset: str) -> Index:
    """Apply offset to Unix epoch `values`."""
    if match := OFFSET_PATTERN.match(offset):
        sign, scale_name = match.groups()
        scale = DATETIME_SCALE[scale_name]
        if sign in ('', '+'):
            return (values - values[0]) / scale
        else:
            return (values - values[-1]) / scale
    else:
        raise ValueError(f'Unsupported offset \'{offset}\'')


class DataSet:
    """Relational dataset with rows and columns."""

    frame: DataFrame

    def __init__(self: DataSet, source: Union[DataFrame, DataSet]) -> None:
        """Direct initialization with existing `pandas.DataFrame`."""
        if isinstance(source, DataSet):
            self.frame = source.frame
        else:
            self.frame = DataFrame(source)

    @classmethod
    def from_text(cls: Type[DataSet], block: str, **options) -> DataSet:
        """Build by parsing raw text `block`."""
        return cls.from_io(StringIO(block), **options)

    @classmethod
    def from_io(cls: Type[DataSet], stream: IO, **options) -> DataSet:
        """Parse input data from existing I/O `stream`."""
        return cls(source=read_csv(filepath_or_buffer=stream, **options))  # noqa: pandas doesn't understand type?

    @classmethod
    def from_local(cls: Type[DataSet], filepath: str, encoding: str = 'utf-8', **options) -> DataSet:
        """Parse local file from `filepath`."""
        with open(filepath, mode='r', encoding=encoding) as stream:
            return cls.from_io(stream, **options)

    def __getitem__(self: DataSet, key: str) -> Series:
        """Select a column from the dataset."""
        series = self.frame[key]
        if series.dtype in ('object', ):
            raise RuntimeError(f'Unsupported dtype \'{series.dtype}\'')
        else:
            return series

    def set_index(self: DataSet, name: str = None, datetime_offset: str = None) -> None:
        """Set the index for the x-axis of the plot."""
        self.frame = self.frame.set_index(name)
        if self.index.dtype == 'datetime64[ns]':
            self.frame.index = self.frame.index.astype('int64') / 10**9
        if datetime_offset:
            self.frame.index = apply_datetime_offset(self.frame.index, offset=datetime_offset)

    @property
    def index(self: DataSet) -> Index:
        """The index for columns in the dataset (used for x-axis of plot)."""
        return self.frame.index

    @property
    def columns(self: DataSet) -> List[str]:
        """List of columns names."""
        return list(self.frame.columns)
