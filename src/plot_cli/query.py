# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""DuckDB query building and data loading."""


# Type annotations
from __future__ import annotations
from typing import List, Optional, Dict, Any

# Standard libs
import re
import sys
import logging
from pathlib import Path

# External libs
import duckdb
from pandas import DataFrame

# Public interface
__all__ = ['QueryBuilder', 'apply_datetime_scale', ]

# Module level logger
log = logging.getLogger(__name__)


# Scale factors for datetime offset conversion
DAY_SCALE = 86400
HOUR_SCALE = 3600
MINUTE_SCALE = 60
SECOND_SCALE = 1

OFFSET_PATTERN = re.compile(
    r'([+-]?)(d|day|days|h|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)'
)
DATETIME_SCALE: Dict[str, int] = {
    'd': DAY_SCALE, 'day': DAY_SCALE, 'days': DAY_SCALE,
    'h': HOUR_SCALE, 'hour': HOUR_SCALE, 'hours': HOUR_SCALE,
    'm': MINUTE_SCALE, 'min': MINUTE_SCALE, 'mins': MINUTE_SCALE,
    'minute': MINUTE_SCALE, 'minutes': MINUTE_SCALE,
    's': SECOND_SCALE, 'sec': SECOND_SCALE, 'secs': SECOND_SCALE,
    'second': SECOND_SCALE, 'seconds': SECOND_SCALE,
}


def apply_datetime_scale(df: DataFrame, column: str, scale: str) -> DataFrame:
    """
    Apply datetime scale offset to convert timestamps to relative values.

    Args:
        df: DataFrame with datetime column
        column: Name of the datetime column
        scale: Offset specification (e.g., '+hours', '-days', 'minutes')

    Returns:
        DataFrame with column converted to numeric offset values.
    """
    if match := OFFSET_PATTERN.match(scale):
        sign, scale_name = match.groups()
        divisor = DATETIME_SCALE[scale_name]

        # Convert datetime to epoch seconds
        # DuckDB returns datetime64[us] (microseconds), pandas uses int64 representation
        df = df.copy()
        dtype_str = str(df[column].dtype)
        if 'datetime64[us]' in dtype_str:
            # Microsecond precision (DuckDB default)
            epoch_values = df[column].astype('int64') / 10**6
        elif 'datetime64[ns]' in dtype_str:
            # Nanosecond precision (pandas default)
            epoch_values = df[column].astype('int64') / 10**9
        elif 'datetime64[ms]' in dtype_str:
            # Millisecond precision
            epoch_values = df[column].astype('int64') / 10**3
        else:
            # Try to convert via timestamp
            epoch_values = df[column].apply(lambda x: x.timestamp() if hasattr(x, 'timestamp') else float(x))

        # Apply offset from start or end
        if sign in ('', '+'):
            df[column] = (epoch_values - epoch_values.iloc[0]) / divisor
        else:
            df[column] = (epoch_values - epoch_values.iloc[-1]) / divisor

        return df
    else:
        raise ValueError(f"Unsupported scale offset: '{scale}'")


# Bucket interval patterns for parsing shorthand like '15min', '1h', '1d'
BUCKET_PATTERN = re.compile(r'^(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hour|hours|d|day|days)$')
BUCKET_UNITS: Dict[str, str] = {
    's': 'seconds', 'sec': 'seconds', 'secs': 'seconds', 'second': 'seconds', 'seconds': 'seconds',
    'm': 'minutes', 'min': 'minutes', 'mins': 'minutes', 'minute': 'minutes', 'minutes': 'minutes',
    'h': 'hours', 'hour': 'hours', 'hours': 'hours',
    'd': 'days', 'day': 'days', 'days': 'days',
}


def parse_bucket_interval(interval: str) -> str:
    """
    Parse bucket interval shorthand into DuckDB INTERVAL syntax.

    Args:
        interval: Shorthand like '15min', '1h', '1d' or full syntax like '15 minutes'

    Returns:
        DuckDB-compatible interval string (e.g., '15 minutes')
    """
    # Already in full format?
    if ' ' in interval:
        return interval

    if match := BUCKET_PATTERN.match(interval.lower()):
        value, unit = match.groups()
        return f"{value} {BUCKET_UNITS[unit]}"

    # Return as-is and let DuckDB handle it
    return interval


class QueryBuilder:
    """
    Build and execute DuckDB queries for data loading and transformation.

    Supports CSV, Parquet, JSON, and NDJSON formats with automatic detection
    based on file extension. Provides SQL-based filtering and time-series
    bucketing via DuckDB's time_bucket() function.
    """

    source: str
    format: Optional[str]
    x_column: Optional[str]
    y_columns: List[str]
    where_clause: Optional[str]
    after_datetime: Optional[str]
    before_datetime: Optional[str]
    bucket_interval: Optional[str]
    agg_method: Optional[str]
    timeseries: bool
    scale: Optional[str]
    _cached_columns: Optional[List[str]]

    def __init__(
        self: QueryBuilder,
        source: str,
        format: Optional[str] = None,
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        after_datetime: Optional[str] = None,
        before_datetime: Optional[str] = None,
        bucket_interval: Optional[str] = None,
        agg_method: Optional[str] = None,
        timeseries: bool = False,
        scale: Optional[str] = None,
    ) -> None:
        """Initialize query builder with data source and options."""
        self.source = source
        self.format = format
        self.x_column = x_column
        self.y_columns = y_columns or []
        self.where_clause = where_clause
        self.after_datetime = after_datetime
        self.before_datetime = before_datetime
        self.bucket_interval = bucket_interval
        self.agg_method = agg_method
        self.timeseries = timeseries
        self.scale = scale
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._cached_columns: Optional[List[str]] = None
        self._stdin_loaded: bool = False

    @property
    def conn(self: QueryBuilder) -> duckdb.DuckDBPyConnection:
        """Lazy-initialize DuckDB connection."""
        if self._conn is None:
            self._conn = duckdb.connect()
        return self._conn

    def detect_format(self: QueryBuilder) -> str:
        """Detect file format from extension or explicit format."""
        if self.format:
            return self.format
        if self.source == '-':
            return 'csv'  # Default stdin to CSV
        path = Path(self.source)
        ext = path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.json': 'json',
            '.ndjson': 'ndjson',
            '.jsonl': 'ndjson',
        }
        return format_map.get(ext, 'csv')

    def _read_function(self: QueryBuilder) -> str:
        """Return DuckDB read function for detected format."""
        fmt = self.detect_format()
        read_funcs = {
            'csv': 'read_csv_auto',
            'parquet': 'read_parquet',
            'json': 'read_json_auto',
            'ndjson': 'read_json_auto',
        }
        return read_funcs.get(fmt, 'read_csv_auto')

    def _build_source_expr(self: QueryBuilder, stdin_data: Optional[str] = None) -> str:
        """Build the FROM clause source expression."""
        if self.source == '-':
            # For stdin, we'll register the data as a temp table
            return '__stdin_data'
        return f"'{self.source}'"

    def _load_stdin(self: QueryBuilder) -> None:
        """Load stdin data into a temporary table."""
        if self.source != '-' or self._stdin_loaded:
            return
        log.info('Reading from <stdin>')
        fmt = self.detect_format()
        if fmt == 'csv':
            self.conn.execute("CREATE TEMP TABLE __stdin_data AS SELECT * FROM read_csv_auto('/dev/stdin')")
        elif fmt in ('json', 'ndjson'):
            self.conn.execute("CREATE TEMP TABLE __stdin_data AS SELECT * FROM read_json_auto('/dev/stdin')")
        else:
            raise ValueError(f"Unsupported stdin format: {fmt}")
        self._stdin_loaded = True

    def get_columns(self: QueryBuilder) -> List[str]:
        """Get available column names from the data source."""
        if self._cached_columns is not None:
            return self._cached_columns

        if self.source == '-':
            self._load_stdin()
            result = self.conn.execute("SELECT * FROM __stdin_data LIMIT 0")
        else:
            read_func = self._read_function()
            source_expr = self._build_source_expr()
            result = self.conn.execute(f"SELECT * FROM {read_func}({source_expr}) LIMIT 0")

        self._cached_columns = [desc[0] for desc in result.description]
        return self._cached_columns

    def build_query(self: QueryBuilder) -> str:
        """
        Build the SQL query based on configured options.

        Returns the SQL string to execute.
        """
        read_func = self._read_function()
        source_expr = self._build_source_expr()

        # Determine columns to select
        columns = self.get_columns()
        x_col = self.x_column or columns[0]

        if self.y_columns:
            y_cols = self.y_columns
        else:
            # All numeric columns except x
            # TODO: Filter to numeric columns only
            y_cols = [c for c in columns if c != x_col]

        # Build SELECT clause
        if self.bucket_interval and self.agg_method:
            # Time bucketing with aggregation
            interval_str = parse_bucket_interval(self.bucket_interval)
            x_select = f"time_bucket(INTERVAL '{interval_str}', \"{x_col}\") AS \"{x_col}\""
            y_selects = [f"{self.agg_method.upper()}(\"{y}\") AS \"{y}\"" for y in y_cols]
            select_clause = ', '.join([x_select] + y_selects)
            group_clause = "GROUP BY 1 ORDER BY 1"
        else:
            # Simple select - quote column names for safety
            select_clause = ', '.join([f'"{x_col}"'] + [f'"{y}"' for y in y_cols])
            group_clause = f'ORDER BY "{x_col}"'

        # Build WHERE clause
        where_parts = []
        if self.where_clause:
            where_parts.append(f"({self.where_clause})")
        if self.after_datetime:
            where_parts.append(f'"{x_col}" > \'{self.after_datetime}\'')
        if self.before_datetime:
            where_parts.append(f'"{x_col}" < \'{self.before_datetime}\'')

        where_clause = ''
        if where_parts:
            where_clause = 'WHERE ' + ' AND '.join(where_parts)

        # Build full query
        if self.source == '-':
            from_clause = '__stdin_data'
        else:
            from_clause = f"{read_func}({source_expr})"

        query = f"SELECT {select_clause} FROM {from_clause} {where_clause} {group_clause}"
        log.debug(f'Query: {query}')
        return query

    def execute(self: QueryBuilder) -> DataFrame:
        """Execute the query and return results as pandas DataFrame."""
        if self.source == '-':
            self._load_stdin()
        query = self.build_query()
        log.debug(f'Executing: {query}')
        result = self.conn.execute(query)
        df = result.fetchdf()

        # Apply scale offset if specified
        if self.scale:
            x_col = self.x_column or df.columns[0]
            df = apply_datetime_scale(df, x_col, self.scale)

        return df

    @classmethod
    def from_file(
        cls,
        filepath: str,
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> QueryBuilder:
        """Create QueryBuilder from a file path."""
        return cls(
            source=filepath,
            x_column=x_column,
            y_columns=y_columns,
            **kwargs,
        )

    @classmethod
    def from_stdin(
        cls,
        format: str = 'csv',
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> QueryBuilder:
        """Create QueryBuilder for stdin input."""
        return cls(
            source='-',
            format=format,
            x_column=x_column,
            y_columns=y_columns,
            **kwargs,
        )
