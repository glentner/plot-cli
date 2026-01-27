# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""DuckDB query building and data loading."""


# Type annotations
from __future__ import annotations
from typing import List, Optional, IO

# Standard libs
import sys
import logging
from io import StringIO
from pathlib import Path

# External libs
import duckdb
from pandas import DataFrame

# Public interface
__all__ = ['QueryBuilder', ]

# Module level logger
log = logging.getLogger(__name__)


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
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

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
        if self.source != '-':
            return
        log.info('Reading from <stdin>')
        data = sys.stdin.read()
        fmt = self.detect_format()
        if fmt == 'csv':
            self.conn.execute(f"CREATE TEMP TABLE __stdin_data AS SELECT * FROM read_csv_auto('/dev/stdin')")
        # TODO: Handle other formats for stdin (json, ndjson)

    def get_columns(self: QueryBuilder) -> List[str]:
        """Get available column names from the data source."""
        if self.source == '-':
            self._load_stdin()
            result = self.conn.execute("SELECT * FROM __stdin_data LIMIT 0")
        else:
            read_func = self._read_function()
            source_expr = self._build_source_expr()
            result = self.conn.execute(f"SELECT * FROM {read_func}({source_expr}) LIMIT 0")
        return [desc[0] for desc in result.description]

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
            # TODO: Implement time_bucket() query building
            x_select = f"time_bucket(INTERVAL '{self.bucket_interval}', {x_col}) AS {x_col}"
            y_selects = [f"{self.agg_method.upper()}({y}) AS {y}" for y in y_cols]
            select_clause = ', '.join([x_select] + y_selects)
            group_clause = f"GROUP BY 1 ORDER BY 1"
        else:
            # Simple select
            select_clause = ', '.join([x_col] + y_cols)
            group_clause = f"ORDER BY {x_col}"

        # Build WHERE clause
        where_parts = []
        if self.where_clause:
            where_parts.append(f"({self.where_clause})")
        if self.after_datetime:
            where_parts.append(f"{x_col} > '{self.after_datetime}'")
        if self.before_datetime:
            where_parts.append(f"{x_col} < '{self.before_datetime}'")

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
        result = self.conn.execute(query)
        return result.fetchdf()

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
