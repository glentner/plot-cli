# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QueryBuilder and data processing."""


# Standard libs
import os
import json
import tempfile
from datetime import datetime, timedelta

# External libs
from pytest import fixture, mark, raises
import pandas as pd

# Internal libs
from plot_cli.query import (
    QueryBuilder,
    apply_datetime_scale,
    parse_bucket_interval,
)


@fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp,value,count\n"
        "2024-01-01 00:00:00,10.5,100\n"
        "2024-01-01 01:00:00,20.3,150\n"
        "2024-01-01 02:00:00,15.8,120\n"
        "2024-01-01 03:00:00,25.1,200\n"
        "2024-01-01 04:00:00,18.7,180\n"
    )
    return str(csv_path)


@fixture
def sample_json_file(tmp_path):
    """Create a temporary JSON file with sample data."""
    json_path = tmp_path / "sample.json"
    data = [
        {"timestamp": "2024-01-01 00:00:00", "value": 10.5, "count": 100},
        {"timestamp": "2024-01-01 01:00:00", "value": 20.3, "count": 150},
        {"timestamp": "2024-01-01 02:00:00", "value": 15.8, "count": 120},
    ]
    json_path.write_text(json.dumps(data))
    return str(json_path)


@fixture
def sample_ndjson_file(tmp_path):
    """Create a temporary NDJSON file with sample data."""
    ndjson_path = tmp_path / "sample.ndjson"
    lines = [
        '{"timestamp": "2024-01-01 00:00:00", "value": 10.5, "count": 100}',
        '{"timestamp": "2024-01-01 01:00:00", "value": 20.3, "count": 150}',
        '{"timestamp": "2024-01-01 02:00:00", "value": 15.8, "count": 120}',
    ]
    ndjson_path.write_text("\n".join(lines))
    return str(ndjson_path)


@fixture
def sample_parquet_file(tmp_path):
    """Create a temporary Parquet file with sample data."""
    parquet_path = tmp_path / "sample.parquet"
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ]),
        "value": [10.5, 20.3, 15.8],
        "count": [100, 150, 120],
    })
    df.to_parquet(parquet_path)
    return str(parquet_path)


# =========================================================
# Format detection tests
# =========================================================


@mark.unit
class TestFormatDetection:
    """Tests for automatic format detection."""

    def test_csv_extension(self, sample_csv_file):
        """CSV format detected from .csv extension."""
        qb = QueryBuilder(source=sample_csv_file)
        assert qb.detect_format() == "csv"

    def test_json_extension(self, sample_json_file):
        """JSON format detected from .json extension."""
        qb = QueryBuilder(source=sample_json_file)
        assert qb.detect_format() == "json"

    def test_ndjson_extension(self, sample_ndjson_file):
        """NDJSON format detected from .ndjson extension."""
        qb = QueryBuilder(source=sample_ndjson_file)
        assert qb.detect_format() == "ndjson"

    def test_explicit_format_override(self, sample_csv_file):
        """Explicit format overrides extension detection."""
        qb = QueryBuilder(source=sample_csv_file, format="json")
        assert qb.detect_format() == "json"

    def test_stdin_defaults_to_csv(self):
        """Stdin defaults to CSV format."""
        qb = QueryBuilder(source="-")
        assert qb.detect_format() == "csv"


# =========================================================
# QueryBuilder execution tests
# =========================================================


@mark.unit
class TestQueryBuilderExecution:
    """Tests for QueryBuilder query execution."""

    def test_load_csv(self, sample_csv_file):
        """Load data from CSV file."""
        qb = QueryBuilder(source=sample_csv_file)
        df = qb.execute()
        assert len(df) == 5
        assert list(df.columns) == ["timestamp", "value", "count"]

    def test_load_json(self, sample_json_file):
        """Load data from JSON file."""
        qb = QueryBuilder(source=sample_json_file)
        df = qb.execute()
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "value" in df.columns

    def test_load_ndjson(self, sample_ndjson_file):
        """Load data from NDJSON file."""
        qb = QueryBuilder(source=sample_ndjson_file)
        df = qb.execute()
        assert len(df) == 3

    def test_load_parquet(self, sample_parquet_file):
        """Load data from Parquet file."""
        qb = QueryBuilder(source=sample_parquet_file)
        df = qb.execute()
        assert len(df) == 3

    def test_select_columns(self, sample_csv_file):
        """Select specific x and y columns."""
        qb = QueryBuilder(
            source=sample_csv_file,
            x_column="timestamp",
            y_columns=["value"],
        )
        df = qb.execute()
        assert list(df.columns) == ["timestamp", "value"]

    def test_get_columns(self, sample_csv_file):
        """Get available column names."""
        qb = QueryBuilder(source=sample_csv_file)
        columns = qb.get_columns()
        assert columns == ["timestamp", "value", "count"]


# =========================================================
# Filtering tests
# =========================================================


@mark.unit
class TestFiltering:
    """Tests for SQL WHERE clause filtering."""

    def test_where_clause(self, sample_csv_file):
        """Filter with custom WHERE clause."""
        qb = QueryBuilder(
            source=sample_csv_file,
            where_clause="value > 20",
        )
        df = qb.execute()
        assert len(df) == 2  # 20.3 and 25.1
        assert all(df["value"] > 20)

    def test_after_datetime(self, sample_csv_file):
        """Filter rows after timestamp."""
        qb = QueryBuilder(
            source=sample_csv_file,
            x_column="timestamp",
            after_datetime="2024-01-01 02:00:00",
        )
        df = qb.execute()
        assert len(df) == 2  # 03:00 and 04:00

    def test_before_datetime(self, sample_csv_file):
        """Filter rows before timestamp."""
        qb = QueryBuilder(
            source=sample_csv_file,
            x_column="timestamp",
            before_datetime="2024-01-01 02:00:00",
        )
        df = qb.execute()
        assert len(df) == 2  # 00:00 and 01:00

    def test_combined_after_before(self, sample_csv_file):
        """Filter with both after and before."""
        qb = QueryBuilder(
            source=sample_csv_file,
            x_column="timestamp",
            after_datetime="2024-01-01 00:30:00",
            before_datetime="2024-01-01 02:30:00",
        )
        df = qb.execute()
        assert len(df) == 2  # 01:00 and 02:00


# =========================================================
# Bucket interval parsing tests
# =========================================================


@mark.unit
class TestBucketIntervalParsing:
    """Tests for bucket interval parsing."""

    @mark.parametrize("shorthand,expected", [
        ("15min", "15 minutes"),
        ("1h", "1 hours"),
        ("30m", "30 minutes"),
        ("1d", "1 days"),
        ("60s", "60 seconds"),
        ("5sec", "5 seconds"),
        ("2hour", "2 hours"),
        ("3day", "3 days"),
    ])
    def test_shorthand_parsing(self, shorthand, expected):
        """Parse shorthand bucket intervals."""
        assert parse_bucket_interval(shorthand) == expected

    def test_full_syntax_passthrough(self):
        """Full syntax passes through unchanged."""
        assert parse_bucket_interval("15 minutes") == "15 minutes"


# =========================================================
# Aggregation tests
# =========================================================


@mark.unit
class TestAggregation:
    """Tests for time bucketing and aggregation."""

    @fixture
    def minute_data_csv(self, tmp_path):
        """CSV with per-minute data for aggregation tests."""
        csv_path = tmp_path / "minute_data.csv"
        rows = ["timestamp,value"]
        base = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(60):
            ts = base + timedelta(minutes=i)
            rows.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{i * 1.5}")
        csv_path.write_text("\n".join(rows))
        return str(csv_path)

    def test_bucket_with_mean(self, minute_data_csv):
        """Bucket by 15 minutes with mean aggregation."""
        qb = QueryBuilder(
            source=minute_data_csv,
            x_column="timestamp",
            y_columns=["value"],
            bucket_interval="15min",
            agg_method="mean",
            timeseries=True,
        )
        df = qb.execute()
        assert len(df) == 4  # 60 minutes / 15 = 4 buckets

    def test_bucket_with_sum(self, minute_data_csv):
        """Bucket by 30 minutes with sum aggregation."""
        qb = QueryBuilder(
            source=minute_data_csv,
            x_column="timestamp",
            y_columns=["value"],
            bucket_interval="30m",
            agg_method="sum",
            timeseries=True,
        )
        df = qb.execute()
        assert len(df) == 2  # 60 minutes / 30 = 2 buckets

    def test_bucket_with_count(self, minute_data_csv):
        """Bucket by 15 minutes with count aggregation."""
        qb = QueryBuilder(
            source=minute_data_csv,
            x_column="timestamp",
            y_columns=["value"],
            bucket_interval="15min",
            agg_method="count",
            timeseries=True,
        )
        df = qb.execute()
        assert len(df) == 4
        assert all(df["value"] == 15)  # 15 minutes per bucket

    def test_bucket_with_max(self, minute_data_csv):
        """Bucket with max aggregation."""
        qb = QueryBuilder(
            source=minute_data_csv,
            x_column="timestamp",
            y_columns=["value"],
            bucket_interval="1h",
            agg_method="max",
            timeseries=True,
        )
        df = qb.execute()
        assert len(df) == 1
        assert df["value"].iloc[0] == 59 * 1.5  # max value

    def test_bucket_with_min(self, minute_data_csv):
        """Bucket with min aggregation."""
        qb = QueryBuilder(
            source=minute_data_csv,
            x_column="timestamp",
            y_columns=["value"],
            bucket_interval="1h",
            agg_method="min",
            timeseries=True,
        )
        df = qb.execute()
        assert len(df) == 1
        assert df["value"].iloc[0] == 0.0  # min value


# =========================================================
# Datetime scale tests
# =========================================================


@mark.unit
class TestDatetimeScale:
    """Tests for datetime scale offset conversion."""

    @fixture
    def datetime_df(self):
        """DataFrame with datetime column."""
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 02:00:00",
                "2024-01-01 03:00:00",
            ]),
            "value": [10, 20, 30, 40],
        })

    def test_scale_positive_hours(self, datetime_df):
        """Scale to positive hours offset."""
        result = apply_datetime_scale(datetime_df, "timestamp", "+hours")
        assert result["timestamp"].iloc[0] == 0.0
        assert result["timestamp"].iloc[1] == 1.0
        assert result["timestamp"].iloc[-1] == 3.0

    def test_scale_negative_hours(self, datetime_df):
        """Scale to negative hours offset (from end)."""
        result = apply_datetime_scale(datetime_df, "timestamp", "-hours")
        assert result["timestamp"].iloc[-1] == 0.0
        assert result["timestamp"].iloc[0] == -3.0

    def test_scale_minutes(self, datetime_df):
        """Scale to minutes offset."""
        result = apply_datetime_scale(datetime_df, "timestamp", "+minutes")
        assert result["timestamp"].iloc[0] == 0.0
        assert result["timestamp"].iloc[1] == 60.0

    def test_scale_days(self, datetime_df):
        """Scale to days offset."""
        result = apply_datetime_scale(datetime_df, "timestamp", "+days")
        assert result["timestamp"].iloc[0] == 0.0
        # 3 hours = 3/24 = 0.125 days
        assert abs(result["timestamp"].iloc[-1] - 0.125) < 0.001

    def test_invalid_scale(self, datetime_df):
        """Invalid scale raises ValueError."""
        with raises(ValueError, match="Unsupported scale offset"):
            apply_datetime_scale(datetime_df, "timestamp", "invalid")
