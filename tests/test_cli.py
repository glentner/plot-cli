# SPDX-FileCopyrightText: 2023 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CLI output modes and options."""


# Standard libs
import json

# External libs
from pytest import fixture, mark, CaptureFixture

# Internal libs
from plot_cli import PlotApp


@fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp,value,count\n"
        "2024-01-01 00:00:00,10.5,100\n"
        "2024-01-01 01:00:00,20.3,150\n"
        "2024-01-01 02:00:00,15.8,120\n"
    )
    return str(csv_path)


# =========================================================
# Output mode tests
# =========================================================


@mark.integration
class TestOutputModes:
    """Tests for --json and --csv output modes."""

    def test_json_output(self, capsys: CaptureFixture, sample_csv_file):
        """Output data as JSON."""
        PlotApp.main([sample_csv_file, "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 3
        assert data[0]["value"] == 10.5
        assert data[0]["count"] == 100

    def test_csv_output(self, capsys: CaptureFixture, sample_csv_file):
        """Output data as CSV."""
        PlotApp.main([sample_csv_file, "--csv"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "timestamp,value,count" in lines[0]

    def test_json_with_where_filter(self, capsys: CaptureFixture, sample_csv_file):
        """JSON output with WHERE filter applied."""
        PlotApp.main([sample_csv_file, "--json", "--where", "value > 15"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2  # 20.3 and 15.8
        for row in data:
            assert row["value"] > 15

    def test_csv_with_column_selection(self, capsys: CaptureFixture, sample_csv_file):
        """CSV output with specific column selection."""
        PlotApp.main([sample_csv_file, "--csv", "-x", "timestamp", "-y", "value"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "timestamp,value" in lines[0]
        assert "count" not in lines[0]


# =========================================================
# Filtering option tests
# =========================================================


@mark.integration
class TestFilteringOptions:
    """Tests for --where, --after, --before options."""

    def test_where_option(self, capsys: CaptureFixture, sample_csv_file):
        """Filter with --where option."""
        PlotApp.main([sample_csv_file, "--json", "--where", "count >= 120"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2  # 150 and 120

    def test_after_option(self, capsys: CaptureFixture, sample_csv_file):
        """Filter with --after option."""
        PlotApp.main([
            sample_csv_file, "--json",
            "-x", "timestamp",
            "--after", "2024-01-01 00:30:00"
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2  # 01:00 and 02:00

    def test_before_option(self, capsys: CaptureFixture, sample_csv_file):
        """Filter with --before option."""
        PlotApp.main([
            sample_csv_file, "--json",
            "-x", "timestamp",
            "--before", "2024-01-01 01:30:00"
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2  # 00:00 and 01:00


# =========================================================
# Aggregation option tests
# =========================================================


@mark.integration
class TestAggregationOptions:
    """Tests for -B/--bucket and aggregation method options."""

    @fixture
    def minute_data_csv(self, tmp_path):
        """CSV with per-minute data for aggregation tests."""
        csv_path = tmp_path / "minute_data.csv"
        rows = ["timestamp,value"]
        from datetime import datetime, timedelta
        base = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(30):
            ts = base + timedelta(minutes=i)
            rows.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{i}")
        csv_path.write_text("\n".join(rows))
        return str(csv_path)

    def test_bucket_with_mean(self, capsys: CaptureFixture, minute_data_csv):
        """Bucket with --mean aggregation."""
        PlotApp.main([
            minute_data_csv, "--json",
            "-x", "timestamp", "-y", "value",
            "-B", "15min", "--mean"
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2  # 30 minutes / 15 = 2 buckets

    def test_bucket_with_sum(self, capsys: CaptureFixture, minute_data_csv):
        """Bucket with --sum aggregation."""
        PlotApp.main([
            minute_data_csv, "--json",
            "-x", "timestamp", "-y", "value",
            "-B", "30min", "--sum"
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1

    def test_bucket_shorthand_formats(self, capsys: CaptureFixture, minute_data_csv):
        """Bucket interval shorthand formats."""
        # Test '15m' shorthand
        PlotApp.main([
            minute_data_csv, "--json",
            "-x", "timestamp", "-y", "value",
            "-B", "15m", "--count"
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2
        # Each bucket should have 15 rows
        assert all(row["value"] == 15 for row in data)


# =========================================================
# Deprecated option tests
# =========================================================


@mark.integration
class TestDeprecatedOptions:
    """Tests for deprecated -F/--resample option."""

    @fixture
    def minute_data_csv(self, tmp_path):
        """CSV with per-minute data."""
        csv_path = tmp_path / "minute_data.csv"
        rows = ["timestamp,value"]
        from datetime import datetime, timedelta
        base = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(30):
            ts = base + timedelta(minutes=i)
            rows.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{i}")
        csv_path.write_text("\n".join(rows))
        return str(csv_path)

    def test_resample_deprecation_works(self, capsys: CaptureFixture, caplog, minute_data_csv):
        """Deprecated -F/--resample still works but issues warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            PlotApp.main([
                minute_data_csv, "--json",
                "-x", "timestamp", "-y", "value",
                "-F", "15min", "--mean"
            ])
        captured = capsys.readouterr()
        # Should still produce output
        data = json.loads(captured.out)
        assert len(data) == 2
        # Warning logged via cmdkit logger
        assert any("deprecated" in record.message.lower() for record in caplog.records)


# =========================================================
# Format option tests
# =========================================================


@mark.integration
class TestFormatOption:
    """Tests for --format option."""

    @fixture
    def ndjson_file(self, tmp_path):
        """Create NDJSON file."""
        path = tmp_path / "data.ndjson"
        lines = [
            '{"x": 1, "y": 10}',
            '{"x": 2, "y": 20}',
            '{"x": 3, "y": 30}',
        ]
        path.write_text("\n".join(lines))
        return str(path)

    def test_ndjson_format(self, capsys: CaptureFixture, ndjson_file):
        """Read NDJSON format."""
        PlotApp.main([ndjson_file, "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 3
        assert data[0]["x"] == 1
