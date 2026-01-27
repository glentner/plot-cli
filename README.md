Simple Command-line Plotting Tool
=================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;
[![Version](https://img.shields.io/github/v/release/glentner/plot-cli?sort=semver)](https://github.com/glentner/plot-cli)
&nbsp;
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads)

---

A simple command-line plotting tool.

This project uses [DuckDB](https://duckdb.org/) for data loading and transformation,
and [tplot](https://pypi.org/project/tplot/) for terminal-based rendering. It supports
multiple input formats (CSV, JSON, NDJSON, Parquet), SQL-style filtering, and time-series
bucketing with aggregation.


Install
-------

This project should not be confused for an older, abandoned project already on the
package index by the same name. Install directly from GitHub:

```shell
uv tool install git+https://github.com/glentner/plot-cli@v0.3.2
```


Usage
-----

```
Usage:
  plot [-h] [-v] [FILE] [-x NAME] [-y NAME] [--line | --hist] ...
  Simple command-line plotting tool.
```

### Input Formats

By default, format is auto-detected from file extension. Use `--format` to override:

```shell
# CSV (default for stdin)
cat data.csv | plot -x timestamp -y value -T

# Parquet
plot data.parquet -x time -y metric

# JSON array
plot data.json --format json -x x -y y

# Newline-delimited JSON
plot logs.ndjson --format ndjson -x timestamp -y count
```

### Filtering

Filter data using SQL WHERE clauses or datetime bounds:

```shell
# Custom WHERE clause
plot data.csv --where "value > 100 AND status = 'active'"

# Datetime filtering (requires -x to specify timestamp column)
plot timeseries.csv -x timestamp -y value --after "2024-01-01" --before "2024-02-01"
```

### Time-Series Bucketing

Aggregate time-series data into buckets using `-B/--bucket` with an aggregation method:

```shell
# 15-minute buckets with mean aggregation
plot metrics.csv -x timestamp -y value -T -B 15min --mean

# Hourly buckets with sum
plot events.csv -x time -y count -T -B 1h --sum

# Supported aggregation methods: --mean, --sum, --count, --min, --max, --first, --last
```

Bucket interval formats: `15min`, `1h`, `30m`, `1d`, `60s`, or full syntax like `15 minutes`.

### Output Modes

Instead of plotting, output processed data as JSON or CSV:

```shell
# Output as JSON (useful for piping to jq)
plot data.csv --json --where "value > 50"

# Output as CSV
plot data.parquet --csv -x timestamp -y value -B 1h --mean
```

### Relative Time Scaling

Convert datetime axis to relative offset from start or end:

```shell
# Hours from start
plot timeseries.csv -x timestamp -y value -T -S +hours

# Minutes from end (negative offset)
plot timeseries.csv -x timestamp -y value -T -S -minutes
```


Options Reference
-----------------

| Option | Description |
|--------|-------------|
| `-x, --xdata NAME` | Column for x-axis |
| `-y, --ydata NAME...` | Column(s) for y-axis |
| `--format NAME` | Input format: `csv`, `json`, `ndjson`, `parquet` |
| `--line` | Line plot (default) |
| `--hist` | Histogram |
| `--where EXPR` | SQL WHERE clause for filtering |
| `--after TIME` | Filter rows after timestamp |
| `--before TIME` | Filter rows before timestamp |
| `-T, --timeseries` | Treat x-axis as datetime |
| `-S, --scale SCALE` | Relative offset (e.g., `+hours`, `-days`) |
| `-B, --bucket INTERVAL` | Time bucket interval (e.g., `15min`, `1h`) |
| `-A, --agg-method NAME` | Aggregation: `mean`, `sum`, `count`, `min`, `max`, `first`, `last` |
| `--mean`, `--sum`, etc. | Aggregation method aliases |
| `-b, --bins NUM` | Histogram bins (default: 10) |
| `-d, --density` | Show histogram as percentage |
| `-t, --title NAME` | Plot title |
| `-s, --size W,H` | Plot size in characters |
| `-c, --color SEQ` | Comma-separated colors |
| `-l, --legend POS` | Legend position |
| `-X, --xlabel NAME` | X-axis label |
| `-Y, --ylabel NAME` | Y-axis label |
| `--json` | Output data as JSON |
| `--csv` | Output data as CSV |
| `-h, --help` | Show help |
| `-v, --version` | Show version |


Example
-------

Using the basic line plot example from
[seaborn](https://seaborn.pydata.org/examples/wide_data_lineplot.html):

![plot-cli example](https://github.com/glentner/plot-cli/assets/8965948/fa5179c8-93b5-427e-a562-a26f6599de39)
