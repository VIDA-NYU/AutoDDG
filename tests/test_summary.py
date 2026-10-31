"""Statistical profiler (autoddg.profiling.summary)

Each test names the upstream output it replaces (AutoDDG 0.1.2).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from autoddg.profiling import profile_dataset, profile_metadata, summarize


def column_line(text: str) -> str:
    """The first column's line: after the opening sentence and the rows line"""
    return text.splitlines()[2]


def test_range_does_not_start_at_zero():
    # was "Coverage spans from 0 to 51.69"
    line = column_line(profile_dataset(pd.DataFrame({"lat": np.linspace(40.50, 51.87, 500)}))[0])
    assert "Minimum 40.5, maximum 51.87." in line
    assert " 0 to" not in line


def test_negative_values_do_not_end_at_zero():
    # was "Coverage spans from -49.25 to 0"
    line = column_line(profile_dataset(pd.DataFrame({"v": np.linspace(-50, -3, 500)}))[0])
    assert "Minimum -50, maximum -3." in line


def test_outlier_is_kept_in_the_extremes():
    # was "Coverage spans from 0 to 98.0": datamart's clusters drop the 10000
    line = column_line(profile_dataset(pd.DataFrame({"v": list(range(1, 101)) + [10000]}))[0])
    assert "Minimum 1, maximum 10000." in line
    assert "Main value ranges:" in line


def test_dates_are_utc_and_span_the_whole_column():
    # was "covering from 2000-01-06 to 2000-12-24" (clusters, shifted by the local time zone)
    daily = pd.date_range("2000-01-01", "2000-12-31", freq="D").strftime("%Y-%m-%d")
    assert "from 2000-01-01 to 2000-12-31" in profile_dataset(pd.DataFrame({"d": daily}))[1]
    yearly = pd.date_range("1990-01-01", "2020-01-01", freq="YS").strftime("%Y-%m-%d")
    assert "from 1990-01-01 to 2020-01-01" in profile_dataset(pd.DataFrame({"d": yearly}))[1]


def test_computed_fields_are_reported():
    # missing-value ratio, mean, standard deviation and semantic types were computed but dropped
    line = column_line(profile_dataset(pd.DataFrame({"v": [5.0] * 70 + [None] * 30}))[0])
    assert "Missing values: 30.0%." in line
    assert "Mean 5, standard deviation 0." in line
    dates = pd.DataFrame({"d": [f"2020-01-0{i}" for i in range(1, 10)]})
    assert "Semantic types: DateTime." in column_line(profile_dataset(dates)[0])


def test_integer_column_names():
    # was AttributeError: 'int' object has no attribute 'lower' (a headerless sheet's columns)
    frame = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]])
    assert "Minimum 1, maximum 3." in profile_dataset(frame)[0]


def test_large_frame_is_sampled_with_datamart_rule():
    # datamart sampled files but profiled every row of a frame
    metadata = profile_metadata(pd.DataFrame({"v": np.arange(1_000_000, dtype=float)}))
    assert metadata["nb_rows"] == 1_000_000
    assert metadata["sampled"] and metadata["nb_profiled_rows"] < 1_000_000
    assert "profiled a random sample of" in summarize(metadata)[0]
    small = profile_metadata(pd.DataFrame({"v": [1, 2, 3]}))
    assert small["nb_rows"] == 3 and not small["sampled"]
    assert "Rows: 3 (all rows profiled). Columns: 1." in summarize(small)[0]


def test_caller_sampled_rows_are_marked():
    metadata = profile_metadata(pd.DataFrame({"v": [1, 2, 3]}), nb_rows=300)
    assert metadata["nb_rows"] == 300 and metadata["sampled"]


def test_index_turned_into_columns_as_datamart_does():
    # a header shorter than its rows: pandas reads the extra leading fields as an index, which
    # datamart turns into columns; the extremes must follow the same columns
    rows = ["a,b"] + [f"{i},{i % 7},{i * 2},{i * 3}" for i in range(2000)]
    frame = pd.read_csv(io.StringIO("\n".join(rows)))
    metadata = profile_metadata(frame)
    assert len(metadata["columns"]) == 4
    assert metadata["columns"][-1]["maximum"] == 1999 * 3
    big = pd.concat([frame] * 1000)  # over 5 MB, so sampled; its index is not a row number
    sampled = profile_metadata(big)
    assert sampled["sampled"] and len(sampled["columns"]) == 4
