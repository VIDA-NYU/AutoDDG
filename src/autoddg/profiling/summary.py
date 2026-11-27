from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy
import pandas
from atlas_profiler import process_dataset
from beartype import beartype
from pandas import DataFrame
from profiler.core import RANDOM_SEED
from profiler.profile_types import parse_dates
from profiler.temporal import parse_date

NUMERIC_TYPES = ("Integer", "Float")
DATE_TIME = "http://schema.org/DateTime"
NUMBER_BOUND = 3.4e38  # atlas-profiler ignores larger numbers
DATE_CHECK_VALUES = 200  # bulk-parsed dates checked against atlas-profiler's own parser
YEARS = (1000, 2999)  # a date column of bare 4-digit numbers in this range holds years


def _plain_index(frame: DataFrame) -> bool:
    """True for a 0..n-1 integer index, the only index atlas-profiler does not turn into columns"""
    return frame.index.dtype == numpy.int64 and pandas.Index(numpy.arange(len(frame))).equals(
        frame.index
    )


def _text(values: pandas.Series) -> pandas.Series:
    """Values as stripped strings, missing values as empty strings (as atlas-profiler reads them)"""
    return values.astype(object).fillna("").astype(str).str.strip()


def _date_extent(text: pandas.Series, sample_text: Any) -> tuple[float, float] | None:
    """
    Earliest and latest timestamps of a date column

    Every value is parsed in bulk when that reads them all and agrees with atlas-profiler's
    parser on the first values; otherwise the profiled rows are parsed with that parser.

    Args:
        text: Non-empty values of every row
        sample_text: Callable returning the non-empty values of the profiled rows

    Returns:
        (earliest, latest) as UTC POSIX timestamps, or None when no value parses
    """
    # Bare years ("2001"): atlas-profiler's parser reads none of them, so neither check below would
    years = pandas.to_numeric(text, errors="coerce")
    if len(text) and text.str.fullmatch(r"\d{4}").all() and years.between(*YEARS).all():
        first, last = int(years.min()), int(years.max())
        return (
            datetime(first, 1, 1, tzinfo=timezone.utc).timestamp(),
            datetime(last, 1, 1, tzinfo=timezone.utc).timestamp(),
        )
    if len(text):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "could not infer format"
            parsed = pandas.to_datetime(text, errors="coerce", utc=True)
        head = zip(text.iloc[:DATE_CHECK_VALUES], parsed.iloc[:DATE_CHECK_VALUES], strict=True)
        if not parsed.isna().any() and all(
            parse_date(raw) == bulk.to_pydatetime() for raw, bulk in head
        ):
            return parsed.min().timestamp(), parsed.max().timestamp()
    dates = parse_dates(sample_text())
    if not dates:
        return None
    return min(dates).timestamp(), max(dates).timestamp()


@beartype
def profile_dataset(data_frame: DataFrame, *, geo_classifier: bool = False) -> tuple[str, str]:
    """
    Run atlas-profiler and produce short textual summaries

    Ref: https://github.com/VIDA-NYU/atlas-profiler

    Args:
        data_frame: Input frame
        geo_classifier: Run atlas-profiler's spatial column classifier (loads PyTorch and
            downloads its model on first use)

    Returns:
        (profile_text, semantic_notes)
    """

    return summarize(profile_metadata(data_frame, geo_classifier=geo_classifier))


@beartype
def profile_metadata(
    data_frame: DataFrame, nb_rows: int | None = None, *, geo_classifier: bool = False
) -> dict[str, Any]:
    """
    Run atlas-profiler, then set each column's missing-value ratio and add its true extremes

    atlas-profiler detects types on a random sample of a large frame (about two million values,
    at least 10,000 rows). The missing-value ratios and extremes set here cover every row.

    Args:
        data_frame: Input frame
        nb_rows: Full row count, when the caller has already sampled the rows (for example while
            reading a large file); the profile is then marked as sampled
        geo_classifier: Run atlas-profiler's spatial column classifier

    Returns:
        atlas-profiler metadata, plus 'nb_rows', 'sampled' and, per column,
        'missing_values_ratio' and 'minimum' and 'maximum' (numeric) or 'earliest' and 'latest'
        (dates, as UTC timestamps)
    """

    total = len(data_frame) if nb_rows is None else nb_rows
    metadata = process_dataset(data_frame, geo_classifier=geo_classifier, coverage=True)
    metadata["nb_rows"] = total
    metadata["sampled"] = metadata.get("nb_profiled_rows", 0) < total

    # atlas-profiler's coverage keeps the 5th-95th percentiles of up to three k-means clusters
    # (for a frame, also for dates), so it is not the column's range: add the true extremes,
    # parsing values the way atlas-profiler does. Columns line up with the frame's, plus its
    # index when atlas-profiler turned that into columns.
    columns = metadata.get("columns", [])
    frame = data_frame if _plain_index(data_frame) else data_frame.reset_index()
    profiled = metadata.get("nb_profiled_rows", len(frame))

    def sample_text(i: int) -> list[str]:
        rows = frame if profiled >= len(frame) else frame.sample(profiled, random_state=RANDOM_SEED)
        text = _text(rows.iloc[:, i])
        return text[text != ""].tolist()

    for i, column_meta in enumerate(columns):
        text = _text(frame.iloc[:, i])
        values = text[text != ""]
        if len(text):  # atlas-profiler's ratio comes from the sample, and only when above 0
            column_meta["missing_values_ratio"] = 1 - len(values) / len(text)
        if column_meta.get("structural_type", "").split("/")[-1] in NUMERIC_TYPES:
            numbers = pandas.to_numeric(values, errors="coerce").dropna()
            numbers = numbers[numbers.abs() < NUMBER_BOUND]
            if len(numbers):
                column_meta["minimum"] = float(numbers.min())
                column_meta["maximum"] = float(numbers.max())
        if DATE_TIME in column_meta.get("semantic_types", []):
            extent = _date_extent(values, lambda i=i: sample_text(i))
            if extent is not None:
                column_meta["earliest"], column_meta["latest"] = extent
    for temp_cov in metadata.get("temporal_coverage", []):
        index = temp_cov.get("column_indexes", [None])[0]
        if index is not None and "earliest" in columns[index]:
            temp_cov["earliest"] = columns[index]["earliest"]
            temp_cov["latest"] = columns[index]["latest"]
    return metadata


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _date(timestamp: float) -> str:
    # atlas-profiler's timestamps are UTC; formatting them in the local time zone shifts the dates
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


@beartype
def summarize(metadata: dict[str, Any]) -> tuple[str, str]:
    """
    Produce short textual summaries from profile_metadata's output

    Args:
        metadata: Output of profile_metadata

    Returns:
        (profile_text, semantic_notes)
    """

    profile_summary: list[str] = []

    for column_meta in metadata.get("columns", []):
        column_summary = f"**{column_meta['name']}**: "

        structural_type = column_meta.get("structural_type", "Unknown")
        type_name = structural_type.split("/")[-1].lower()
        column_summary += f"Data is of type {type_name}. "

        semantic_types = [t.split("/")[-1] for t in column_meta.get("semantic_types", [])]
        if semantic_types:
            column_summary += f"Semantic types: {', '.join(semantic_types)}. "

        if "num_distinct_values" in column_meta:
            num_distinct_values = column_meta["num_distinct_values"]
            column_summary += f"There are {num_distinct_values} unique values. "

        if "missing_values_ratio" in column_meta:
            column_summary += f"Missing values: {column_meta['missing_values_ratio']:.1%}. "
        if column_meta.get("unclean_values_ratio"):
            column_summary += (
                f"Values that do not parse as {type_name}: "
                f"{column_meta['unclean_values_ratio']:.1%}. "
            )
        if "mean" in column_meta and "stddev" in column_meta:
            column_summary += (
                f"Mean {column_meta['mean']:g}, standard deviation {column_meta['stddev']:g}. "
            )

        if "minimum" in column_meta:
            column_summary += (
                f"Minimum {_number(column_meta['minimum'])}, "
                f"maximum {_number(column_meta['maximum'])}. "
            )

        if "coverage" in column_meta:
            # atlas-profiler's clusters, each reported as it is (they are not the column's range)
            ranges = [
                f"{_number(coverage['range']['gte'])} to {_number(coverage['range']['lte'])}"
                for coverage in column_meta["coverage"]
            ]
            column_summary += f"Main value ranges: {'; '.join(ranges)}. "

        profile_summary.append(column_summary)

    rows = metadata.get("nb_rows", 0)
    profiled = metadata.get("nb_profiled_rows", rows)
    coverage_note = (
        "all rows profiled" if profiled >= rows else f"profiled a random sample of {profiled}"
    )
    nb_columns = metadata.get("nb_columns", len(metadata.get("columns", [])))
    header = f"Rows: {rows} ({coverage_note}). Columns: {nb_columns}."
    final_profile_summary = (
        "The key data profile information for this dataset includes:\n"
        + header
        + "\n"
        + "\n".join(profile_summary)
    )

    semantic_summary: list[str] = []
    if "temporal_coverage" in metadata:
        for temp_cov in metadata["temporal_coverage"]:
            column_names = ", ".join(temp_cov.get("column_names", []))
            temporal_resolution = temp_cov.get("temporal_resolution", "unknown")
            ranges = [
                f"{_date(entry['range']['gte'])} to {_date(entry['range']['lte'])}"
                for entry in temp_cov.get("ranges", [])
                if entry.get("range")
            ]
            if ranges or "earliest" in temp_cov:
                text = (
                    f"**Temporal coverage** for columns {column_names} with resolution "
                    f"{temporal_resolution}"
                )
                if "earliest" in temp_cov:
                    text += f", from {_date(temp_cov['earliest'])} to {_date(temp_cov['latest'])}"
                if ranges:
                    text += f"; main ranges {'; '.join(ranges)}"
                semantic_summary.append(text + ".")

    if "spatial_coverage" in metadata:
        for spatial_cov in metadata["spatial_coverage"]:
            column_names = ", ".join(spatial_cov.get("column_names", []))
            spatial_resolution = spatial_cov.get("type", "unknown")
            semantic_summary.append(
                f"**Spatial coverage** for columns {column_names}, "
                f"with type {spatial_resolution}."
            )

    final_semantic_summary = "\n".join(semantic_summary)
    return final_profile_summary, final_semantic_summary
