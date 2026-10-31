from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any

import datamart_profiler
import numpy
import pandas
from beartype import beartype
from datamart_profiler.core import MAX_SIZE, RANDOM_SEED
from datamart_profiler.profile_types import parse_dates
from pandas import DataFrame

NUMERIC_TYPES = ("Integer", "Float")
DATE_TIME = "http://schema.org/DateTime"


def _plain_index(frame: DataFrame) -> bool:
    """True for a 0..n-1 integer index, the only index datamart does not turn into columns"""
    return frame.index.dtype == numpy.int64 and pandas.Index(numpy.arange(len(frame))).equals(
        frame.index
    )


@beartype
def profile_dataset(data_frame: DataFrame) -> tuple[str, str]:
    """
    Run datamart profiling and produce short textual summaries

    Ref: https://pypi.org/project/datamart-profiler/

    Args:
        data_frame: Input frame

    Returns:
        (profile_text, semantic_notes)
    """

    return summarize(profile_metadata(data_frame))


@beartype
def profile_metadata(data_frame: DataFrame, nb_rows: int | None = None) -> dict[str, Any]:
    """
    Run datamart profiling and add the true extremes of each column

    Large frames are sampled with datamart's own rule for files: when the frame's text exceeds
    datamart's MAX_SIZE (5 MB), ceil(MAX_SIZE / size * rows) rows are drawn with datamart's
    RANDOM_SEED. The result records the full row count and whether it profiled a sample.

    Args:
        data_frame: Input frame
        nb_rows: Full row count, when the caller has already sampled the rows (for example while
            reading a large file); the profile is then marked as sampled

    Returns:
        datamart metadata, plus 'nb_rows', 'sampled' and, per column, 'minimum' and 'maximum'
        (numeric) or 'earliest' and 'latest' (dates, as UTC timestamps)
    """

    total = len(data_frame) if nb_rows is None else nb_rows

    # datamart treats column names as strings and fails on the integer names pandas gives a
    # headerless sheet
    if not all(isinstance(column, str) for column in data_frame.columns):
        data_frame = data_frame.set_axis([str(column) for column in data_frame.columns], axis=1)

    # datamart samples a file to MAX_SIZE but profiles every row of a DataFrame (k-means over every
    # value), so apply its file rule here; the size is measured in chunks so a large frame is never
    # copied whole
    size = 0
    for start in range(0, len(data_frame), 100_000):
        text = data_frame.iloc[start : start + 100_000].astype(object).fillna("").astype(str)
        size += int(sum(text.iloc[:, i].str.len().sum() for i in range(text.shape[1]))) + text.size
    if size > MAX_SIZE and len(data_frame) > 1:
        keep = math.ceil(MAX_SIZE / size * len(data_frame))
        rows = sorted(random.Random(RANDOM_SEED).sample(range(len(data_frame)), keep))
        plain = _plain_index(data_frame)
        data_frame = data_frame.iloc[rows]
        if plain:  # a sampled row number is not an index datamart should turn into a column
            data_frame = data_frame.reset_index(drop=True)

    metadata = datamart_profiler.process_dataset(data_frame)
    metadata["nb_rows"] = total
    metadata["sampled"] = metadata.get("nb_profiled_rows", 0) < total

    # datamart's coverage keeps the 5th-95th percentiles of up to three k-means clusters and drops
    # clusters under 10% of the values, so it is not the column's range: add the true extremes over
    # the profiled rows, parsing values the way datamart does
    columns = metadata.get("columns", [])
    frame = data_frame if _plain_index(data_frame) else data_frame.reset_index()
    for i, column_meta in enumerate(columns):
        values = frame.iloc[:, i].astype(object).fillna("").astype(str)
        if column_meta.get("structural_type", "").split("/")[-1] in NUMERIC_TYPES:
            numbers = []
            for value in values:
                try:
                    number = float(value)
                except ValueError:
                    continue
                if -3.4e38 < number < 3.4e38:  # datamart's own bounds
                    numbers.append(number)
            if numbers:
                column_meta["minimum"], column_meta["maximum"] = min(numbers), max(numbers)
        if DATE_TIME in column_meta.get("semantic_types", []):
            dates = parse_dates(values)
            if dates:
                column_meta["earliest"] = min(dates).timestamp()
                column_meta["latest"] = max(dates).timestamp()
    for temp_cov in metadata.get("temporal_coverage", []):
        index = temp_cov.get("column_indexes", [None])[0]
        if index is not None and "earliest" in columns[index]:
            temp_cov["earliest"] = columns[index]["earliest"]
            temp_cov["latest"] = columns[index]["latest"]
    return metadata


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _date(timestamp: float) -> str:
    # datamart's timestamps are UTC; formatting them in the local time zone shifts the dates
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
            # datamart's clusters, each reported as it is (they are not the column's range)
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
