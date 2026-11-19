"""Check the numbers of a description against its dataset's profile

A deterministic complement to the LLM judges: every number a description states is looked up in
the profile, so made-up or overstated figures can be found without another model call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..description.record import DatasetDescription

Status = Literal["match", "close", "missing"]

# A number written in prose: 3,619 / 12.5 / 40%; not part of a word, a date written with dashes
# or an identifier such as COVID-19
_NUMBER = re.compile(
    r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*(?:%|percent\b))?(?![\w\-/]|\.\d)"
)
_ANY_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?")
_YEARS = (1800, 2100)
CLOSE = 0.02  # relative difference still counted as close ("about 3,600" for 3,619)


@dataclass
class NumberCheck:
    """A number found in a description and what in the profile supports it"""

    text: str
    start: int
    end: int
    value: float
    status: Status
    source: str = ""


def _numbers(text: str) -> Iterable[tuple[re.Match[str], float, bool]]:
    for match in _NUMBER.finditer(text):
        yield match, float(match.group(1).replace(",", "")), bool(match.group(2))


def _facts(metadata: dict[str, Any]) -> list[tuple[float, str]]:
    """Numbers the profile states, each with where it comes from"""
    facts: list[tuple[float, str]] = []

    def add(value: Any, source: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            facts.append((float(value), source))

    add(metadata.get("nb_rows"), "row count")
    add(metadata.get("nb_columns"), "column count")
    for column in metadata.get("columns", []):
        name = column.get("name", "?")
        add(column.get("num_distinct_values"), f"distinct values of {name}")
        for key in ("minimum", "maximum", "mean"):
            add(column.get(key), f"{key} of {name}")
        for coverage in column.get("coverage", []):
            for bound in ("gte", "lte"):
                add(coverage.get("range", {}).get(bound), f"value range of {name}")
    return facts


def _percentages(metadata: dict[str, Any]) -> list[tuple[float, str]]:
    return [
        (100 * column["missing_values_ratio"], f"missing values of {column.get('name', '?')}")
        for column in metadata.get("columns", [])
        if isinstance(column.get("missing_values_ratio"), (int, float))
    ]


def _year_span(metadata: dict[str, Any]) -> tuple[int, int] | None:
    stamps = []
    for column in metadata.get("columns", []):
        stamps += [column[key] for key in ("earliest", "latest") if key in column]
    for temporal in metadata.get("temporal_coverage", []):
        for entry in temporal.get("ranges", []):
            stamps += list(entry.get("range", {}).values())
    years = [datetime.fromtimestamp(s, tz=timezone.utc).year for s in stamps]
    return (min(years), max(years)) if years else None


def _compare(value: float, written: str, fact: float) -> Status | None:
    decimals = len(written.split(".")[1]) if "." in written else 0
    if round(fact, decimals) == value:
        return "match"
    if fact and abs(value - fact) / abs(fact) <= CLOSE:
        return "close"
    return None


def check_numbers(
    text: str,
    metadata: dict[str, Any] | None,
    *,
    sample: str | None = None,
    references: dict[str, str | None] | None = None,
) -> list[NumberCheck]:
    """
    Check every number of a description against a profile

    A number matches when the profile states it (row and column counts, distinct counts,
    extremes, means, value ranges, missing-value percentages) up to the precision it is written
    with, and is close within 2%. A year matches when it falls within the dataset's dates. Numbers
    that appear in the sample or in a reference text (the dataset name, the original description)
    match that text. Everything else is missing: not necessarily wrong, but worth checking.

    Args:
        text: Description
        metadata: Output of ``autoddg.profiling.profile_metadata``
        sample: CSV sample given to the model
        references: Other texts to look for numbers in, by label

    Returns:
        One check per number, in order
    """
    metadata = metadata or {}
    facts = _facts(metadata)
    percentages = _percentages(metadata)
    span = _year_span(metadata)
    column_names = " ".join(str(c.get("name", "")) for c in metadata.get("columns", []))
    texts = {"sample": sample, "column names": column_names, **(references or {})}
    # Any digits count here: values of a sample or a name also sit inside identifiers, dates and
    # snake_case words (2018_central_park)
    in_texts = {
        label: {float(n.replace(",", "")) for n in _ANY_NUMBER.findall(content or "")}
        for label, content in texts.items()
    }

    checks = []
    for match, value, percent in _numbers(text):
        written = match.group(1).replace(",", "")
        status: Status = "missing"
        source = ""
        is_year = not percent and "," not in match.group(1) and value.is_integer()
        is_year = is_year and _YEARS[0] <= value <= _YEARS[1]
        if is_year and span and span[0] <= value <= span[1]:
            status, source = "match", f"within the dates ({span[0]}–{span[1]})"
        else:
            candidates = percentages if percent else facts
            for fact, fact_source in candidates:
                found = _compare(value, written, fact)
                if is_year and found == "close":  # 2023 is not "close" to a value of 2,024.6
                    continue
                if found == "match" or (found == "close" and status == "missing"):
                    status, source = found, f"{fact_source}: {fact:,.10g}"
                    if found == "match":
                        break
        if status == "missing":
            for label, numbers in in_texts.items():
                if value in numbers:
                    status, source = "match", f"in the {label}"
                    break
        checks.append(
            NumberCheck(match.group(0), match.start(), match.end(), value, status, source)
        )
    return checks


def check_description(record: DatasetDescription, text: str | None = None) -> list[NumberCheck]:
    """
    Check a generated description against the profile, sample and texts of its record

    Args:
        record: Output of ``AutoDDG.describe_dataframe`` (or one filled in by hand)
        text: Text to check instead of ``record.description``, e.g. its search-expanded version

    Returns:
        One check per number, in order
    """
    return check_numbers(
        record.description if text is None else text,
        record.metadata,
        sample=record.dataset_sample,
        references={
            "dataset name": record.name,
            "original description": record.original_description,
        },
    )
