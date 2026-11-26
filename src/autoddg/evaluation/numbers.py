"""Check the numbers of a description against its dataset's profile

A deterministic complement to the LLM judges: every number a description states is looked up in
the profile, so made-up or overstated figures can be found without another model call.

What is checked:

- single numbers against the profile's facts: row and column counts, distinct counts, minimums,
  maximums and means, up to the precision they are written with ("match") or within 2% ("close").
  "over 400" and "up to 50" are bounds: they match a fact at most 1.5 times past them. Close
  matches and bounds only count against dataset-level facts (row and column counts, the smallest
  and largest value of any column) and columns the same sentence names
- ranges ("0 to 51", "between 2 and 9", "2005–2023") as pairs against each column's minimum and
  maximum (or earliest and latest date): both ends must belong to the same column
- dates ("May 14, 2020", "2016-03-22", "June 2021", and ranges of them) against each date
  column's earliest and latest dates
- percentages against missing-value ratios and against values (also when stored as fractions)
- "17.5 million" and "3 billion" at the precision they are written with (17.5 million matches
  17,482,113)
- coordinates written with hemispheres: 89.1°W (or 89.1 degrees West) is -89.1, "54 to 141
  degrees West" is -141 to -54, and "42.0°N, 71.3°W to 46.0°N, 68.3°W" is a latitude range and
  a longitude range
- years within the dates of the dataset, and anything that appears in the sample, the column
  names or a reference text (dataset name, original description)

k-means value clusters are not facts about a column, so they never support a number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..description.record import DatasetDescription

Status = Literal["match", "close", "missing"]

CLOSE = 0.02  # relative difference still counted as close ("about 3,600" for 3,619)
BOUND = 1.5  # "over 400" matches a fact from 400 to 600
_YEARS = (1800, 2100)

# Dashes and spaces that models write in place of "-" and " "; replaced one character for one, so
# positions in the normalized text are positions in the original
_NORMALIZE = str.maketrans(dict.fromkeys("‐‑‒–—―−", "-") | {" ": " ", " ": " "})

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?"
# A minus sign only where it can't be a dash: "-73.9" and "from -5 to -3", not "COVID-19" or
# "2005-2023" (the character before a sign must not be part of a word or number)
_SIGNED = rf"-?(?:{_NUM})"
_APPROX = r"(?:(?:about|roughly|approximately|around|nearly|almost|some)\s+)?"
_PCT = r"(\s*(?:%|percent\b))?"
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_SCALE = r"(?:\s+(thousand|million|billion|trillion)\b)?"
# A number written in prose: 3,619 / 12.5 / 40% / -73.9 / 17.5 million; not part of a word or of
# an identifier such as COVID-19 or 18b
_SINGLE = re.compile(
    rf"(?<![\w.\-/])({_SIGNED}){_PCT}{_SCALE}(?![\w\-/]|\.\d)", re.IGNORECASE
)
_RANGE = re.compile(
    rf"(?<![\w.\-/])(?:(?P<between>between)\s+)?{_APPROX}\$?(?P<low>{_SIGNED})"
    rf"(?P<low_pct>\s*(?:%|percent\b))?(?:\s+(?P<low_scale>thousand|million|billion|trillion)\b)?"
    rf"\s*(?(between)and|(?:-|to|through|until))\s*{_APPROX}\$?(?P<high>{_SIGNED})"
    rf"(?P<high_pct>\s*(?:%|percent\b))?(?:\s+(?P<high_scale>thousand|million|billion|trillion)\b)?"
    r"(?![\w\-/]|\.\d)",
    re.IGNORECASE,
)
_WORD_SEPARATOR = re.compile(r"\b(?:and|to|through|until)\b", re.IGNORECASE)
# Coordinates written with a hemisphere: 89.1°W is -89.1. Rewritten in place, padded with spaces
# so positions don't move
_HEMISPHERES = r"(North|South|East|West|[NSEW])(?![A-Za-z])"
_DEGREES = r"\s?(?:°|degrees?\s)\s?"
_HEMISPHERE = re.compile(rf"(?<![\w.\-])(\d+(?:\.\d+)?){_DEGREES}{_HEMISPHERES}")
# One hemisphere for both ends: "54 to 141 degrees West", "22-24°N"
_HEMISPHERE_RANGE = re.compile(
    rf"(?<![\w.\-])(\d+(?:\.\d+)?)°?\s*(?:-|to|through|and)\s*(\d+(?:\.\d+)?){_DEGREES}{_HEMISPHERES}"
)
# Two points, each a latitude and a longitude in either order: "42.0°N, 71.3°W to 46.0°N, 68.3°W"
_COORD = r"(\d+(?:\.\d+)?)\s?°\s?([NSEW])"
_POINT = rf"{_COORD}\s*,?\s*{_COORD}"
_POINT_RANGE = re.compile(
    rf"(?<![\w.\-])(?:(between)\s+)?\(?{_POINT}\)?\s*(?(1)and|(?:-|to|through|until))\s*"
    rf"\(?{_POINT}\)?(?![A-Za-z])",
    re.IGNORECASE,
)
_HEDGED = re.compile(
    r"(?:about|roughly|approximately|around|nearly|almost|some)\s+\$?$", re.IGNORECASE
)
_ANY_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?")
_NEGATIVE = re.compile(r"(?<![\w.])-(\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)")
_LOWER = re.compile(
    r"(?:over|more than|above|exceeds?|exceeding|at least|upwards of)\s+" + _APPROX + r"\$?$",
    re.IGNORECASE,
)
_UPPER = re.compile(
    r"(?:up to|under|less than|below|at most|fewer than|reaching)\s+" + _APPROX + r"\$?$",
    re.IGNORECASE,
)

_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
}
_MONTH = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DATE_PATTERNS = (
    # 2016-03-22
    (re.compile(r"(?<![\w\-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\w\-])"), ("y", "m", "d")),
    # May 14, 2020
    (
        re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I),
        ("M", "d", "y"),
    ),
    # 14 May 2020
    (re.compile(rf"\b(\d{{1,2}})\s+{_MONTH}\s+(\d{{4}})\b", re.I), ("d", "M", "y")),
    # May 2020
    (re.compile(rf"\b{_MONTH}\s+(\d{{4}})\b", re.I), ("M", "y")),
)
_DATE_RANGE_GAP = re.compile(r"^\s*(?:-|to|through|until|and)\s*$", re.I)


@dataclass
class NumberCheck:
    """A number (or date) found in a description and what in the profile supports it"""

    text: str
    start: int
    end: int
    value: float
    status: Status
    source: str = ""


@dataclass
class _Date:
    start: int
    end: int
    first: date  # first day the text covers
    last: date  # last day it covers ("May 2020": May 1 to May 31)


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _fmt(value: float) -> str:
    return f"{value:,.10g}"


def _day(timestamp: float) -> date:
    # not datetime.fromtimestamp: it fails on dates before 1970 on Windows
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=timestamp)).date()


def _signed_coordinates(text: str) -> str:
    """89.1°W -> "-89.1  ", 42°N -> "42  ": same length, so positions still index the original"""

    def rewrite(match: re.Match[str]) -> str:
        number = ("-" if match.group(2)[0] in "SW" else "") + match.group(1)
        return number.ljust(len(match.group(0)))

    return _HEMISPHERE.sub(rewrite, text).replace("°", " ")


def _numbers_in(content: str) -> set[float]:
    """Every number in a text, and the negative ones also with their sign"""
    found = {_number(n) for n in _ANY_NUMBER.findall(content)}
    return found | {-_number(n) for n in _NEGATIVE.findall(content)}


def _timestamp(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def _compare(value: float, written: str, fact: float, scale: float = 1.0) -> Status | None:
    """value is written as ``written`` times ``scale`` ("17.5" million)"""
    decimals = len(written.split(".")[1]) if "." in written else 0
    if math.isclose(round(fact / scale, decimals), value / scale, rel_tol=1e-12, abs_tol=0):
        return "match"
    if fact and abs(value - fact) / abs(fact) <= CLOSE:
        return "close"
    return None


def _columns(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in metadata.get("columns", []) if isinstance(c, dict)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _facts(metadata: dict[str, Any]) -> list[tuple[float, str, str | None]]:
    """
    Numbers the profile states about the data: (value, where it comes from, column)

    The column is None for dataset-level facts: row and column counts, and the smallest and
    largest value of any column.
    """
    facts: list[tuple[float, str, str | None]] = [
        (float(metadata[key]), label, None)
        for key, label in (("nb_rows", "row count"), ("nb_columns", "column count"))
        if _is_number(metadata.get(key))
    ]
    for column in _columns(metadata):
        name = str(column.get("name", "?"))
        for key, label in (
            ("num_distinct_values", "distinct values"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("mean", "mean"),
        ):
            if _is_number(column.get(key)):
                facts.append((float(column[key]), f"{label} of {name}", name))
    minimums = [(f, s) for f, s, c in facts if s.startswith("minimum of ")]
    maximums = [(f, s) for f, s, c in facts if s.startswith("maximum of ")]
    if minimums:
        value, label = min(minimums)
        facts.append((value, f"smallest value ({label})", None))
    if maximums:
        value, label = max(maximums)
        facts.append((value, f"largest value ({label})", None))
    return facts


def _percent_facts(
    metadata: dict[str, Any], facts: list[tuple[float, str, str | None]]
) -> list[tuple[float, str, str | None]]:
    percents: list[tuple[float, str, str | None]] = [
        (100 * c["missing_values_ratio"], f"missing values of {c.get('name', '?')}", None)
        for c in _columns(metadata)
        if _is_number(c.get("missing_values_ratio"))
    ]
    # values may be percentages already, or fractions (0.53 for 53%)
    values = [(f, s, c) for f, s, c in facts if s not in ("row count", "column count")]
    percents += values
    percents += [(100 * f, f"{s} ({_fmt(f)})", c) for f, s, c in values if 0 < abs(f) <= 1]
    return percents


def _named(column: str, sentence: str) -> bool:
    """True when the sentence mentions the column: a word of its name starts a word of the text"""
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    tokens = [t for t in re.findall(r"[a-z0-9]+", column.lower()) if len(t) > 1]
    return any(word.startswith(token) for token in tokens for word in words)


def _sentence(text: str, position: int) -> str:
    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position)) + 1
    end = min(
        i for i in (text.find(". ", position), text.find("\n", position), len(text)) if i >= 0
    )
    return text[start:end]


def _ranges(metadata: dict[str, Any]) -> list[tuple[float, float, str]]:
    return [
        (float(c["minimum"]), float(c["maximum"]), str(c.get("name", "?")))
        for c in _columns(metadata)
        if _is_number(c.get("minimum")) and _is_number(c.get("maximum"))
    ]


def _date_ranges(metadata: dict[str, Any]) -> list[tuple[date, date, str]]:
    return [
        (_day(c["earliest"]), _day(c["latest"]), str(c.get("name", "?")))
        for c in _columns(metadata)
        if _is_number(c.get("earliest")) and _is_number(c.get("latest"))
    ]


def _find_dates(text: str) -> list[_Date]:
    found: list[_Date] = []
    taken: list[tuple[int, int]] = []
    for pattern, parts in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            values = dict(zip(parts, match.groups(), strict=True))
            try:
                year = int(values["y"])
                month = _MONTHS[values["M"][:3].lower()] if "M" in values else int(values["m"])
                if "d" in values:
                    first = last = date(year, month, int(values["d"]))
                else:
                    first = date(year, month, 1)
                    following = date(year + month // 12, month % 12 + 1, 1)
                    last = date.fromordinal(following.toordinal() - 1)
            except (ValueError, KeyError):
                continue
            found.append(_Date(match.start(), match.end(), first, last))
            taken.append((match.start(), match.end()))
    return sorted(found, key=lambda d: d.start)


def check_numbers(
    text: str,
    metadata: dict[str, Any] | None,
    *,
    sample: str | None = None,
    references: dict[str, str | None] | None = None,
) -> list[NumberCheck]:
    """
    Check every number and date of a description against a profile

    See the module documentation for the rules. A number that is "missing" is not necessarily
    wrong, but nothing in the profile, the sample or the references supports it.

    Args:
        text: Description
        metadata: Output of ``autoddg.profiling.profile_metadata``
        sample: CSV sample given to the model
        references: Other texts to look for numbers in, by label

    Returns:
        One check per number or date, in order
    """
    metadata = metadata or {}
    normalized = _signed_coordinates(text.translate(_NORMALIZE))
    facts = _facts(metadata)
    percent_facts = _percent_facts(metadata, facts)
    ranges = _ranges(metadata)
    date_ranges = _date_ranges(metadata)
    years = [y for first, last, _ in date_ranges for y in (first.year, last.year)]
    span = (min(years), max(years)) if years else None
    column_names = " ".join(str(c.get("name", "")) for c in _columns(metadata))
    texts = {"sample": sample, "column names": column_names, **(references or {})}
    # Any digits count here: values of a sample or a name also sit inside identifiers, dates and
    # snake_case words (2018_central_park)
    in_texts = {label: _numbers_in(content or "") for label, content in texts.items()}
    raw_texts = {label: (content or "").translate(_NORMALIZE) for label, content in texts.items()}

    checks: list[NumberCheck] = []
    taken: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return not any(start < e and s < end for s, e in taken)

    def add(start: int, end: int, value: float, status: Status, source: str) -> None:
        checks.append(NumberCheck(text[start:end], start, end, value, status, source))
        taken.append((start, end))

    def in_references(value: float) -> str | None:
        for label, numbers in in_texts.items():
            if value in numbers:
                return f"in the {label}"
        return None

    def literal_in_references(snippet: str) -> str | None:
        for label, content in raw_texts.items():
            if snippet and snippet in content:
                return f"in the {label}"
        return None

    # 1. Dates, and ranges of two dates
    dates = _find_dates(normalized)
    paired: set[int] = set()
    for i, (a, b) in enumerate(zip(dates, dates[1:], strict=False)):
        if not _DATE_RANGE_GAP.match(normalized[a.end : b.start]):
            continue
        paired |= {i, i + 1}
        exact = [
            n for lo, hi, n in date_ranges if a.first <= lo <= a.last and b.first <= hi <= b.last
        ]
        if exact:
            status, source = "match", f"dates of {exact[0]}"
        else:
            status = "missing"
            source = "no date column has this range" + "".join(
                f"; {n}: {lo} to {hi}" for lo, hi, n in date_ranges[:3]
            )
        for d in (a, b):
            add(d.start, d.end, _timestamp(d.first), status, source)
    for i, d in enumerate(dates):
        if i in paired:
            continue
        within = [n for lo, hi, n in date_ranges if d.last >= lo and d.first <= hi]
        literal = literal_in_references(normalized[d.start : d.end])
        if within:
            status, source = "match", f"within the dates of {within[0]}"
        elif literal:
            status, source = "match", literal
        else:
            status = "missing"
            source = "outside the dates" + "".join(
                f"; {n}: {lo} to {hi}" for lo, hi, n in date_ranges[:3]
            )
        add(d.start, d.end, _timestamp(d.first), status, source)

    def coordinate_range(ends: list[tuple[float, str, int, int]]) -> None:
        (low, low_text, *_), (high, high_text, *_) = sorted(ends)
        status: Status = "missing"
        source = "no column has this range"
        for lo, hi, name in ranges:
            found = (_compare(low, low_text, lo), _compare(high, high_text, hi))
            if found == ("match", "match"):
                status, source = "match", f"range of {name}: {_fmt(lo)} to {_fmt(hi)}"
                break
            if None not in found and status == "missing":
                status, source = "close", f"range of {name}: {_fmt(lo)} to {_fmt(hi)}"
        for value, _, start, end in ends:
            add(start, end, value, status, source)

    # 2. Ranges of two numbers: coordinates first, then plain ranges
    plain = text.translate(_NORMALIZE)
    for match in _POINT_RANGE.finditer(plain):
        groups = [
            (match.start(g), match.end(g), match.group(g), match.group(g + 1).upper())
            for g in (2, 4, 6, 8)
        ]
        axes: dict[str, list[tuple[float, str, int, int]]] = {"lat": [], "lon": []}
        for start, end, written, hemisphere in groups:
            value = -_number(written) if hemisphere in "SW" else _number(written)
            axes["lat" if hemisphere in "NS" else "lon"].append((value, written, start, end))
        if any(len(ends) != 2 for ends in axes.values()):
            continue  # "42°N, 46°N": not two points
        for ends in axes.values():
            coordinate_range(ends)
    for match in _HEMISPHERE_RANGE.finditer(plain):
        if not free(match.start(1), match.end(2)):
            continue
        sign = -1 if match.group(3)[0].upper() in "SW" else 1
        coordinate_range(
            [
                (sign * _number(match.group(g)), match.group(g), match.start(g), match.end(g))
                for g in (1, 2)
            ]
        )

    for match in _RANGE.finditer(normalized):
        if not free(match.start("low"), match.end("high")):
            continue
        low_text, high_text = match.group("low"), match.group("high")
        low_scale = _SCALES.get((match.group("low_scale") or "").lower(), 1.0)
        high_scale = _SCALES.get((match.group("high_scale") or "").lower(), 1.0)
        if low_scale != high_scale and match.group("low_scale") is None:
            low_scale = high_scale  # "1.2 to 3.4 million"
        low, high = _number(low_text) * low_scale, _number(high_text) * high_scale
        low_end = match.end("low_pct") if match.group("low_pct") else match.end("low")
        between = normalized[low_end : match.start("high")]
        # "89.1°W to 92.5°W" is -89.1 to -92.5: a range written high to low. Only negative ends
        # are read that way; "30% in 2013 to about 53%" is not a range
        reversed_range = low > high and high < 0 and bool(_WORD_SEPARATOR.search(between))
        if low >= high and not reversed_range:
            continue  # "4-3" or a list, not a range
        if reversed_range:
            low_text, high_text, low, high = high_text, low_text, high, low
            low_scale, high_scale = high_scale, low_scale
        is_years = low_scale == high_scale == 1.0 and all(
            "," not in t and v.is_integer() and _YEARS[0] <= v <= _YEARS[1]
            for t, v in ((low_text, low), (high_text, high))
        )
        status: Status = "missing"
        source = ""
        if is_years:
            exact = [n for lo, hi, n in date_ranges if (lo.year, hi.year) == (low, high)]
            # not the sample: any two years of a date column sit somewhere in its values
            named = [
                label
                for label, nums in in_texts.items()
                if label != "sample" and low in nums and high in nums
            ]
            if exact:
                status, source = "match", f"years of {exact[0]}"
            elif named:
                status, source = "match", f"in the {named[0]}"
            else:
                source = "no date column spans these years" + "".join(
                    f"; {n}: {lo.year} to {hi.year}" for lo, hi, n in date_ranges[:3]
                )
        else:
            for lo, hi, name in ranges:
                ends = (
                    _compare(low, low_text, lo, low_scale),
                    _compare(high, high_text, hi, high_scale),
                )
                if ends == ("match", "match"):
                    status, source = "match", f"range of {name}: {_fmt(lo)} to {_fmt(hi)}"
                    break
                if None not in ends and status == "missing":
                    status, source = "close", f"range of {name}: {_fmt(lo)} to {_fmt(hi)}"
            if status == "missing":
                literal = literal_in_references(normalized[match.start("low") : match.end("high")])
                if literal:
                    status, source = "match", literal
                else:
                    near = [
                        f"{n}: {_fmt(lo)} to {_fmt(hi)}"
                        for lo, hi, n in ranges
                        if _compare(low, low_text, lo, low_scale)
                        or _compare(high, high_text, hi, high_scale)
                    ]
                    source = "no column has this range" + (
                        "; " + "; ".join(near[:3]) if near else ""
                    )
        for group, scale in (("low", low_scale), ("high", high_scale)):
            if reversed_range:
                scale = high_scale if group == "low" else low_scale
            value = _number(match.group(group)) * scale
            add(match.start(group), match.end(group), value, status, source)

    # 3. Single numbers
    for match in _SINGLE.finditer(normalized):
        start, end = match.start(), match.end()
        if not free(start, end):
            continue
        written = match.group(1).replace(",", "")
        scale = _SCALES.get((match.group(3) or "").lower(), 1.0)
        value = _number(match.group(1)) * scale
        percent = bool(match.group(2))
        before = normalized[max(0, start - 40) : start]
        bound = "lower" if _LOWER.search(before) else "upper" if _UPPER.search(before) else None
        is_year = not percent and scale == 1.0 and "," not in match.group(1) and value.is_integer()
        is_year = is_year and _YEARS[0] <= value <= _YEARS[1]
        status = "missing"
        source = ""
        if is_year and span and span[0] <= value <= span[1]:
            status, source = "match", f"within the dates ({span[0]}–{span[1]})"
        else:
            candidates = percent_facts if percent else facts
            sentence = _sentence(normalized, start)

            def nearby(column: str | None, sentence: str = sentence) -> bool:
                # approximate matches only count against dataset-level facts and columns the
                # sentence names: "about 2,000 segments" is not about 1,991 distinct street names
                return column is None or _named(column, sentence)

            if bound and not is_year:
                for fact, fact_source, column in candidates:
                    inside = (
                        value <= fact <= value * BOUND
                        if bound == "lower"
                        else value / BOUND <= fact <= value
                    )
                    if inside and nearby(column):
                        word = "over" if bound == "lower" else "up to"
                        status = "match"
                        source = f"{word} {match.group(1)}: {fact_source}: {_fmt(fact)}"
                        break
            if status == "missing":
                hedged = bool(_HEDGED.search(before))
                for fact, fact_source, column in candidates:
                    found = _compare(value, written, fact, scale)
                    if is_year and found == "close":  # 2023 is not "close" to a value of 2,024.6
                        continue
                    if found == "close" and not nearby(column):
                        continue
                    # 595 is not close to "over 600", unless the text hedges ("over about 600")
                    wrong_side = (bound == "lower" and fact < value) or (
                        bound == "upper" and fact > value
                    )
                    if found == "close" and wrong_side and not hedged:
                        continue
                    if found == "match" or (found == "close" and status == "missing"):
                        status, source = found, f"{fact_source}: {_fmt(fact)}"
                        if found == "match":
                            break
        if status == "missing":
            source = in_references(value) or ""
            status = "match" if source else "missing"
        add(start, end, value, status, source)

    return sorted(checks, key=lambda c: c.start)


def check_description(record: DatasetDescription, text: str | None = None) -> list[NumberCheck]:
    """
    Check a generated description against the profile, sample and texts of its record

    Args:
        record: Output of ``AutoDDG.describe_dataframe`` (or one filled in by hand)
        text: Text to check instead of ``record.description``, e.g. its search-expanded version

    Returns:
        One check per number or date, in order
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
