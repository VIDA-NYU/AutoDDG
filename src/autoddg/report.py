"""Review page for generated descriptions

Renders a batch of descriptions as one standalone HTML page: each dataset's generated description
next to its original one, with every number in the generated text checked against the dataset's
profile, and the inputs the description was built from (profile, semantic profile, topic, prompt)
one click away.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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


@dataclass
class ReportEntry:
    """One dataset of the report

    Args:
        name: Dataset name
        description: Generated description
        original_description: Existing description, shown next to the generated one
        metadata: Output of ``autoddg.profiling.profile_metadata``, used to check numbers
        dataset_sample: CSV sample given to the model
        topic: Generated topic
        search_description: Search-expanded description
        profile_text: Profile summary given to the model
        semantic_profile: Semantic profile given to the model
        prompt: Prompt that produced the description
        profile_report_url: Link to the dataset's atlas-profiler report
    """

    name: str
    description: str
    original_description: str | None = None
    metadata: dict[str, Any] | None = None
    dataset_sample: str | None = None
    topic: str | None = None
    search_description: str | None = None
    profile_text: str | None = None
    semantic_profile: str | None = None
    prompt: str | None = None
    profile_report_url: str | None = None
    checks: list[NumberCheck] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.checks = check_numbers(
            self.description,
            self.metadata,
            sample=self.dataset_sample,
            references={
                "dataset name": self.name,
                "original description": self.original_description,
            },
        )


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


def _highlight(text: str, checks: list[NumberCheck]) -> str:
    parts, last = [], 0
    for check in checks:
        parts.append(html.escape(text[last : check.start]))
        title = check.source or "not found in the profile, the sample or the references"
        parts.append(
            f'<mark class="{check.status}" title="{html.escape(title)}">'
            f"{html.escape(text[check.start : check.end])}</mark>"
        )
        last = check.end
    parts.append(html.escape(text[last:]))
    return "".join(parts).replace("\n", "<br>")


def _details(summary: str, body: str | None, pre: bool = True) -> str:
    if not body:
        return ""
    content = f"<pre>{html.escape(body)}</pre>" if pre else body
    return f"<details><summary>{summary}</summary>{content}</details>"


def _entry_html(entry: ReportEntry, index: int) -> str:
    missing = sum(check.status == "missing" for check in entry.checks)
    meta = entry.metadata or {}
    facts = []
    if "nb_rows" in meta:
        facts.append(f"{meta['nb_rows']:,} rows")
    if "nb_columns" in meta:
        facts.append(f"{meta['nb_columns']} columns")
    if meta.get("sampled"):
        facts.append(f"{meta.get('nb_profiled_rows', 0):,} profiled")
    if entry.topic:
        facts.append(f"topic: {html.escape(entry.topic)}")
    if entry.profile_report_url:
        facts.append(f'<a href="{html.escape(entry.profile_report_url)}">profile report</a>')
    badge = (
        f'<span class="badge missing">{missing} number{"s" * (missing != 1)} to check</span>'
        if missing
        else '<span class="badge match">numbers check out</span>' if entry.checks else ""
    )
    original = (
        f"<p>{html.escape(entry.original_description)}</p>"
        if entry.original_description
        else '<p class="none">No original description.</p>'
    )
    search = (
        _highlight(
            entry.search_description,
            check_numbers(
                entry.search_description,
                entry.metadata,
                sample=entry.dataset_sample,
                references={
                    "dataset name": entry.name,
                    "original description": entry.original_description,
                },
            ),
        )
        if entry.search_description
        else None
    )
    return f"""
<section class="entry" id="d{index}" data-missing="{missing}">
  <header>
    <h2>{html.escape(entry.name)}</h2>{badge}
    <p class="facts">{" · ".join(facts)}</p>
  </header>
  <div class="columns">
    <div><h3>Original</h3>{original}</div>
    <div><h3>Generated</h3><p>{_highlight(entry.description, entry.checks)}</p></div>
  </div>
  {_details("Search-expanded description", f"<p>{search}</p>" if search else None, pre=False)}
  {_details("Profile given to the model", entry.profile_text)}
  {_details("Semantic profile given to the model", entry.semantic_profile)}
  {_details("Prompt", entry.prompt)}
</section>"""


_CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e4e3df;
  --ok:#d9f0dd; --ok-fg:#1f5f2c; --close:#fbefc8; --close-fg:#6e5300;
  --bad:#fbd9d5; --bad-fg:#8a1f12; --accent:#2b59c3; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#161615; --fg:#e9e8e4; --muted:#a19f98; --card:#1f1f1d; --line:#33322f;
  --ok:#1e3a24; --ok-fg:#a6e0b1; --close:#3d3313; --close-fg:#f0d987;
  --bad:#4a1f1a; --bad-fg:#f5b3aa; --accent:#8fb0ff; } }
:root[data-theme="dark"] { --bg:#161615; --fg:#e9e8e4; --muted:#a19f98; --card:#1f1f1d;
  --line:#33322f; --ok:#1e3a24; --ok-fg:#a6e0b1; --close:#3d3313; --close-fg:#f0d987; --bad:#4a1f1a;
  --bad-fg:#f5b3aa; --accent:#8fb0ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; }
.lede, .facts, .none { color:var(--muted); }
.controls { display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:16px 0 24px;
  padding:12px 16px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
.legend mark { margin-right:6px; }
.entry { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:16px 20px; margin-bottom:16px; }
.entry header h2 { display:inline; font-size:17px; margin-right:10px; overflow-wrap:anywhere; }
.facts { margin:4px 0 0; font-size:13px; }
.columns { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-top:12px; }
@media (max-width:760px) { .columns { grid-template-columns:1fr; } }
h3 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
  margin:0 0 4px; }
.columns p { margin:0; }
mark { border-radius:3px; padding:0 2px; }
mark.match { background:var(--ok); color:var(--ok-fg); }
mark.close { background:var(--close); color:var(--close-fg); }
mark.missing { background:var(--bad); color:var(--bad-fg); }
.badge { font-size:12px; border-radius:10px; padding:2px 8px; white-space:nowrap; }
.badge.match { background:var(--ok); color:var(--ok-fg); }
.badge.missing { background:var(--bad); color:var(--bad-fg); }
details { margin-top:10px; border-top:1px solid var(--line); padding-top:8px; }
summary { cursor:pointer; color:var(--accent); font-size:14px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; font:12.5px/1.5 ui-monospace, Menlo, monospace;
  background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; }
a { color:var(--accent); }
body.only-missing .entry[data-missing="0"] { display:none; }
"""


class DescriptionReport:
    """
    Standalone HTML page for reviewing a batch of generated descriptions

    Examples:
        >>> report = DescriptionReport([ReportEntry(name="trips", description=text, metadata=m)])
        >>> report.save("descriptions.html")
    """

    def __init__(self, entries: list[ReportEntry], title: str = "Dataset descriptions") -> None:
        self.entries = entries
        self.title = title

    def to_html(self) -> str:
        """Render the page"""
        checks = [check for entry in self.entries for check in entry.checks]
        counts = {s: sum(c.status == s for c in checks) for s in ("match", "close", "missing")}
        flagged = sum(any(c.status == "missing" for c in e.checks) for e in self.entries)
        entries = "".join(_entry_html(entry, i) for i, entry in enumerate(self.entries))
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>
<body><main>
<h1>{html.escape(self.title)}</h1>
<p class="lede">{len(self.entries)} datasets · {len(checks)} numbers in the generated
descriptions: {counts["match"]} found in the profile, {counts["close"]} close,
{counts["missing"]} not found ({flagged} dataset{"s" * (flagged != 1)} to check).</p>
<div class="controls">
  <label><input type="checkbox" id="only-missing"> Only datasets with numbers to check</label>
  <span class="legend"><mark class="match">found</mark><mark class="close">within 2%</mark><mark
  class="missing">not found</mark> Hover a number to see its source.</span>
</div>
{entries}
</main>
<script>
document.getElementById("only-missing").addEventListener("change", (e) =>
  document.body.classList.toggle("only-missing", e.target.checked));
</script>
</body></html>
"""

    def save(self, path: str | Path) -> Path:
        """Write the page to ``path`` and return it"""
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return self.to_html()
