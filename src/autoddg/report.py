"""Review page for generated descriptions

Renders a batch of ``DatasetDescription`` records as one standalone HTML page. The page only
displays: the descriptions come from ``AutoDDG.describe_dataframe`` and the number checks from
``autoddg.evaluation.check_description``.
"""

from __future__ import annotations

import html
from pathlib import Path

from .description.record import DatasetDescription
from .evaluation.numbers import NumberCheck, check_description


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


def _entry_html(
    entry: DatasetDescription, checks: list[NumberCheck], link: str | None, index: int
) -> str:
    missing = sum(check.status == "missing" for check in checks)
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
    if link:
        facts.append(f'<a href="{html.escape(link)}">profile report</a>')
    badge = (
        f'<span class="badge missing">{missing} number{"s" * (missing != 1)} to check</span>'
        if missing
        else '<span class="badge match">numbers check out</span>' if checks else ""
    )
    original = (
        f"<p>{html.escape(entry.original_description)}</p>"
        if entry.original_description
        else '<p class="none">No original description.</p>'
    )
    search = (
        _highlight(entry.search_description, check_description(entry, entry.search_description))
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
    <div><h3>Generated</h3><p>{_highlight(entry.description, checks)}</p></div>
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

    Each dataset's generated description is shown next to its original one, with every number
    checked against the profile (see ``autoddg.evaluation.check_description``), and the inputs it
    was built from one click away.

    Args:
        records: Outputs of ``AutoDDG.describe_dataframe``
        title: Page title
        links: Optional link per dataset name, e.g. to its atlas-profiler report

    Examples:
        >>> records = [ddg.describe_dataframe(df, title="Taxi trips") for df in frames]
        >>> DescriptionReport(records).save("descriptions.html")
    """

    def __init__(
        self,
        records: list[DatasetDescription],
        title: str = "Dataset descriptions",
        links: dict[str, str] | None = None,
    ) -> None:
        self.records = records
        self.title = title
        self.links = links or {}
        self.checks = [check_description(record) for record in records]

    def to_html(self) -> str:
        """Render the page"""
        checks = [check for record_checks in self.checks for check in record_checks]
        counts = {s: sum(c.status == s for c in checks) for s in ("match", "close", "missing")}
        flagged = sum(any(c.status == "missing" for c in cs) for cs in self.checks)
        entries = "".join(
            _entry_html(record, record_checks, self.links.get(record.name), i)
            for i, (record, record_checks) in enumerate(zip(self.records, self.checks, strict=True))
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>
<body><main>
<h1>{html.escape(self.title)}</h1>
<p class="lede">{len(self.records)} datasets · {len(checks)} numbers in the generated
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
