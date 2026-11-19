"""Description review page (autoddg.report)"""

from __future__ import annotations

from autoddg import DescriptionReport, ReportEntry
from autoddg.report import check_numbers

METADATA = {
    "nb_rows": 3619,
    "nb_columns": 13,
    "columns": [
        {"name": "count", "num_distinct_values": 4, "minimum": 0.0, "maximum": 3.0, "mean": 1.16},
        {"name": "position", "missing_values_ratio": 0.067},
        {"name": "2021", "maximum": 3578.7},
        {"name": "installed", "earliest": 946684800.0, "latest": 1609459200.0},  # 2000 to 2021
    ],
}


def statuses(text, **kwargs):
    return [(c.text, c.status) for c in check_numbers(text, METADATA, **kwargs)]


def test_numbers_are_checked_against_the_profile():
    assert statuses("It lists 3,619 fountains in 13 columns, at most 3 per site.") == [
        ("3,619", "match"),
        ("13", "match"),
        ("3", "match"),
    ]


def test_rounded_and_approximate_numbers():
    assert statuses("A mean of 1.2 and about 3,600 dollars at most, or 3,700.") == [
        ("1.2", "match"),  # 1.16 written with one decimal
        ("3,600", "close"),  # within 2% of 3,578.7, but not what the data says
        ("3,700", "missing"),
    ]


def test_percentages_are_checked_against_missing_values():
    assert statuses("About 7% of positions are missing, not 20 percent.") == [
        ("7%", "match"),
        ("20 percent", "missing"),
    ]


def test_years_match_dates_column_names_or_references_but_are_never_close():
    assert statuses("From 2000 to 2021, and 2023, with 2021 values.") == [
        ("2000", "match"),
        ("2021", "match"),
        ("2023", "missing"),  # outside the dates, and not "close" to 2,021
        ("2021", "match"),
    ]
    named = statuses("A 2018 census.", references={"dataset name": "2018_squirrel_census"})
    assert named == [("2018", "match")]


def test_identifiers_and_dashed_dates_are_not_numbers():
    assert statuses("COVID-19 data from 2006-present, see Local Law 18b.") == []


def test_page_marks_numbers_and_escapes_text():
    entry = ReportEntry(
        name="fountains <b>",
        description="It lists 3,619 fountains and 99 benches.",
        original_description="Fountains & more",
        metadata=METADATA,
        profile_report_url="reports/fountains.html",
    )
    page = DescriptionReport([entry], title="Review").to_html()
    assert '<mark class="match" title="row count: 3,619">3,619</mark>' in page
    assert '<mark class="missing"' in page and ">99</mark>" in page
    assert "fountains &lt;b&gt;" in page and "Fountains &amp; more" in page
    assert 'href="reports/fountains.html"' in page
    assert "1 numbers to check" not in page and "1 number to check" in page
