"""Description records, number checks and the review page"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from autoddg import AutoDDG, DatasetDescription, DescriptionReport, check_description, check_numbers
from autoddg.llm import LLMClient

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


def test_check_description_uses_the_record():
    record = DatasetDescription(
        name="2018_squirrel_census",
        description="3,619 sightings in 2018, more than the 3,000 of 2012.",
        original_description="Census of 2012.",
        metadata=METADATA,
    )
    assert [(c.text, c.status) for c in check_description(record)] == [
        ("3,619", "match"),
        ("2018", "match"),  # dataset name
        ("3,000", "missing"),
        ("2012", "match"),  # original description
    ]
    assert [c.text for c in check_description(record, "Over 13 columns.")] == ["13"]


def test_record_round_trips_through_json():
    record = DatasetDescription(
        name="d",
        description="text",
        metadata={
            "nb_rows": np.int64(3),
            "columns": [{"mean": np.float64(1.5), "ok": np.bool_(1)}],
        },
    )
    data = json.loads(json.dumps(record.to_dict()))
    assert data["metadata"] == {"nb_rows": 3, "columns": [{"mean": 1.5, "ok": True}]}
    assert DatasetDescription.from_dict({**data, "unknown": 1}) == DatasetDescription(**data)


def test_page_marks_numbers_and_escapes_text():
    record = DatasetDescription(
        name="fountains <b>",
        description="It lists 3,619 fountains and 99 benches.",
        original_description="Fountains & more",
        metadata=METADATA,
    )
    page = DescriptionReport(
        [record], title="Review", links={"fountains <b>": "reports/fountains.html"}
    ).to_html()
    assert '<mark class="match" title="row count: 3,619">3,619</mark>' in page
    assert '<mark class="missing"' in page and ">99</mark>" in page
    assert "fountains &lt;b&gt;" in page and "Fountains &amp; more" in page
    assert 'href="reports/fountains.html"' in page
    assert "1 numbers to check" not in page and "1 number to check" in page


class ScriptedClient(LLMClient):
    """Answers each step of the pipeline by its prompt"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_completions_create(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any
    ) -> dict[str, Any]:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Column:" in prompt:  # group semantic profiling
            reply = json.dumps({"rent": {"Entity Type": "rent amount"}, "area": {}})
        elif "initial description" in prompt:
            reply = "SEARCH VERSION"
        elif "Title:" in prompt:
            reply = "Rent Trends"
        else:
            reply = "Median rent in 4 areas."
        return {"choices": [{"message": {"role": "assistant", "content": reply}}]}


def test_describe_dataframe_keeps_every_step():
    client = ScriptedClient()
    frame = pd.DataFrame({"area": ["a", "b", "c", "d"], "rent": [1200, 1500, 1800, 3500]})
    record = AutoDDG(client=client, model_name="m").describe_dataframe(
        frame, "Median rent", "Rents by area.", semantic_options={"use_group_prompting": True}
    )
    assert record.name == "Median rent" and record.original_description == "Rents by area."
    assert record.description == "Median rent in 4 areas."
    assert record.topic == "Rent Trends" and record.search_description == "SEARCH VERSION"
    assert record.metadata["nb_rows"] == 4
    assert record.metadata["columns"][1]["maximum"] == 3500
    assert "Minimum 1200, maximum 3500." in record.profile_text
    assert "**rent**: Represents rent amount." in record.semantic_profile
    assert record.prompt in client.prompts and "Rent Trends" in record.prompt
    assert len(client.prompts) == 4  # topic, semantics, description, search
    assert check_description(record)[0].status == "match"  # 4 rows


# Cases found in urbantrace's existing descriptions
COLUMNS = {
    "nb_rows": 28983,
    "columns": [
        {"name": "BoroCD", "minimum": 101.0, "maximum": 595.0, "num_distinct_values": 71},
        {"name": "CounDist", "minimum": 1.0, "maximum": 51.0, "num_distinct_values": 51},
        {"name": "codes", "num_distinct_values": 427},
        {"name": "income", "minimum": 17210.0, "maximum": 206489.5},
        {"name": "ela_a", "mean": 0.3},
        {"name": "ela_b", "mean": 0.531},
        {
            "name": "bikeid",
            "minimum": 0.0,
            "maximum": 52000.0,  # the cluster below ends at 1,981, the column does not
            "coverage": [{"range": {"gte": 0.0, "lte": 1981.0}}],
        },
        {
            "name": "desdate",
            "earliest": -129081600.0,
            "latest": 1683590400.0,
        },  # 1965-11-29 .. 2023-05-09
        {"name": "created_at", "earliest": 1589412490.0, "latest": 1589412490.0},  # 2020-05-13
    ],
}


def check(text, **kwargs):
    return [(c.text, c.status) for c in check_numbers(text, COLUMNS, **kwargs)]


def test_ranges_must_be_a_columns_extremes():
    assert check("BoroCD spans 0 to 503, CounDist 1 to 51.") == [
        ("0", "missing"),
        ("503", "missing"),
        ("1", "match"),
        ("51", "match"),
    ]
    assert check("Values span 101–595.") == [("101", "match"), ("595", "match")]
    assert check("between 1 and 51") == [("1", "match"), ("51", "match")]
    source = check_numbers("from 0 to 503", COLUMNS)[0].source
    assert "no column has this range" in source


def test_year_ranges_must_be_a_date_columns_years():
    assert check("designation dates from 1966 to 2022", sample="1966-01-01,2022-02-02") == [
        ("1966", "missing"),
        ("2022", "missing"),
    ]
    assert check("designated from 1965 to 2023") == [("1965", "match"), ("2023", "match")]
    named = check("yearly values (2005–2023)", references={"names": "2005 2006 2023"})
    assert named == [("2005", "match"), ("2023", "match")]


def test_bounds():
    assert check("over 400 violation codes, over $200,000 at most") == [
        ("400", "match"),
        ("200,000", "match"),
    ]
    assert check("over 600 codes") == [("600", "missing")]  # the max of 595 is not over 600
    assert check("BoroCD reaches over about 600") == [("600", "close")]  # hedged, column named
    assert check("over about 600 codes") == [("600", "missing")]  # BoroCD is not this sentence's


def test_percentages_can_be_stored_as_fractions():
    assert check("from roughly 30% in 2013 to about 53% in 2024") == [
        ("30%", "match"),
        ("2013", "match"),
        ("53%", "match"),
        ("2024", "missing"),  # after the dates
    ]


def test_value_clusters_do_not_support_numbers():
    assert check("about 2,000 segments") == [("2,000", "missing")]


def test_close_matches_need_the_column_in_the_sentence():
    assert check("about 52,000 segments.") == [("52,000", "match")]  # largest value: exact
    assert check("roughly 425 types.") == [("425", "missing")]  # 427 distinct codes, unnamed
    assert check("roughly 425 codes.") == [("425", "close")]
    assert check("Codes number 427. About 425 types.") == [("427", "match"), ("425", "missing")]


def test_written_dates():
    snapshot = {"columns": [COLUMNS["columns"][-1]]}  # created_at only
    snapshot_checks = check_numbers("a single snapshot date (May 14, 2020)", snapshot)
    assert [(c.text, c.status) for c in snapshot_checks] == [("May 14, 2020", "missing")]
    assert "created_at: 2020-05-13 to 2020-05-13" in snapshot_checks[0].source
    assert check_numbers("on May 13, 2020", snapshot)[0].status == "match"
    assert check("dates from 1965‑11‑29 to 2023‑05‑09") == [  # non-breaking hyphens
        ("1965‑11‑29", "match"),
        ("2023‑05‑09", "match"),
    ]
    assert check("dates from 1965-11-29 to 2023-06-01") == [
        ("1965-11-29", "missing"),
        ("2023-06-01", "missing"),
    ]
    assert check("placeholder values such as 9999-09-09", sample="x,9999-09-09\n") == [
        ("9999-09-09", "match")
    ]
    assert check("designated in June 1990") == [("June 1990", "match")]
