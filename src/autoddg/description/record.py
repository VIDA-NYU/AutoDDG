from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy


def _plain(value: Any) -> Any:
    """Copy of a value with numpy scalars and arrays turned into JSON types"""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, numpy.ndarray):
        return _plain(value.tolist())
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


@dataclass
class DatasetDescription:
    """
    Everything one description run produced: the description and the inputs it was built from

    Returned by ``AutoDDG.describe_dataframe``; can also be filled in by hand from the separate
    steps. Plain data, so it can be saved as JSON with ``to_dict`` and read back with
    ``from_dict``.

    Args:
        name: Dataset title
        description: Generated description
        original_description: Existing description, if any (it informs the topic)
        dataset_sample: CSV sample given to the model
        metadata: Profile metadata (``autoddg.profiling.profile_metadata``)
        profile_text: Profile summary given to the model
        semantic_profile: Semantic profile given to the model (coverage notes and column semantics)
        topic: Generated topic
        prompt: Prompt that produced the description
        search_description: Search-expanded description
    """

    name: str
    description: str
    original_description: str | None = None
    dataset_sample: str | None = None
    metadata: dict[str, Any] | None = None
    profile_text: str | None = None
    semantic_profile: str | None = None
    topic: str | None = None
    prompt: str | None = None
    search_description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-serializable dict"""
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetDescription:
        """A record from ``to_dict``'s output; unknown keys are ignored"""
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})
