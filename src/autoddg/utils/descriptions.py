from __future__ import annotations

from typing import Any, Dict, Iterable

from beartype import beartype

from .logging import log_print


@beartype
def get_various_descriptions(
    temperatures: Iterable[float],
    description_words: Iterable[int],
    use_semantic_profile_choices: Iterable[bool],
    client: Any,
    model_name: str,
    dataset_sample: str,
    dataset_profile: str | None,
    semantic_profile: str | None,
    data_topic: str | None,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Generate descriptions across temperature length and context settings for a dataset

    Args:
        temperatures: List of temperatures
        description_words: List of target word counts
        use_semantic_profile_choices: Whether to include semantic profile
        client: OpenAI-compatible client
        model_name: Model identifier
        dataset_sample: CSV text sample
        dataset_profile: Structural profile
        semantic_profile: Semantic profile
        data_topic: Short topic string
        verbose: Print progress if True

    Returns:
        Mapping config_key -> {"prompt": str, "description": str}
    """
    from autoddg.description import DatasetDescriptionGenerator

    sub_data_descriptions: Dict[str, Dict[str, str]] = {}

    for temperature in temperatures:
        for num_words in description_words:
            description_generator = DatasetDescriptionGenerator(
                client=client,
                model_name=model_name,
                temperature=temperature,
                description_words=num_words,
            )
            prompt, description = description_generator.generate_description(
                dataset_sample=dataset_sample
            )
            sub_data_descriptions["baseline"] = {
                "prompt": prompt,
                "description": description,
            }

            for use_semantic_profile in use_semantic_profile_choices:
                use_topic = use_semantic_profile
                log_key = "+".join(
                    [
                        f"temperature-{temperature}",
                        f"num_words-{num_words}",
                        f"use_semantic_profile-{use_semantic_profile}",
                        f"use_topic-{use_topic}",
                    ]
                )
                if verbose:
                    log_print(
                        (
                            "Generating description for\n"
                            f"\ttemperature={temperature}\n"
                            f"\tnum_words={num_words}\n"
                            f"\tuse_semantic_profile={use_semantic_profile}\n"
                            f"\tuse_topic={use_topic}"
                        ),
                        symbol_begin="-",
                    )

                description_generator = DatasetDescriptionGenerator(
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    description_words=num_words,
                )
                prompt, description = description_generator.generate_description(
                    dataset_sample=dataset_sample,
                    dataset_profile=dataset_profile,
                    use_profile=True,
                    semantic_profile=semantic_profile,
                    use_semantic_profile=use_semantic_profile,
                    data_topic=data_topic,
                    use_topic=use_topic,
                )

                sub_data_descriptions[log_key] = {
                    "prompt": prompt,
                    "description": description,
                }

    return sub_data_descriptions
