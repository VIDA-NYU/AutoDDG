from datetime import datetime

from autoddg.generate_description import DatasetDescriptionGenerator


def log_print(text, symbol_begin="=", symbol_end=None, num=30):
    print(symbol_begin * num)
    if text is not None:
        print(text)
    if symbol_end:
        print(symbol_end * num)


def get_log_time():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_sample(data_pd, sample_size, random_state=9):
    if sample_size <= len(data_pd):
        data_sample = data_pd.sample(sample_size, random_state=random_state)
    else:
        data_sample = data_pd
    sample_csv = data_sample.to_csv(index=False)
    sample_df = data_sample
    return sample_df, sample_csv


def get_various_descriptions(
    temperatures,
    description_words,
    use_semantic_profile_choices,
    client,
    model_name,
    dataset_sample,
    dataset_profile,
    semantic_profile,
    data_topic,
    verbose=False,
):
    sub_data_descriptions = {}

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
                        "temperature-" + str(temperature),
                        "num_words-" + str(num_words),
                        "use_semantic_profile-" + str(use_semantic_profile),
                        "use_topic-" + str(use_topic),
                    ]
                )
                if verbose:
                    print("-" * 30)
                    print(
                        f"Genearting description for\
                    \n\ttemperature={temperature}\
                    \n\tnum_words={num_words}\
                    \n\tuse_semantic_profile={use_semantic_profile}\
                    \n\tuse_topic={use_topic}"
                    )

                # Generate description with dataset profile and semantic types
                description_generator = DatasetDescriptionGenerator(
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    description_words=num_words,
                )
                # if use_semantic_profile==False:
                #     previous_description = sub_data_descriptions["baseline"]["description"]
                # else:
                #     previous_description = sub_data_descriptions["temperature-"+str(temperature)+"+num_words-"+str(num_words)+"+use_semantic_profile-False+use_topic-False"]["description"]
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
