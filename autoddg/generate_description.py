import json
import re


class DatasetDescriptionGenerator:
    def __init__(self, client, model_name, temperature=0.0, description_words=100):
        """
        Initializes the DatasetDescriptionGenerator with the OpenAI client and model parameters.

        :param client: OpenAI client for making requests.
        :param model: The model to use for generating the description (default: gpt-3.5-turbo-0125).
        :param temperature: Temperature for controlling randomness in the generation (default: 0.3).
        :param description_words: Target number of words for the description (default: 100).
        """
        self.client = client  # Use the client instance
        self.model = model_name
        self.temperature = temperature
        self.description_words = description_words
        print(
            f"Dataset Description Generator initialized with model: {model_name}, temperature: {temperature}, description words: {description_words}"
        )

    def _generate_prompt(
        self,
        dataset_sample,
        dataset_profile=None,
        use_profile=False,
        semantic_profile=None,
        use_semantic_profile=False,
        data_topic=None,
        use_topic=False,
    ):
        """
        Generates the prompt for the OpenAI model based on the provided inputs.

        :param dataset_sample: Sample of the dataset to include in the description.
        :param dataset_profile: Optional dataset profile to include in the description.
        :param use_profile: Boolean flag to include the dataset profile.
        :param semantic_profile: Optional semantic types to include in the description.
        :param use_semantic_profile: Boolean flag to include semantic types.
        :param instruction: Additional instruction for generating the description.
        :return: The generated prompt as a string.
        """

        prompt = "Answer the question using the following information.\n"

        # Add dataset sample
        prompt += f"First, consider the dataset sample:\n\n{dataset_sample}\n"

        # Add dataset profile
        if use_profile and dataset_profile:
            prompt += (
                f"Additionally, the dataset profile is as follows:\n\n{dataset_profile}\n\n"
                f"Based on this profile, please add sentence(s) to enrich the dataset description.\n\n"
            )

        # Add semantic types if the flag is set to True
        if use_semantic_profile and semantic_profile:
            prompt += (
                f"Furthermore, the semantic profile of the dataset columns is as follows:\n{semantic_profile}\n\n"
                "Based on this information, please add sentence(s) discussing the semantic profile in the description.\n\n"
            )

        # Add data topic if the flag is set to True
        if use_topic and data_topic:
            prompt += (
                f"Moreover, the dataset topic is: {data_topic}. "
                f"Based on this topic, please add sentence(s) describing what this dataset can be used for.\n\n"
            )

        prompt += (
            "Question: Based on the information above and the requirements, provide a dataset description in sentences. "
            "Use only natural, readable sentences without special formatting. "
            "\nAnswer:"
        )

        return prompt

    def generate_description(
        self,
        dataset_sample,
        dataset_profile=None,
        use_profile=False,
        semantic_profile=None,
        use_semantic_profile=False,
        data_topic=None,
        use_topic=False,
    ):
        """
        Generates a dataset description using the provided dataset sample, profile, and semantic types.

        :param dataset_sample: Sample of the dataset to include in the description.
        :param dataset_profile: Optional dataset profile.
        :param use_profile: Boolean flag to include the dataset profile in the description.
        :param semantic_profile: Optional semantic types for dataset columns.
        :param use_semantic_profile: Boolean flag to include semantic types in the description.
        :param instruction: Additional instruction for the description generation.
        :return: Generated description as a string.
        """
        # Create the prompt using the provided parameters
        prompt = self._generate_prompt(
            dataset_sample,
            dataset_profile,
            use_profile,
            semantic_profile,
            use_semantic_profile,
            data_topic,
            use_topic,
        )

        # Make a request to the OpenAI API to generate the description
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for a dataset search engine. Your goal is to improve the readability of dataset description for dataset search engine users.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        # Extract the response content
        description = response.choices[0].message.content
        return prompt, description


class SearchFocusedDescription:
    def __init__(self, client, model_name="gpt-4o-mini"):
        self.client = client
        self.model = model_name
        print(f"Search Focused Description initialized with model: {model_name}")

    def expand_description(self, initial_description, topic):
        template = """
        Dataset Overview:
        - Please keep the exact initial description of the dataset as shown in beginning the prompt.

        Key Themes or Topics:
        - Central focus on a broad area of interest (e.g., urban planning, socio-economic factors, environmental analysis).
        - Data spans multiple subtopics or related areas that contribute to a holistic understanding of the primary theme.
        Example:
        - theme1/topic1
        - theme2/topic2
        - theme3/topic3

        Applications and Use Cases:
        - Facilitates analysis for professionals, policymakers, researchers, or stakeholders.
        - Useful for specific applications, such as planning, engineering, policy formulation, or statistical modeling.
        - Enables insights into patterns, trends, and relationships relevant to the domain.
        Example:
        - application1/usecase1
        - application2/usecase2
        - application3/usecase3

        Concepts and Synonyms:
        - Includes related concepts, terms, and variations to ensure comprehensive coverage of the topic.
        - Synonyms and alternative phrases improve searchability and retrieval effectiveness.
        Example:
        - concept1/synonym1
        - concept2/synonym2
        - concept3/synonym3

        Keywords and Themes:
        - Lists relevant keywords and themes for indexing, categorization, and enhancing discoverability.
        - Keywords reflect the dataset's content, scope, and relevance to the domain.
        Example:
        - keyword1
        - keyword2
        - keyword3

        Additional Context:
        - Highlights the dataset's relevance to specific challenges or questions in the domain.
        - May emphasize its value for interdisciplinary applications or integration with related datasets.
        Example:
        - context1
        - context2
        - context3
        """

        prompt = f"""
        You are given a dataset about the topic {topic}, with the following initial description:\n\n{initial_description}.

        Please expand the description by including the exact topic. Additionally, add as many related concepts, synonyms, and relevant terms 
        as possible based on the initial description and the topic.

        Unlike the initial description, which is focused on presentation and readability, the expanded description is intended to be indexed 
        at backend of a dataset search engine to improve searchability.

        Therefore, focus less on readability and more on including all relevant terms related to the topic. Make sure to include any variations 
        of the key terms and concepts that could help improve retrieval in search results.

        Please follow the structure of following example template:
        {template}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for a dataset search engine. Your goal is to improve the performance of the dataset search engine for keyword queries.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        expanded_description = response.choices[0].message.content
        return prompt, expanded_description


class SemanticProfiler:
    TEMPLATE = """
    {'Temporal': 
        {
            'isTemporal': Does this column contain temporal information? Yes or No,
            'resolution': If Yes, specify the resolution (Year, Month, Day, Hour, etc.).
        },
     'Spatial': {'isSpatial': Does this column contain spatial information? Yes or No,
                 'resolution': If Yes, specify the resolution (Country, State, City, Coordinates, etc.).},
     'Entity Type': What kind of entity does the column describe? (e.g., Person, Location, Organization, Product),
     'Domain-Specific Types': What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic),
     'Function/Usage Context': How might the data be used (e.g., Aggregation Key, Ranking/Scoring, Interaction Data, Measurement).}
    """

    RESPONSE_EXAMPLE = """
    {
    "Domain-Specific Types": "General",
    "Entity Type": "Temporal Entity",
    "Function/Usage Context": "Aggregation Key",
    "Spatial": {"isSpatial": false,
                "resolution": ""},
    "Temporal": {"isTemporal": true,
                "resolution": "Year"}
    }
    """

    def __init__(self, client, model_name="gpt-4o-mini"):
        self.client = client  # Use the client instance
        self.model = model_name
        print(f"Semantic Type Analyzer initialized with model: {model_name}")

    def _fix_json_response(self, response_text):
        """
        Automatically close all open braces by counting mismatched '{' and '}' in the response text.

        :param response_text: The response text to fix.
        :return: The fixed response text.
        """
        response_text = re.search(r"\{.*\}", response_text, re.DOTALL).group()

        # Append the required number of closing braces
        open_braces = response_text.count("{")
        close_braces = response_text.count("}")
        response_text += "}" * (open_braces - close_braces)

        # Use regex to remove any trailing comma before the final closing brace
        response_text = re.sub(r",\s*}", "}", response_text)
        return response_text

    def get_semantic_type(self, column_name, sample_values):
        prompt = f"""
        You are a dataset semantic analyzer. Based on the column name and sample values, classify the column into multiple semantic types. 
        Please group the semantic types under the following categories: 
        'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'. 
        Following is the template {self.TEMPLATE}
        Please follow these rules:
        1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {self.RESPONSE_EXAMPLE}
        2. All keys from the template must be present in the response.
        3. All keys and string values must be enclosed in double quotes.
        4. There must be no trailing commas.
        5. Use booleans (true/false) and numbers without quotes.
        6. Do not include any additional information or context in the response.
        7. If you are unsure about a specific category, you can leave it as an empty string.

        Column name: {column_name}
        Sample values: {sample_values}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant skilled in dataset semantic analysis.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message.content

        response_text = self._fix_json_response(response_text)

        try:
            semantic_dict = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Failed to parse GPT response as JSON for column: {column_name}")
            print(f"Response text: {response_text}")
            semantic_dict = None
        return semantic_dict

    def analyze_dataframe(self, dataframe):
        """
        Analyzes a pandas DataFrame and returns semantic types for each column.

        :param dataframe: pandas DataFrame to be analyzed.
        :return: Dictionary of semantic types for each column.
        """

        def _get_sample(data_pd, sample_size):
            if sample_size < len(data_pd):
                data_sample = data_pd.sample(sample_size, random_state=9)
            else:
                data_sample = data_pd
            return data_sample

        semantic_summary = []
        dataframe_sample = _get_sample(dataframe, 5)

        # Iterate through the columns to profile each one
        for column in dataframe.columns:
            try:
                # Get the first few values as a sample to provide context
                sample_values = dataframe_sample[column].astype(str).tolist()

                # Call GPT to get the semantic type
                semantic_description = None
                retry_count = 0
                while semantic_description is None and retry_count < 3:
                    if retry_count > 0:
                        print(f"Retrying for column: {column}")
                    semantic_description = self.get_semantic_type(column, sample_values)
                    retry_count += 1
                if retry_count == 3:
                    print(f"Failed to get semantic type for column: {column}")
                    continue
                # print(column, semantic_description)

                # Create a human-readable summary for the column
                column_summary = f"**{column}**: "
                entity_type = semantic_description.get("Entity Type", "Unknown")
                if entity_type != "" and entity_type != "Unknown":
                    column_summary += f"Represents {entity_type.lower()}. "

                # Handle spatial and temporal cases
                isTemporal = semantic_description["Temporal"].get("isTemporal", False)
                if isTemporal and semantic_description["Temporal"]["isTemporal"]:
                    column_summary += f"Contains temporal data (resolution: {semantic_description['Temporal']['resolution']}). "
                isSpatial = semantic_description["Spatial"].get("isSpatial", False)
                if isSpatial and semantic_description["Spatial"]["isSpatial"]:
                    column_summary += f"Contains spatial data (resolution: {semantic_description['Spatial']['resolution']}). "

                domain_type = semantic_description.get(
                    "Domain-Specific Types", "Unknown"
                )
                if domain_type != "" and domain_type != "Unknown":
                    column_summary += f"Domain-specific type: {domain_type.lower()}. "

                function_context = semantic_description.get(
                    "Function/Usage Context", "Unknown"
                )
                if function_context != "" and function_context != "Unknown":
                    column_summary += (
                        f"Function/Usage context: {function_context.lower()}. "
                    )

                # Add sample values
                # if sample_values:
                #     sample_str = ', '.join(sample_values)
                #     column_summary += f"Sample values include {sample_str}."

                # Append the summary for this column
                semantic_summary.append(column_summary)
            except Exception:
                continue

        # Join the semantic summary into a readable format
        final_summary = (
            "The key semantic information for this dataset includes:\n"
            + "\n".join(semantic_summary)
        )
        return final_summary
