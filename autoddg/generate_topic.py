class DatasetTopicGenerator:
    def __init__(self, client, model_name, temperature=0.0):
        """
        Initializes the DatasetTopicGenerator with the client and model parameters.

        :param client: OpenAI client for making requests.
        :param model_name: The model to use for generating the topic (default: gpt-3.5-turbo-0125).
        :param temperature: Temperature for controlling randomness in the generation (default: 0.0).
        """
        self.client = client
        self.model = model_name
        self.temperature = temperature
        print(f"Dataset Topic Generator initialized with model: {model_name}")

    def _generate_prompt(self, title, original_description, dataset_sample):
        """
        Generates the prompt for the model to generate a concise topic.

        :param title: Title of the dataset.
        :param original_description: Original description of the dataset.
        :param dataset_sample: Sample data for the dataset.
        :return: The generated prompt as a string.
        """

        prompt = "Using the dataset information provided, generate a concise topic in 2-3 words that best describes the dataset's primary theme:\n\n"
        if original_description:
            prompt += f"Title: {title}\nOriginal Description: {original_description}\n"
        else:
            prompt += f"Title: {title}\n"

        prompt += f"Dataset Sample: {dataset_sample}\n\nTopic (2-3 words):"

        # prompt = (f"Using the dataset information provided, generate a concise topic in 2-3 words that best "
        #           f"describes the dataset's primary theme:\n\n"
        #           f"Title: {title}\n"
        #           f"Original Description: {original_description}\n"
        #           f"Dataset Sample: {dataset_sample}\n\n"
        #           f"Topic (2-3 words):")

        return prompt

    def generate_topic(self, title, original_description, dataset_sample):
        """
        Generates a concise dataset topic using the provided title, original description, and data sample.

        :param title: Title of the dataset.
        :param original_description: Original description of the dataset.
        :param dataset_sample: Sample of the dataset.
        :return: Generated topic as a string.
        """
        # Create the prompt using the provided parameters
        prompt = self._generate_prompt(title, original_description, dataset_sample)

        # Make a request to the OpenAI API to generate the topic
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for generating concise dataset topics.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        # Extract the response content
        return response.choices[0].message.content.strip()
