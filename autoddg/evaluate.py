import openai

# from openai import OpenAI


class Prompt:
    def __init__(self):
        self.EVALUATE_SYSETM = """
            You are a helpful and precise assistant for checking the quality of the dataset description.
            """
        self.EVALUATE_PROMPT = """
            You will be given one tabular dataset description. Your task is to rate the description on 3 metrics.
            Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            1. Completeness (1-10) - Evaluates how thoroughly the dataset description covers essential aspects such as the scope of data, query workloads, summary statistics, and possible tasks or applications. 
            A high score indicates that the description provides a comprehensive overview, including details on dataset size, structure, fields, and potential use cases.    
            2. Conciseness (1-10) - Measures the efficiency of the dataset description in conveying necessary information without redundancy. 
            A high score indicates that the description is succinct, avoiding unnecessary details while employing semantic types (e.g., categories, entities) to streamline communication.
            3. Readability (1-10) -  Evaluates the logical flow and readability of the dataset description. 
            A high score suggests that the description progresses logically from one section to the next, creating a coherent and integrated narrative that facilitates understanding of the dataset.

            Evaluation Steps:
            Read the dataset description carefully and identify the main topic and key points. Assign a score for each criteria on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the Evaluation Criteria.

            Example 1:
            Description: The dataset provides information on alcohol-impaired driving deaths and occupant deaths across various states in the United States. It includes data for 51 states, detailing the number of alcohol-impaired driving deaths and occupant deaths, with values ranging from 0 to 3723 and 0 to 10406, respectively. Each entry also contains the state abbreviation and its geographical coordinates. The dataset is structured with categorical and numerical data types, focusing on traffic safety and casualty statistics. Key attributes include state names, death counts, and location coordinates, making it a valuable resource for analyzing traffic safety trends and issues related to impaired driving.
            Evaluation Form (scores ONLY): Completeness: 7, Conciseness: 9, Readability: 9

            Example 2:
            Description: The dataset provides a comprehensive overview of traffic safety statistics across various states in the United States, specifically focusing on alcohol-impaired driving deaths and occupant deaths. It includes data from 51 unique states, represented by their two-letter postal abbreviations, such as MA (Massachusetts), SD (South Dakota), AK (Alaska), MS (Mississippi), and ME (Maine). Each entry in the dataset captures critical information regarding the number of alcohol-impaired driving deaths and the total occupant deaths resulting from traffic incidents.
            The column "Alcohol-Impaired Driving Deaths" is represented as an integer, indicating the number of fatalities attributed to alcohol impairment while driving. The dataset reveals a range of values, with the highest recorded number being 2367 deaths in Mississippi, highlighting the severity of the issue in certain regions. In contrast, states like Alaska report significantly lower figures, with only 205 alcohol-impaired driving deaths.
            The "Occupant Deaths" column also consists of integer values, representing the total number of deaths among vehicle occupants, regardless of the cause. This data spans from 0 to 10406, with Mississippi again showing the highest number of occupant deaths at 6100, which raises concerns about overall traffic safety in the state.
            Additionally, the dataset includes a "Location" column that provides geographical coordinates for each state, enhancing the spatial understanding of the data. The coordinates are formatted as latitude and longitude pairs, allowing for potential mapping and geographical analysis of traffic safety trends.
            Overall, this dataset serves as a valuable resource for researchers, policymakers, and public safety advocates aiming to understand and address the impact of alcohol on driving safety across different states. It highlights the need for targeted interventions and policies to reduce alcohol-impaired driving incidents and improve occupant safety on the roads.
            Evaluation Form (scores ONLY): Completeness: 8, Conciseness: 7, Readability: 8

            Please provide scores for the given dataset description based on the Evaluation Criteria. Do not include any additional information or comments in your response.
            Evaluation Form (scores ONLY):
            """


class GPTEvaluator:
    def __init__(self, gpt4_api_key=""):
        # openai.api_key = GPT4_API_KEY
        self.client = openai.OpenAI(
            api_key=gpt4_api_key,
        )
        self.model = "gpt-4o"
        self.prompt = Prompt()

    def evaluate(self, description):
        content = (
            self.prompt.EVALUATE_PROMPT
            + "Description: "
            + description
            + "\n"
            + "Evaluation Form (scores ONLY): "
        )
        return self.generate(content)

    def generate(self, content):
        evaluation = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt.EVALUATE_SYSETM},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        score = evaluation.choices[0].message.content
        return score


class LLaMAEvaluator:
    def __init__(self, llama_api_key=""):
        self.client = openai.OpenAI(
            api_key=llama_api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.prompt = Prompt()

    def evaluate(self, description):
        content = (
            self.prompt.EVALUATE_PROMPT
            + "Description: "
            + description
            + "\n"
            + "Evaluation Form (scores ONLY): "
        )
        return self.generate(content)

    def generate(self, content):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        score = chat_completion.choices[0].message.content
        return score
