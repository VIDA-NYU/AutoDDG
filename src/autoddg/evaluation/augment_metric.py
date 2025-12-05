from __future__ import annotations
from typing import Any, Dict
from beartype import beartype
from .base import BaseEvaluator
from ..utils import load_prompts


@beartype
class AugmentMetricEvaluator(BaseEvaluator):
    """
    Extends Baseevaluator and evaluates description for a series of metrics. 
    
    Reference free metrics: eval quality  
       Coverage score
       LLM as a judge for: relevance, hallunications
    
    """

    def __init__(self, client: Any, model_name: str):
        super().__init__(client=client, model_name=model_name)

        prompts = load_prompts()["multi_metric"]

        # Load the metric prompts from a config file
        self.metric_prompts = prompts["metrics"]
        self.system_message = prompts["system_message"]

    def evaluate(self, description: str) -> Dict[str, Any]:
        """
        Returns a dictionary of scores instead of a single raw string.
        """

        results = {}

        for metric_name, metric_prompt in self.metric_prompts.items():
            query = (
                f"{metric_prompt}\n"
                f"Description:\n{description}\n\n"
                "Your answer must be ONLY a number from 1–10."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
            )

            score = response.choices[0].message.content.strip()
            results[metric_name] = float(score)

        return results
