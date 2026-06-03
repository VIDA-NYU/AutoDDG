"""Run AutoDDG through the NYU Portkey AI gateway instead of OpenAI directly.

AutoDDG accepts any OpenAI-compatible client. Portkey's client exposes the same
``chat.completions.create(...)`` interface, so we can hand a Portkey instance to
AutoDDG unchanged.

Usage:
    # PORTKEY_API_KEY is read from a local .env file (git-ignored) or the
    # environment, so a plain run is enough:
    uv run python examples/portkey_quickstart.py
"""

import os

from dotenv import load_dotenv
from portkey_ai import Portkey

from autoddg import AutoDDG

# Load PORTKEY_API_KEY from a local .env file (git-ignored, never committed).
load_dotenv()

# Portkey gateway acts as a drop-in, OpenAI-compatible client.
client = Portkey(
    base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
    api_key=os.environ["PORTKEY_API_KEY"],  # from .env, never hard-coded
)

autoddg = AutoDDG(
    client=client,
    model_name="@vertexai/anthropic.claude-sonnet-4-6",
    description_words=100,
)

sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)
print(description)
