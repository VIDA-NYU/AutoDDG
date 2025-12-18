# 🏷️ AutoDDG: Automated Dataset Description Generation using Large Language Models

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2502.01050"><img src="https://img.shields.io/badge/arXiv-2502.01050-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <img src="https://img.shields.io/static/v1?label=UV&message=compliant&color=2196F3&style=flat-square" alt="UV">
    <img src="https://img.shields.io/static/v1?label=RUFF&message=lint%2Fformat&color=9C27B0&style=flat-square&logo=ruff&logoColor=white" alt="Ruff">
    <img src="https://img.shields.io/badge/Black-formatted-000000?style=flat-square&logo=python&logoColor=white" alt="Black formatted">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10">
    <img src="https://img.shields.io/badge/OpenAI-Model-blue?style=flat-square&logo=openai" alt="OpenAI">
    <img src="https://img.shields.io/badge/Local-LLM-green?style=flat-square&logo=huggingface" alt="Local LLM">
  </p>
</div>

## Overview

This repository contains the official implementation for the **SIGMOD 2026** paper:

> **AutoDDG: Automated Dataset Description Generation using Large Language Models**

AutoDDG is an automated system for generating comprehensive, accurate, readable, and concise dataset descriptions. The framework combines a data-driven approach to summarize dataset contents with large language models (LLMs) to enrich summaries with semantic information and produce human-readable descriptions. AutoDDG supports both API-based (OpenAI) and local LLM (transformers) modes, providing flexibility for different deployment scenarios.

## Installation

Clone the repository and install dependencies via [uv (recommended)](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/VIDA-NYU/AutoDDG.git
cd AutoDDG
uv sync
# If you do not have uv installed:
# * `curl -LsSf https://astral.sh/uv/install.sh | sh`
# * or look at https://docs.astral.sh/uv/getting-started/installation/
```

Then launch Jupyter Lab to explore:

```bash
uv run --with jupyter jupyter lab
```

Alternatively, install directly via pip:

```bash
pip install git+https://github.com/VIDA-NYU/AutoDDG@main
```

**For local LLM support** (Qwen, Llama, etc.), install with optional dependencies:

```bash
pip install git+https://github.com/VIDA-NYU/AutoDDG@main[local-llm]
# or
pip install git+https://github.com/VIDA-NYU/AutoDDG@main transformers torch
```

> [!CAUTION]
> This installation method is temporary. A **PyPI release** of `AutoDDG` will soon be available. The `git+https` method will be deprecated in favor of the PyPI index.

---

## Getting Started

AutoDDG supports both **API-based** (OpenAI) and **local LLM** (transformers) modes.

### Using OpenAI API

The simplest way to use AutoDDG is with an OpenAI API client:

```python
from openai import OpenAI
from autoddg import AutoDDG

# Setup OpenAI client
client = OpenAI(api_key="sk-...")

# Initialize AutoDDG
autoddg = AutoDDG(client=client, model_name="gpt-4o-mini")

# Generate description from a small CSV sample
sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)

print(description)
# >>> This dataset contains medical information about patients, including their unique Case_ID, Age, and Body Mass Index (BMI). etc.
```

### Using Local LLM

AutoDDG also supports local LLMs via transformers (Qwen, Llama, etc.):

```python
from autoddg import AutoDDG

# Initialize AutoDDG with local LLM
autoddg = AutoDDG(
    client=None,
    model_name="Qwen/Qwen2.5-7B-Instruct",  # or any HuggingFace model
    use_local_llm=True,
    local_llm_device="cuda",  # or "cpu" if no GPU
    local_llm_dtype="bfloat16",  # or "float16", "float32"
)

# Generate description
sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)
print(description)
```

**Note:** For local LLM support, ensure you have installed the optional dependencies:
```bash
pip install transformers torch
```

### Semantic Analysis Processing Modes

AutoDDG provides multiple processing modes for semantic analysis to optimize performance based on your use case:

| Mode | OpenAI API | Local LLM | Description |
|------|------------|-----------|-------------|
| **Sequential** | ✅ | ✅ | Default mode, processes columns one by one |
| **Multi-threading** | ✅ | ❌ | Concurrent processing for faster execution |
| **Group-prompting** | ✅ | ✅ | Processes multiple columns in one prompt |
| **Batch processing** | ❌ | ✅ | Efficient GPU utilization for local models |

#### Sequential Mode (Default)

The default mode processes columns sequentially. Works with both API and local LLMs:

```python
# Sequential mode (default)
semantic_profile = autoddg.analyze_semantics(dataframe)
```

#### Multi-threading Mode (OpenAI API Only)

Use multi-threading to process columns in parallel for faster execution. **Only available for OpenAI API clients:**

```python
# Multi-threading mode (OpenAI API only)
semantic_profile = autoddg.analyze_semantics(
    dataframe,
    use_multi_threading=True,
    max_workers=32,  # Optional: number of parallel workers
)
```

#### Group-prompting Mode (Both API and Local LLM)

Process multiple columns in a single prompt to reduce API calls. Efficient for both API and local LLMs:

```python
# Group-prompting: process all columns at once
semantic_profile = autoddg.analyze_semantics(
    dataframe,
    use_group_prompting=True,
    group_size=0,  # 0 = all columns at once, >0 = group size
)

# Or process in groups of 5 columns
semantic_profile = autoddg.analyze_semantics(
    dataframe,
    use_group_prompting=True,
    group_size=5,
)
```

#### Batch Processing Mode (Local LLM Only)

For local LLMs, use batch processing for efficient GPU utilization. **Only available for local LLMs:**

```python
# Batch processing mode (Local LLM only)
semantic_profile = autoddg.analyze_semantics(
    dataframe,
    use_batch_processing=True,
    batch_size=32,  # Number of columns to process per batch
)
```

**Important Notes:**
- Multi-threading is only available for OpenAI API clients
- Batch processing is only available for local LLMs
- Group-prompting works with both API and local LLMs
- Batch processing takes precedence over other modes if enabled

### Quick Jupyter Notebook Start

For a much better introduction, we **highly recommend** starting with the [quick_start notebook with an example dataset](./examples/quick_start.ipynb).

---

## How to Cite

If you use `AutoDDG` in your research, please cite our work:

```bibtex
@misc{2502.01050,
Author = {Haoxiang Zhang and Yurong Liu and Wei-Lun Hung and Aécio Santos and Juliana Freire},
Title = {AutoDDG: Automated Dataset Description Generation using Large Language Models},
Year = {2025},
Eprint = {arXiv:2502.01050},
}
```

---

## License

`AutoDDG` is released under the [Apache License 2.0](./LICENSE).
