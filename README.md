<div align="center">
  <h1>AutoDDG</h1>
  <h3><i>Automated Dataset Description Generation using Large Language Models</i></h3>
  <h4><i>submitted to VLDB 2025</i></h4>
  <p>
    <a href="https://arxiv.org/abs/2502.01050">ArXiv Extended Paper Version</a>
  </p>
  <p>
    <img src="https://img.shields.io/static/v1?label=UV&message=compliant&color=2196F3&style=for-the-badge" alt="UV">
    <img src="https://img.shields.io/static/v1?label=RUFF&message=lint%2Fformat&color=9C27B0&style=for-the-badge&logo=ruff&logoColor=white" alt="Ruff">
    <img src="https://img.shields.io/badge/Black-formatted-000000?style=for-the-badge&logo=python&logoColor=white" alt="Black formatted">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python >= 3.10">
    <img src="https://img.shields.io/badge/OpenAI-Model-blue?style=for-the-badge&logo=openai" alt="OpenAI">
  </p>
</div>

---

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

> [!CAUTION]
> This installation method is temporary. A **PyPI release** of `AutoDDG` will soon be available. The `git+https` method will be deprecated in favor of the PyPI index.

---

## Getting Started

A very basic way to use `AutoDDG`:


## Getting Started

The simplest way to use AutoDDG is to create an instance and generate a dataset description:

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

prompt, description = autoddg.generate_description(dataset_sample=sample_csv)

print(description)
# >>> This dataset contains medical information about patients, including their unique Case_ID, Age, and Body Mass Index (BMI). etc.
```

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
