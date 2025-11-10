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

Via [uv (recommended)](https://docs.astral.sh/uv/):

```bash
uv add autoddg
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
pip install autoddg
```

---

## Getting Started

A very basic way to use `AutoDDG`:

```python
from autoddg import AutoDDG

# Create an AutoDDG pipeline and bind it to OpenAI
autoddg = AutoDDG(description_words=100).with_provider(
    provider="openai",
    model_name="gpt-4o-mini",
    # api_key="sk-...",  # Optional if OPENAI_API_KEY is exported i.e. $> export OPENAI_API_KEY="sk-..."
)

# Generate description from a small CSV sample
sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)

print(description)
# >>> This dataset contains medical information about patients, including their unique Case_ID, Age, and Body Mass Index (BMI).
```
<details>
<summary><strong>📣 Wants To Try Other LLMs (Plug-and-Play)? 👇</strong></summary>

AutoDDG speaks OpenAI-compatible APIs via <a href="https://github.com/BerriAI/litellm">LiteLLM</a>.

**Built-in providers**

| Provider    | Env var(s)                               | Aliases |
|-------------|-------------------------------------------|---------|
| OpenAI      | `OPENAI_API_KEY`                          | `oai`   |
| Anthropic   | `ANTHROPIC_API_KEY`                       | `claude`|
| Mistral     | `MISTRAL_API_KEY`, `MISTRAL_API_TOKEN`    | —       |
| Grok (xAI)  | `XAI_API_KEY`, `GROK_API_KEY`             | `xai`   |

---

### 1) Discover providers & models
```python
from autoddg import AutoDDG

print(AutoDDG.list_providers())                 # ('anthropic','grok','mistral','openai',...)
print(AutoDDG.list_model_names("openai")[:8])   # sample of known model ids
print(AutoDDG.describe_provider("grok"))        # env vars, base_url, aliases, options
```

---

### 2) Switch provider (one line)
```python
from autoddg import AutoDDG

autoddg = AutoDDG(description_words=100).with_provider(
    provider="anthropic",            # alias "claude" also works
    model_name="claude-3-sonnet-20241022",
    # api_key="...",                # optional if ANTHROPIC_API_KEY is set in env
)
```

Use a proxy/custom gateway:
```python
autoddg = AutoDDG().with_provider(
    provider="openai",
    model_name="gpt-4o-mini",
    api_key="sk-...",                               # overrides env
    factory_options={"base_url": "https://my-proxy.example/v1"},
)
```

---

### 3) Add a new provider

**A. Via config (maintainers / forks) —— Happy to receive Pull Requests!**
Start to edit: `src/autoddg/utils/provider_defaults.yaml`
```yaml
providers:
  local-gateway:
    api_key_env: [LOCAL_GATEWAY_KEY]
    base_url: https://llm.example.com/v1
    aliases: [lgw]
    # extra_options:
    #   custom_llm_provider: openai
```

**B. Register at runtime (no file edits)**
```python
from autoddg import AutoDDG
from autoddg.utils import LLMClientFactory

factory = LLMClientFactory()
factory.register_provider(
    "local-gateway",
    api_key_env=["LOCAL_GATEWAY_KEY"],
    base_url="https://llm.example.com/v1",
    aliases=["lgw"],
)

autoddg = AutoDDG().with_provider(
    provider="local-gateway",
    model_name="llama-3.1-70b",   # whatever your endpoint serves
    factory=factory,
)
```

— You can use provider aliases (`oai`, `claude`, `xai`) anywhere a provider is accepted.
— Prefer `AutoDDG.list_model_names("<provider>")` to pick exact model strings.

</details>

### Quick Jupyter Notebook Start

For a much better introduction, we **highly recommend** starting with the [quick_start notebook with an example dataset](./examples/quick_start.ipynb).

If you want to explore different LLM providers interactively, check out the [multi-provider playground notebook](./examples/provider_playground.ipynb). It walks through configuring API keys, using the shared `AutoDDG` helpers, and comparing outputs from providers such as OpenAI, Anthropic, Mistral, and Grok.

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
