# --- Related Work Prompt Definitions ---

# V0: The original (less effective) prompt
PROMPT_V0_ORIGINAL = """
You are a **Dataset Description Synthesis Expert**. Your task is to extract and synthesize research context *specifically about the dataset* for a search engine description.

Extract key research context about the dataset: **{dataset_name}**.

**INSTRUCTIONS:**
Your summary MUST cover and integrate the following key research aspects:

1. **Research Domain and Applications:** What field or discipline uses this dataset, and what specific research questions or problems does it address?

2. **Dataset Usage and Findings:** How did researchers practically use this dataset (e.g., analyses, experiments, modeling), and what were the key results or findings derived from it?

3. **Characteristics and Provenance:** Describe how the data was collected or generated, any unique value it provides, and any notable preprocessing or curation steps mentioned.

4. **Limitations and Challenges:** Summarize any limitations, challenges, biases, or caveats researchers identified while using this data.

**OUTPUT FORMAT:** Synthesize all the extracted information into **one cohesive, natural-language paragraph** (approximately 300-400 words) that describes the research context of the dataset. **DO NOT** use bullet points, section headings (like "Title," "Abstract," "Results," etc.), or lists. The output must be ready to be inserted directly into the final dataset description.

**RESEARCH PAPER TEXT:**
{paper_text}
"""

# V1: The revised, more restrictive prompt (Recommended: 100-150 words)
PROMPT_V1_REVISED = """
You are a concise synthesis expert for a dataset search engine. Your ONLY goal is to extract factual context about the dataset's usage, findings, and limitations from the provided text and convert it into a single, cohesive, non-conversational paragraph.

Extract key research context about the dataset: **{dataset_name}**.

INSTRUCTIONS:
Your summary MUST cover and integrate the following key research aspects:
1. Research Domain and Applications.
2. Dataset Usage and Findings.
3. Characteristics and Provenance.
4. Limitations and Challenges.

OUTPUT FORMAT: Synthesize all the extracted information into **one cohesive, natural-language paragraph** (approximately 100-150 words). DO NOT use bullet points, section headings, or lists.

RESEARCH PAPER TEXT:
{paper_text}
"""
PROMPT_V2_HYBRID = """
You are an expert researcher writing an entry for a dataset search index. Your goal is to synthesize the most crucial, need-to-know information about the dataset: **{dataset_name}**, specifically for a researcher who is considering using it.

Synthesize the following information from the RESEARCH PAPER TEXT into a single, cohesive, non-conversational paragraph (STRICTLY 100-150 words):
1. The **specific research tasks** or domain for which the paper primarily used **{dataset_name}**.
2. The paper's **key findings or conclusions** regarding the dataset's utility, strengths, or performance.
3. Any **stated limitations, challenges, or characteristics** of the dataset's use in the paper.

OUTPUT CONSTRAINTS:
- **STRICTLY 100-150 words.**
- **DO NOT** use bullet points, numbered lists, section headings, or conversational phrasing.

RESEARCH PAPER TEXT:
{paper_text}
"""

PROMPT_STRUCTURED_V1 = """
    Extract information about the dataset "{dataset_name}" from this research paper.
   
    *CRITICAL*: Only extract information EXPLICITLY stated in the text.
    For any aspect not mentioned, write "Not mentioned in paper."
   
    Return a JSON object with this structure:
    {{
      "domain_and_applications": {{
        "field": "string or null",
        "research_questions": ["list of strings"],
        "confidence": "high/medium/low"
      }},
      "usage_in_study": {{
        "how_used": "string or null",
        "analyses_performed": ["list"],
        "key_findings": ["list"],
        "direct_quotes": ["relevant quotes with context"]
      }},
      "dataset_characteristics": {{
        "collection_method": "string or null",
        "unique_features": ["list"],
        "preprocessing": "string or null",
        "size_or_scope": "string or null"
      }},
      "limitations": {{
        "identified_limitations": ["list"],
        "challenges": ["list"],
        "biases_or_caveats": ["list"]
      }},
      "citation_context": ["Direct sentences mentioning the dataset"]
    }}
   
    *PAPER TEXT:*
    {paper_text}
"""

Research_focus_long = """You are helping researchers decide if "{dataset_name}" is right for their project.

Extract from this paper: What did this dataset enable the researchers to do?

**EXAMPLE:**
{{
  "enabled_research": "Large-scale analysis of gene expression patterns across tissue types",
  "practical_use": "Pre-training for downstream medical imaging tasks",
  "why_useful": "Only dataset with longitudinal patient data spanning 5+ years",
  "requirements_or_setup": "Requires institutional data use agreement; preprocessed with custom pipeline",
  "strengths_shown": "Rich annotations enabled fine-grained classification; diverse sample prevented overfitting",
  "limitations_found": "Small sample size limited statistical power; class imbalance required resampling",
  "usage_quotes": [
    "crucial for validating our hypothesis",
    "enabled comparison across multiple domains"
  ]
}}

**GOAL:** Help a researcher understand:
- What kinds of research is this dataset good for?
- What makes it useful/unique?
- What challenges might they face?

**TEXT:**
{paper_text}

**JSON:**
"""

Research_focus_long_v2= """You are an expert data science assistant helping researchers decide if the dataset "{dataset_name}" described in the TEXT is right for their project.

Your task is to **extract key structured information** about the dataset's utility, uniqueness, and challenges directly from the paper.

**EXAMPLE OF DESIRED OUTPUT:**
{{
  "enabled_research": "What **scientific question** or **scale of analysis** did the dataset uniquely permit? (e.g., Large-scale analysis of gene expression patterns across tissue types)",
  "practical_use": "What **common ML/Data task** (e.g., pre-training, fine-tuning, benchmarking) is it suitable for?",
  "why_useful": "What **unique feature** or data property (e.g., scale, annotations, time span, source) makes it valuable?",
  "requirements_or_setup": "What are the necessary **prerequisites** (e.g., hardware, license, custom code, data use agreement) for a user?",
  "strengths_shown": "What positive claims did the authors or users make about the dataset's qualities? (e.g., Rich annotations enabled fine-grained classification)",
  "limitations_found": "What weaknesses or challenges were identified? (e.g., Small sample size limited statistical power)",
  "usage_quotes": [
    "Extract 1-3 short, impactful quotes from the paper that describe the dataset's value or necessity."
  ]
}}

**INSTRUCTIONS & CONSTRAINTS:**
1. **Strictly** generate only the final JSON object, without any surrounding text or explanation.
2. The keys in the JSON output **must** exactly match the keys in the EXAMPLE.
3. If a specific piece of information is **not found** in the TEXT, the corresponding value must be an **empty string** (`""`) or an **empty list** (`[]`) for lists. **Do not guess or invent information.**

**TEXT:**
{paper_text}

**JSON:**
"""

Research_focus_short = """Extract how "{dataset_name}" was used to help researchers understand what this dataset is good for.

{{
  "what_it_enabled": "What research or analysis did this dataset make possible?",
  "why_chosen": "Why was this dataset suitable for their needs?",
  "how_used": "Specifically how did they use it (training/testing/validation/benchmark)?",
  "strengths": "What advantages or benefits did they mention?",
  "challenges": "Any limitations or difficulties they encountered?",
  "quotes": ["2-4 short quotes under 15 words each showing dataset usage"]
}}

**TEXT:**
{paper_text}

**JSON:**
"""

# Store prompts in a dictionary for easy iteration
ALL_RELATED_WORK_PROMPTS = {
    "V0_Original": PROMPT_V0_ORIGINAL,
    "V1_Revised": PROMPT_V1_REVISED,
    "V2_Hybrid": PROMPT_V2_HYBRID,
    "Structured_v1": PROMPT_STRUCTURED_V1,
    "Research_longv1": Research_focus_long,
    "Research_longv2":Research_focus_long_v2,
    "Research_shortv1": Research_focus_short,
}