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

# Store prompts in a dictionary for easy iteration
ALL_RELATED_WORK_PROMPTS = {
    "V0_Original": PROMPT_V0_ORIGINAL,
    "V1_Revised": PROMPT_V1_REVISED,
    "V2_Hybrid": PROMPT_V2_HYBRID,
    "Structured_v1": PROMPT_STRUCTURED_V1
}