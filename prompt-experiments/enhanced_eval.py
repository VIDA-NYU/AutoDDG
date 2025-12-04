"""
enhanced_eval.py
-----------------------------------------
Evaluation module for the Autoddg dataset description pipeline.

Includes:
    - BERTScore (reference-based)
    - ROUGE (reference-based)
    - Coverage Score (reference-free)
    - LLM-as-a-Judge (reference-free)
    - Unified evaluation function
    - CSV export utility
"""

import json
import re
import pandas as pd
from openai import OpenAI
from evaluate import load
from coverage import CoverageScorer
# ============================================
# INITIALIZATION
# ============================================

bertscore = load("bertscore")
rouge = load("rouge")   


# ============================================
# REFERENCE-BASED METRIC: ROUGE
# ============================================

def compute_rouge(reference: str, prediction: str):
    """
    Compute ROUGE scores using the 'evaluate' library.
    Returns ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum.
    """
    result = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )
    return {
        "rouge1": result.get("rouge1", 0.0),
        "rouge2": result.get("rouge2", 0.0),
        "rougeL": result.get("rougeL", 0.0),
        "rougeLsum": result.get("rougeLsum", 0.0),
    }


# ============================================
# REFERENCE-BASED METRIC: BERTScore
# ============================================

def compute_bertscore(reference: str, prediction: str):
    """Compute BERTScore precision/recall/F1."""
    result = bertscore.compute(
        predictions=[prediction],
        references=[reference],
        lang="en"
    )
    return {
        "bert_precision": result["precision"][0],
        "bert_recall": result["recall"][0],
        "bert_f1": result["f1"][0],
    }



# ============================================
# LLM-AS-A-JUDGE (REFERENCE-FREE)
# ============================================

LLM_JUDGE_PROMPT = """
You are an expert evaluator of dataset descriptions.

Rate the QUALITY of the following dataset description on a scale of 1–10
for each dimension below.

Dataset Description:
---
{description}
---

Rate the following:
1. Accuracy (does it correctly describe the dataset?)
2. Coverage (does it mention important dataset properties, metadata, variables, goals?)
3. Usefulness (does it help a researcher understand how to use the dataset?)
4. Limitations (does it describe constraints, caveats, biases, missingness?)

Respond ONLY as a JSON dictionary with fields:
{
  "accuracy": <number>,
  "coverage": <number>,
  "usefulness": <number>,
  "limitations": <number>,
  "overall_score": <number>
}
"""

def evaluate_llm_judge(description: str, client, model_name):
    """Use an LLM to provide a JSON evaluation of description quality."""
    prompt = LLM_JUDGE_PROMPT.format(description=description)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp.choices[0].message.content

    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        # fallback: no evaluation
        return {
            "accuracy": None,
            "coverage": None,
            "usefulness": None,
            "limitations": None,
            "overall_score": None
        }


import json

def extract_dataset_profile(description_text: str, client, model_name: str):
    """
    Extract a structured profile of a dataset description using your LLM client.
    
    This is REFERENCE-FREE — it analyzes ONLY the generated description.
    It outputs the exact JSON schema required by DatasetDescriptionCoverage().
    """

    prompt = f"""
You are a dataset documentation analysis assistant.

Extract the following fields FROM THE DESCRIPTION BELOW.
Only extract if explicitly present. DO NOT guess or hallucinate.
If not present, set to null.

Return ONLY valid JSON:

{{
  "basic_info": {{
    "dataset_name": null,
    "domain_or_field": null,
    "primary_purpose": null
  }},
  "data_characteristics": {{
    "size_or_scale": null,
    "data_format": null,
    "data_types": null,
    "temporal_coverage": null,
    "sample_unit": null
  }},
  "provenance": {{
    "collection_method": null,
    "data_source": null,
    "collection_date": null,
    "creators_or_curators": null,
    "preprocessing_steps": null
  }},
  "usage_context": {{
    "typical_applications": null,
    "research_questions_addressed": null,
    "how_used_in_paper": null,
    "benchmark_or_evaluation_role": null
  }},
  "quality_and_limitations": {{
    "known_limitations": null,
    "biases_or_caveats": null,
    "quality_issues": null,
    "challenges_in_use": null
  }}
}}

DESCRIPTION:
\"\"\"{description_text}\"\"\"

Return JSON ONLY. No extra text.
"""

    # -------------------------------
    # RUN THE MODEL USING YOUR CLIENT
    # -------------------------------
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content


    # -------------------------------
    # SAFE JSON PARSING
    # -------------------------------
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            cleaned = raw.strip().split("```")[-1]  # remove fences
            return json.loads(cleaned)
        except Exception:
            print("⚠ WARNING: Invalid JSON returned by extractor — returning empty schema.")
            return {
                "basic_info": {},
                "data_characteristics": {},
                "provenance": {},
                "usage_context": {},
                "quality_and_limitations": {}
            }


# ============================================
# UNIFIED EVALUATION
# ============================================
def evaluate_all(row, client,model_name):
    """
    Run all evaluation metrics for one dataset row from results.csv
    """

    dataset_id = row["Dataset_Name"]
    generated_desc = row["Description_Text"]
    reference_desc = row.get("Reference_Description", "")

    metrics = {}
    metrics["dataset_id"] = dataset_id

    # --------------------------
    # BERTScore (reference-based)
    # --------------------------
    metrics.update(compute_bertscore(reference_desc, generated_desc))

    # --------------------------
    # ROUGE (reference-based)
    # --------------------------
    metrics.update(compute_rouge(reference_desc, generated_desc))

    # --------------------------
    # Coverage score (reference-free)
    # --------------------------
    extraction_result = extract_dataset_profile(
        description_text=generated_desc,
        client=client,               
        model_name=model_name
    )
    coverage = CoverageScorer()
    coverage_results = coverage.calculate_coverage(extraction_result)

    metrics["coverage_overall"] = coverage_results["overall_score"]

    # dimension-level scores
    for dim, val in coverage_results["dimension_scores"].items():
        metrics[f"coverage_{dim}"] = val

    # --------------------------
    # LLM-as-a-Judge (reference-free)
    # --------------------------
    judge_scores = evaluate_llm_judge(generated_desc, client, model_name)
    for key, val in judge_scores.items():
        metrics[f"judge_{key}"] = val

    return metrics
