# utils.py
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# NOTE: The constants DATASET_NAME and RESULTS_FILE will be passed 
# or defined in the notebook, so they are not defined here.

def parse_scores(raw_score_text: str) -> dict:
    """Parses the 'Metric: Score' string output into a dictionary."""
    scores = {}
    lines = raw_score_text.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                # Store the key in a normalized way (lowercase, no spaces)
                scores[key.strip().lower()] = int(value.strip())
            except ValueError:
                # Handle cases where the value isn't an integer
                pass
    return scores

def log_result(
    prompt_name: str, 
    description_type: str, 
    description: str, 
    raw_scores: str, 
    dataset_name: str,       # Pass as argument now
    file_path: str,          # Pass as argument now
    related_profile: Optional[dict] = None
) -> None:
    """Logs the results of a single test run to a CSV file."""
    
    parsed_scores = parse_scores(raw_scores)
    
    # Extract the three core metrics for CSV columns
    completeness = parsed_scores.get('completeness', 0)
    conciseness = parsed_scores.get('conciseness', 0)
    readability = parsed_scores.get('readability', 0)
    
    # Serialize the related profile dict to a JSON string if it exists
    related_profile_json = json.dumps(related_profile) if related_profile else ""
    
    # Storing the full evaluation dictionary for detail
    raw_scores_json = json.dumps(parsed_scores) 
    
    new_row = {
        'Test_ID': f"{description_type}-{datetime.now().strftime('%H%M%S')}",
        'Dataset_Name': dataset_name, # Use passed argument
        'Prompt_Type': prompt_name,
        'Description_Source': description_type,
        'Description_Text': description.replace('\n', ' '), 
        'Related_Profile_JSON': related_profile_json,
        'Completeness_Score': completeness,
        'Conciseness_Score': conciseness,
        'Readability_Score': readability,
        'Raw_Scores_JSON': raw_scores_json, 
        'Evaluation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    df_new = pd.DataFrame([new_row])
    
    # Check if file exists to decide whether to write header
    header_needed = not os.path.exists(file_path)
    
    # Append to CSV
    df_new.to_csv(file_path, mode='a', header=header_needed, index=False)
    print(f"Logged {description_type} with Prompt {prompt_name} to {file_path}")

# =========================================================================
# 1. THE EXPERIMENT FUNCTION
# =========================================================================

def run_description_experiment(
    dataset_id: str,
    dataset_info: Dict[str, Any],
    auto_ddg: Any,             # auto_ddg instance
    PROJECT_ROOT: str,         # Absolute path to project root
    RESULTS_FILE: str,         # Absolute path to results CSV
    PROMPTS_TO_TEST: Dict[str, str], # Dictionary of prompts for augmented test
    load_profile_from_cache: Any, # Function to load profiles
    log_result: Any            # Function to log results
) -> None:
    """
    Runs baseline and augmented description generation and evaluation for a 
    single dataset, using profiles loaded from cache.
    """
    
    print(f"\n{'='*70}\n✨ Starting Experiment for Dataset: ID='{dataset_id}', Name='{dataset_info['dataset_name']}'")

    # --- A. Setup and Profile Unpacking ---
    try:
        profiles = load_profile_from_cache(dataset_id=dataset_id)
        if profiles is None:
            # This check is technically redundant if called correctly in the loop, 
            # but serves as a fail-safe.
            print(f"Skipping: Cache not found for ID {dataset_id}.")
            return

        DATASET_NAME = dataset_info['dataset_name'] 
        PAPER_FILE_RELATIVE = dataset_info['related_paper_path'] 
        PAPER_FILE = os.path.join(PROJECT_ROOT, PAPER_FILE_RELATIVE) 

        # Unpack profiles
        basic_profile = profiles["basic_profile"]
        semantic_profile = profiles["semantic_profile"]
        data_topic = profiles["data_topic"]
        dataset_sample = profiles["dataset_sample"] 

        print(f"Setup complete. PDF: {PAPER_FILE}")
        
    except KeyError as e:
        print(f"FATAL ERROR: Missing key in dataset_info or profile for ID {dataset_id}: {e}")
        return
    except Exception as e:
        print(f"FATAL ERROR during setup for ID {dataset_id}: {e}")
        return

    # -----------------------------------------------------------------
    ## 1. Baseline (Vanilla AutoDDG) Description
    # -----------------------------------------------------------------

    print("\n--- Running Baseline (Vanilla) Description ---")

    prompt_baseline, description_baseline = auto_ddg.describe_dataset(
        dataset_sample=dataset_sample,
        dataset_profile=basic_profile,
        use_profile=True,
        semantic_profile=semantic_profile,
        use_semantic_profile=True,
        data_topic=data_topic,
        use_topic=True,
        use_related_profile=False
    )

    baseline_scores = auto_ddg.evaluate_description(description_baseline)
    print(f"Baseline Scores: {baseline_scores}")

    log_result(
        prompt_name="N/A", 
        description_type="Vanilla_AutoDDG", 
        description=description_baseline, 
        raw_scores=baseline_scores,
        dataset_name=DATASET_NAME,
        file_path=RESULTS_FILE
    )

    print("-" * 50)

    # -----------------------------------------------------------------
    ## 2. Augmented (AutoDDG + Related Work) Description
    # -----------------------------------------------------------------

    print("\n--- Running Augmented (AutoDDG + Related Work) Descriptions ---")

    for prompt_name, extraction_prompt in PROMPTS_TO_TEST.items():
        print(f"\n-> Augmented Test with Prompt: {prompt_name}")
        
        # Step A: Analyze related work using the current prompt
        related_profile = auto_ddg.analyze_related(
            pdf_path=PAPER_FILE,
            dataset_name=DATASET_NAME,
            extraction_prompt=extraction_prompt,
            max_pages=10
        )
        print(f"Related Work Summary: {related_profile['summary'][:150]}...")

        # Step B: Generate description with the new related profile
        prompt_augmented, description_augmented = auto_ddg.describe_dataset(
            dataset_sample=dataset_sample,
            dataset_profile=basic_profile,
            use_profile=True,
            semantic_profile=semantic_profile,
            use_semantic_profile=True,
            data_topic=data_topic,
            use_topic=True,
            related_profile=related_profile,
            use_related_profile=True # Augmented
        )
        
        # Step C: Evaluate and Log
        augmented_scores = auto_ddg.evaluate_description(description_augmented)
        print(f"Augmented Scores ({prompt_name}): {augmented_scores}")
        
        log_result(
            prompt_name=prompt_name, 
            description_type="Augmented_AutoDDG", 
            description=description_augmented, 
            raw_scores=augmented_scores,
            dataset_name=DATASET_NAME,
            file_path=RESULTS_FILE,
            related_profile=related_profile
        )
        
    print(f"\n✅ Experiment complete for ID {dataset_id}.")
    print(f"{'='*70}\n")