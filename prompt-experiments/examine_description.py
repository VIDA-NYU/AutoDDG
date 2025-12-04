import pandas as pd
import json
from io import StringIO
from typing import Dict, Any, List

# --- MOCK DATA SETUP ---
# Replace this entire block with code to load your actual CSV file:
df = pd.read_csv('results.csv')
df = df.fillna('') # Replace NaN with empty string for cleaner handling

# --- ANALYSIS FUNCTIONS ---

def format_json_string(json_str: str) -> str:
    """Pretty-prints a JSON string or returns an empty message if invalid."""
    if not json_str:
        return "--- No Related Profile Data ---"
    try:
        data = json.loads(json_str)
        # Use json.dumps with indent=2 for readability
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return f"--- Error parsing JSON profile: {json_str[:50]}... ---"

def analyze_dataset_descriptions(df: pd.DataFrame) -> None:
    """
    Analyzes and prints the comparison of Vanilla vs. Augmented descriptions.
    """
    unique_datasets = df['Dataset_Name'].unique()

    print("=" * 80)
    print("DATASET DESCRIPTION COMPARISON REPORT")
    print("=" * 80)

    for dataset_name in unique_datasets:
        print("\n" + "=" * 40)
        print(f"DATASET: {dataset_name}")
        print("=" * 40)

        # 1. Get Vanilla AutoDDG Entry
        vanilla_entries = df[
            (df['Dataset_Name'] == dataset_name) & 
            (df['Description_Source'] == 'Vanilla_AutoDDG')
        ]
        
        if not vanilla_entries.empty:
            vanilla_desc = vanilla_entries.iloc[0]['Description_Text']
            print("\n--- BASELINE (VANILLA) DESCRIPTION ---")
            print(vanilla_desc)
        else:
            print("\n--- BASELINE (VANILLA) DESCRIPTION NOT FOUND ---")


        # 2. Get Augmented AutoDDG Entries
        augmented_entries = df[
            (df['Dataset_Name'] == dataset_name) & 
            (df['Description_Source'] == 'Augmented_AutoDDG')
        ].to_dict('records') # Convert to list of dictionaries for easy iteration

        if augmented_entries:
            print("\n--- AUGMENTED DESCRIPTIONS AND RELATED PROFILES ---")
            for i, entry in enumerate(augmented_entries):
                print("-" * 30)
                print(f"({i+1}) Prompt Type: {entry['Prompt_Type']} (Test ID: {entry['Test_ID']})")
                print("-" * 30)
                
                # Print Augmented Description
                print("\n[AUGMENTED DESCRIPTION]")
                print(entry['Description_Text'])

                # Print Related Profile JSON
                print("\n[RELATED PROFILE JSON]")
                # Use the helper function to safely format the JSON
                formatted_profile = format_json_string(entry['Related_Profile_JSON'])
                print(formatted_profile)
                
        else:
            print("\n--- NO AUGMENTED DESCRIPTIONS FOUND ---")
            
        print("\n" + "#" * 80 + "\n")


# --- RUN ANALYSIS ---
analyze_dataset_descriptions(df)