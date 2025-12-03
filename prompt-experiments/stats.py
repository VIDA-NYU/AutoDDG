import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def generate_three_way_stats(results_file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads the results CSV and calculates mean, std, and count for 
    Vanilla_AutoDDG, V1_Revised, and V2_Hybrid across all scores.

    Args:
        results_file_path: Absolute path to the autoddg_experiment_results.csv file.

    Returns:
        A pandas DataFrame containing the comparison, or None if the file is not found.
    """
    
    # 1. Load and Prepare the data
    try:
        df = pd.read_csv(results_file_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return None

    score_cols = ['Completeness_Score', 'Conciseness_Score', 'Readability_Score']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Define Groups
    # Create a column to uniquely identify the three methods:
    # - Vanilla is always 'Vanilla_AutoDDG'
    # - Augmented versions are distinguished by their Prompt_Type
    
    df['Comparison_Group'] = df['Description_Source']
    df.loc[df['Prompt_Type'] == 'V1_Revised', 'Comparison_Group'] = 'V1_Revised'
    df.loc[df['Prompt_Type'] == 'V2_Hybrid', 'Comparison_Group'] = 'V2_Hybrid'

    # Filter down to only the three groups we care about
    target_groups = ['Vanilla_AutoDDG', 'V1_Revised', 'V2_Hybrid']
    df_filtered = df[df['Comparison_Group'].isin(target_groups)].copy()

    # 3. Calculate Statistics (Mean, Std, Count)
    
    # Calculate Mean
    mean_stats = df_filtered.groupby('Comparison_Group')[score_cols].mean().T.rename(
        columns=lambda x: f'{x}_Mean'
    )
    
    # Calculate Standard Deviation
    std_stats = df_filtered.groupby('Comparison_Group')[score_cols].std().T.rename(
        columns=lambda x: f'{x}_Std'
    )

    # Calculate Count
    count_stats = df_filtered.groupby('Comparison_Group')[score_cols].count().T.iloc[:, 0].rename('Count')
    
    # 4. Combine Results
    
    # Combine Mean and Std into a single DataFrame
    combined_stats = pd.concat([mean_stats, std_stats], axis=1)
    
    # Add the Count row (reshaping required for concatenation alignment)
    count_df = pd.DataFrame(count_stats).T
    count_df.columns = [f'{col}_Count' for col in count_df.columns]
    
    # Final cleanup and ordering
    final_comparison = pd.concat([combined_stats, count_df], axis=0)
    
    # Select and order the columns for better readability (Vanilla, V1, V2)
    col_order = [
        'Vanilla_AutoDDG_Mean', 'V1_Revised_Mean', 'V2_Hybrid_Mean',
        'Vanilla_AutoDDG_Std', 'V1_Revised_Std', 'V2_Hybrid_Std',
        'Vanilla_AutoDDG_Count', 'V1_Revised_Count', 'V2_Hybrid_Count',
    ]
    
    # Ensure all required columns exist before reordering
    final_comparison = final_comparison.reindex(columns=col_order, fill_value=np.nan)

    return final_comparison

# =========================================================================
# 5. EXECUTION BLOCK
# =========================================================================

# Ensure your RESULTS_FILE variable is correctly defined (absolute path)
RESULTS_FILE = '/home/bia/Documents/AutoDDG-Enhanced/prompt-experiments/results.csv' # Placeholder

print("Calculating Three-Way Comparison: Vanilla vs. V1_Revised vs. V2_Hybrid...\n")
comparison_results = generate_three_way_stats(RESULTS_FILE)

if comparison_results is not None:
    print("## 📊 Score Comparison: Vanilla vs. Augmented Versions")
    print("This table compares the mean, standard deviation, and count of scores for the three methods.")
    print(comparison_results.to_markdown(floatfmt=".2f"))