import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Loads the CSV data and preprocesses the Prompt_Type column.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please update the 'FILE_PATH' variable to point to your CSV file.")
        return None

    df = pd.read_csv(file_path)

    # Preprocessing: Treat empty Prompt_Type as 'Vanilla AutoDDG'
    df['Prompt_Group'] = df['Prompt_Type'].fillna('').replace('', 'Vanilla AutoDDG')

    # Convert score columns to numeric (errors='coerce' will turn non-numeric into NaN)
    score_columns = [
        'Completeness_Score', 'Conciseness_Score', 'Readability_Score',
        'bert_precision', 'bert_recall', 'bert_f1',
        'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
        'coverage_overall', 'coverage_basic_info', 'coverage_data_characteristics',
        'coverage_provenance', 'coverage_usage_context', 'coverage_quality_and_limitations'
    ]
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential scores might be missing
    df = df.dropna(subset=['Prompt_Group'] + score_columns, how='any')

    print(f"Loaded {len(df)} records for analysis.")
    print(f"Prompt groups found: {df['Prompt_Group'].unique()}")
    return df

def generate_violin_plots(df, columns, title, filename):
    """
    Generates a figure with violin plots for a list of score columns.
    The plot size has been increased for better readability.
    """
    num_cols = len(columns)
    
    # Calculate grid dimensions: aiming for roughly 2 columns if possible
    ncols = 2
    nrows = int(np.ceil(num_cols / ncols))

    # Increased figsize for taller, more readable plots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), constrained_layout=True)
    
    # Flatten axes array for easy iteration, even if it's a 1D array
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Data melting for Seaborn to handle multiple columns easily
        plot_df = df[['Prompt_Group', col]].rename(columns={col: 'Score'})
        
        # Create the violin plot
        sns.violinplot(
            data=plot_df, 
            x='Prompt_Group', 
            y='Score', 
            ax=ax, 
            inner='quartile', # Show quartiles (median and IQR)
            palette='Set3',
            cut=0 # Truncate violins at the minimum/maximum data points
        )

        ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Score Value', fontsize=12)
        
        # Rotate X labels and set alignment to right for better correspondence to the plot center
        ax.tick_params(axis='x', rotation=45, labelsize=10) 
        plt.setp(ax.get_xticklabels(), ha="right") # Explicitly set horizontal alignment
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated plot: {filename}")


# --- Main Execution Block ---

# TODO: IMPORTANT - Update this path to your actual CSV file
FILE_PATH = 'results-eval.csv' 

if __name__ == '__main__':
    data = load_and_preprocess_data(FILE_PATH)

    if data is not None:
        
        # Define score groups for plotting
        coverage_cols = [
            'coverage_overall', 'coverage_basic_info', 'coverage_data_characteristics',
            'coverage_provenance', 'coverage_usage_context', 'coverage_quality_and_limitations'
        ]
        
        bert_cols = ['bert_precision', 'bert_recall', 'bert_f1']
        
        rouge_cols = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

        # 1. Coverage Plots
        generate_violin_plots(
            data, 
            columns=coverage_cols, 
            title='Analysis of Coverage Scores by Prompt Type',
            filename='coverage_scores_violin_plots.png'
        )

        # 2. BERT Plots
        generate_violin_plots(
            data, 
            columns=bert_cols, 
            title='Analysis of BERT Similarity Scores by Prompt Type',
            filename='bert_scores_violin_plots.png'
        )

        # 3. ROUGE Plots
        generate_violin_plots(
            data, 
            columns=rouge_cols, 
            title='Analysis of ROUGE Metric Scores by Prompt Type',
            filename='rouge_scores_violin_plots.png'
        )

        print("\nAnalysis complete. Check the generated PNG files.")