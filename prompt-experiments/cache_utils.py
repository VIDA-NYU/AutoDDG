import os
import pickle
import pandas as pd
from typing import Dict, Any, Optional
from autoddg.utils import get_sample


# --- CONFIGURATION & PATHS (Based on user-provided logic) ---
# NOTE: '__file__' is not defined in Jupyter/interactive environments, so we use
# os.getcwd() as a robust fallback to determine the current execution directory.
PROFILE_CACHE_DIR = 'profile_cache'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir))
ABSOLUTE_CACHE_DIR = os.path.join(script_dir, PROFILE_CACHE_DIR)

# Ensure the cache directory exists
os.makedirs(ABSOLUTE_CACHE_DIR, exist_ok=True)
print(f"[SETUP] Project Root: {PROJECT_ROOT}")
print(f"[SETUP] Cache Directory: {ABSOLUTE_CACHE_DIR}")


# --- MOCK DEFINITIONS FOR INDEPENDENCE ---
# You will replace these with your actual imports in your real project.

# class MockAutoDDG:
#     """Mock class simulating the profiling and analysis engine."""
#     def profile_dataframe(self, df):
#         print("  [DDG] Running basic and structural profiling.")
#         # Return mock profiles
#         return {"col_count": len(df.columns), "df_info": "mocked"}, {"row_count": len(df), "missing": 0}
        
#     def analyze_semantics(self, sample_df):
#         print("  [DDG] Running semantic analysis.")
#         return {"semantic_score": 0.85, "sample_analysis": "mocked"}
        
#     def generate_topic(self, dataset_name, data_file, dataset_sample):
#         print("  [DDG] Generating data topic.")
#         return f"Auto-Generated Topic: {dataset_name}"

# def get_sample(df, sample_size):
#     """Mock function to get a sample from the DataFrame."""
#     # This is a placeholder for your actual data sampling logic
#     print(f"  [HELPER] Getting a sample of size {sample_size}.")
#     return df.head(sample_size), ["Sample data point 1", "Sample data point 2"]

# --- STANDALONE CACHE UTILITY FUNCTIONS ---

def save_profile_to_cache(dataset_id: str, profile_data: Dict[str, Any]):
    """Saves the complete profiling results for a dataset to a pickle file."""
    cache_path = os.path.join(ABSOLUTE_CACHE_DIR, f'{dataset_id}_profiles.pkl')
    try:
        # Ensure the cache directory exists before writing
        os.makedirs(ABSOLUTE_CACHE_DIR, exist_ok=True)
        
        # Use 'wb' (write binary) mode for pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(profile_data, f)
        print(f"[CACHE] Saved profiles for {dataset_id} to {cache_path}")
    except Exception as e:
        print(f"ERROR: Could not save profile for {dataset_id}: {e}")

def load_profile_from_cache(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Loads the profiling results from cache, or returns None if not found."""
    cache_path = os.path.join(ABSOLUTE_CACHE_DIR, f'{dataset_id}_profiles.pkl')
    if os.path.exists(cache_path):
        try:
            # Use 'rb' (read binary) mode for pickle
            with open(cache_path, 'rb') as f:
                profile_data = pickle.load(f)
            print(f"[CACHE] Loaded profiles for {dataset_id} from cache.")
            return profile_data
        except Exception as e:
            print(f"ERROR: Could not load profile for {dataset_id} from cache: {e}")
            return None
    return None

def generate_and_cache_profiles(
    dataset_info: Dict[str, Any], 
    dataset_id: str,
    auto_ddg # The actual profiling engine instance
) -> Optional[Dict[str, Any]]:
    """Performs all profiling steps, saves them to cache, and returns the results."""
    dataset_name = dataset_info["dataset_name"]
    relative_data_file = dataset_info["dataset_path"]
    
    print(f"--- Loading Data and Running Core Profiling for {dataset_name} (Generating) ---")
    
    # Resolve the data file path relative to the PROJECT_ROOT
    # This assumes relative_data_file is relative to PROJECT_ROOT
    data_file = os.path.join(PROJECT_ROOT, relative_data_file)
    print(f"  Attempting to load data from: {data_file}")
    
    # --- Data Loading (NOTE: Removed mock file creation logic) ---
    try:
        # Load a manageable sample, using latin-1 for broader CSV compatibility
        # If the file does not exist, this will raise FileNotFoundError
        df = pd.read_csv(data_file, encoding='latin-1', low_memory=False) 
        
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {data_file}.")
        print("Please ensure your dataset_path in the dataset_info is correct relative to PROJECT_ROOT.")
        return None
    except Exception as e:
        print(f"FATAL ERROR: Data loading failed for {data_file} with error: {e}")
        return None

    # Get sample for semantic profiling and topic generation
    sample_df, dataset_sample = get_sample(df, sample_size=100)

    # 1. Profiling (Basic & Structural)
    basic_profile, structural_profile = auto_ddg.profile_dataframe(df)
    
    # 2. Semantic Analysis
    semantic_profile = auto_ddg.analyze_semantics(sample_df)
    
    # 3. Topic Generation
    data_topic = auto_ddg.generate_topic(dataset_name, None, dataset_sample)\
    
    profiles = {
        "basic_profile": basic_profile,
        "structural_profile": structural_profile,
        "semantic_profile": semantic_profile,
        "data_topic": data_topic,
        "dataset_sample": dataset_sample,
    }
    
    # Save to cache
    save_profile_to_cache(dataset_id, profiles)
    
    return profiles

def run_with_caching(dataset_id: str, dataset_info: Dict[str, Any], auto_ddg):
    """
    Demonstrates the core cache-first logic.
    
    Args:
        dataset_id: Unique ID for the dataset (used for caching).
        dataset_info: Dictionary containing "dataset_name" and "dataset_path".
        auto_ddg: An instance of the profiling engine (e.g., MockAutoDDG).
    """
    print(f"\n[RUNNER] Attempting to process {dataset_info['dataset_name']}...")
    
    # 1. Try to load from cache
    profiles = load_profile_from_cache(dataset_id)
    
    if profiles is None:
        # 2. If load fails, generate and cache
        profiles = generate_and_cache_profiles(dataset_info, dataset_id, auto_ddg)
        
    if profiles:
        print(f"[RUNNER] Successfully retrieved profiles for {dataset_info['dataset_name']}.")
        print(f"  Topic: {profiles['data_topic']}")
        print("-" * 50)
    else:
        print(f"[RUNNER] Failed to get profiles for {dataset_info['dataset_name']}.")
        print("-" * 50)