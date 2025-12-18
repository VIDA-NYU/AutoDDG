#!/usr/bin/env python3
"""
Quick test script for PairwiseEvaluator

Usage:
    python test_pairwise.py
"""

from autoddg import GPT4oMiniPairwiseEvaluator, PairwiseEvaluator

# Sample descriptions for testing
DESC1 = """This dataset contains medical information about patients, including their unique Case_ID, Age, and Body Mass Index (BMI). 
The dataset includes data for multiple patients with ages ranging from 30 to 72 years and BMI values from 22.8 to 34.15. 
This dataset can be used for medical research and health analysis."""

DESC2 = """The dataset provides comprehensive patient medical records with three key attributes: Case_ID (unique identifier), 
Age (ranging from 30 to 72 years), and BMI (Body Mass Index, values from 22.8 to 34.15). 
The dataset is structured for medical research applications, enabling analysis of patient demographics and health metrics."""

DESC3 = """Medical dataset with patient identifiers, age, and body mass index. Contains demographic and health information."""


def test_compare():
    """Test basic pairwise comparison"""
    print("=" * 60)
    print("Test 1: Basic Pairwise Comparison")
    print("=" * 60)
    
    # Note: Replace with your actual API key for real testing
    api_key = "YOUR_API_KEY_HERE"
    
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️  Skipping API call test - please set your API key")
        print("   Would compare:")
        print(f"   Description A: {DESC1[:50]}...")
        print(f"   Description B: {DESC2[:50]}...")
        return
    
    try:
        evaluator = GPT4oMiniPairwiseEvaluator(gpt4_api_key=api_key)
        result = evaluator.compare(DESC1, DESC2)
        print(f"✅ Comparison result: {result}")
        print(f"   {'Description A' if result == 'A' else 'Description B'} is better")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_compare_batch():
    """Test batch comparison"""
    print("\n" + "=" * 60)
    print("Test 2: Batch Comparison")
    print("=" * 60)
    
    api_key = "YOUR_API_KEY_HERE"
    
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️  Skipping API call test - please set your API key")
        print("   Would compare 3 descriptions with strategy='no_repeat', max_pairs=3")
        return
    
    try:
        evaluator = GPT4oMiniPairwiseEvaluator(gpt4_api_key=api_key)
        descriptions = [DESC1, DESC2, DESC3]
        
        print(f"Comparing {len(descriptions)} descriptions...")
        results = evaluator.compare_batch(
            descriptions, 
            strategy="no_repeat", 
            max_pairs=3
        )
        
        print(f"✅ Batch comparison completed")
        print(f"   Number of comparisons: {len(results['overall'])}")
        for i, (idx_i, idx_j, winner) in enumerate(results['overall'], 1):
            print(f"   Comparison {i}: Description {idx_i} vs {idx_j} -> Winner: {winner}")
        
        # Test win rates
        print("\n   Computing win rates...")
        win_rates = PairwiseEvaluator.compute_win_rates(results, len(descriptions))
        for aspect, rates in win_rates.items():
            print(f"   {aspect}: {rates}")
        
        # Test ELO ratings
        print("\n   Computing ELO ratings...")
        elo_ratings = PairwiseEvaluator.compute_elo_ratings(results, len(descriptions))
        for aspect, ratings in elo_ratings.items():
            print(f"   {aspect}: {[f'{r:.1f}' for r in ratings]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_sampling_strategies():
    """Test different sampling strategies"""
    print("\n" + "=" * 60)
    print("Test 3: Sampling Strategies")
    print("=" * 60)
    
    n = 5
    strategies = ["full", "no_repeat", "symmetric"]
    
    for strategy in strategies:
        pairs = PairwiseEvaluator.sample_pairs(n, strategy=strategy, max_pairs=10)
        print(f"✅ Strategy '{strategy}': {len(pairs)} pairs")
        print(f"   Sample pairs: {pairs[:5]}{'...' if len(pairs) > 5 else ''}")


def test_static_methods():
    """Test static computation methods"""
    print("\n" + "=" * 60)
    print("Test 4: Static Computation Methods")
    print("=" * 60)
    
    # Mock results: 3 descriptions, 4 comparisons
    # Description 0 wins 2, Description 1 wins 1, Description 2 wins 1
    mock_results = {
        "overall": [
            (0, 1, "A"),  # 0 beats 1
            (0, 2, "A"),  # 0 beats 2
            (1, 2, "A"),  # 1 beats 2
            (0, 1, "B"),  # 1 beats 0 (rematch)
        ]
    }
    
    n_desc = 3
    
    print("Mock results:")
    for i, (idx_i, idx_j, winner) in enumerate(mock_results["overall"], 1):
        print(f"  {i}. Description {idx_i} vs {idx_j} -> Winner: {winner}")
    
    # Test win rates
    win_rates = PairwiseEvaluator.compute_win_rates(mock_results, n_desc)
    print(f"\n✅ Win rates:")
    for aspect, rates in win_rates.items():
        print(f"   {aspect}: {[f'{r:.2f}' for r in rates]}")
        print(f"   (Description 0: {rates[0]:.2%}, Description 1: {rates[1]:.2%}, Description 2: {rates[2]:.2%})")
    
    # Test ELO ratings
    elo_ratings = PairwiseEvaluator.compute_elo_ratings(mock_results, n_desc)
    print(f"\n✅ ELO ratings:")
    for aspect, ratings in elo_ratings.items():
        print(f"   {aspect}: {[f'{r:.1f}' for r in ratings]}")
        print(f"   (Description 0: {ratings[0]:.1f}, Description 1: {ratings[1]:.1f}, Description 2: {ratings[2]:.1f})")


def main():
    """Run all tests"""
    print("PairwiseEvaluator Test Suite")
    print("=" * 60)
    
    # Test static methods (no API needed)
    test_sampling_strategies()
    test_static_methods()
    
    # Test API calls (requires API key)
    test_compare()
    test_compare_batch()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    print("\nNote: API tests require a valid OpenAI API key.")
    print("      Set 'api_key' variable in the script to run API tests.")


if __name__ == "__main__":
    main()

