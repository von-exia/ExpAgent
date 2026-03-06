"""
Test script for GAIA evaluator
"""
from evaluation.eval_gaia import GAIAEvaluator
from agent_interface import AgentInterface

def main():
    # Initialize evaluator
    agent_interface = AgentInterface(planner_type="dag_react")
    evaluator = GAIAEvaluator(agent_interface=agent_interface, cache_dir="./hf_cache")
    
    # Setup dataset (using a small sample for testing)
    evaluator.setup_dataset(level="2023_level1")
    
    # Run a small evaluation (first 3 examples for quick test)
    print("Running test evaluation on first 3 examples...")
    # Limit dataset to first 3 for testing
    original_dataset = evaluator.dataset
    evaluator.dataset = original_dataset.select(range(min(100, len(original_dataset))))
    
    results = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results("./results/test_gaia_evaluation.json")


if __name__ == "__main__":
    main()