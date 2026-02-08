"""
Test script for GAIA evaluator
"""
from evaluation.eval_gaia import GAIAEvaluator
from agent_interface import AgentInterface

def main():
    # Initialize evaluator
    evaluator = GAIAEvaluator(model_name="reactree_agent", cache_dir="./hf_cache")
    
    # Setup dataset (using a small sample for testing)
    evaluator.setup_dataset(level="2023_level1")
    
    # Create agent interface
    agent_interface = AgentInterface(planner_type="reactree")
    
    # Run a small evaluation (first 3 examples for quick test)
    print("Running test evaluation on first 3 examples...")
    # Limit dataset to first 3 for testing
    original_dataset = evaluator.dataset
    evaluator.dataset = original_dataset.select(range(min(3, len(original_dataset))))
    
    results = evaluator.run_evaluation(agent_interface=agent_interface)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results("./results/test_gaia_evaluation.json")


if __name__ == "__main__":
    main()