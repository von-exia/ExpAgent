import os
import json
with open("./api_keys.json", "r") as f:
    api_keys = json.load(f)
if api_keys.get("HF_ENDPOINT", False):
    os.environ["HF_ENDPOINT"] = api_keys["HF_ENDPOINT"]
if api_keys.get("HF_TOKEN", False):
    os.environ["HF_TOKEN"] = api_keys["HF_TOKEN"]

import re
import time
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download, login
from typing import Dict, List, Any, Optional
import argparse


class GAIAEvaluator:
    """
    GAIA Benchmark Evaluator
    Evaluates AI models on GAIA benchmark tasks
    """
    
    def __init__(self, agent_interface=None, cache_dir: str = "./hf_cache"):
        """
        Initialize GAIA evaluator
        
        Args:
            agent_interface: Interface to the agent being evaluated
            cache_dir: Directory to cache datasets and models
        """
        self.model_name = agent_interface.model_name
        self.agent_interface = agent_interface
        self.cache_dir = cache_dir
        self.data_dir = None
        self.dataset = None
        self.results = []
        
        # Set environment variables
        os.environ["HF_HOME"] = self.cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def setup_dataset(self, repo_id: str = "gaia-benchmark/GAIA", level: str = "2023_level1"):
        """
        Download and setup GAIA dataset
        
        Args:
            repo_id: HuggingFace dataset repository ID
            level: GAIA level to evaluate (e.g., "2023_level1", "2023_level2", "2023_level3")
        """
        print(f"Downloading GAIA dataset from {repo_id}...")
        self.data_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=self.cache_dir)
        self.dataset = load_dataset(self.data_dir, level, split="validation")
        print(f"Dataset loaded with {len(self.dataset)} examples")
    
    def get_required_tools(self, example: Dict[str, Any]) -> List[str]:
        """
        Extract required tools from example metadata
        
        Args:
            example: GAIA dataset example
            
        Returns:
            List of required tools
        """
        res = example.get('Annotator Metadata', {}).get("Tools", "")
        pattern = r'\d+\.\s*(.*)'
        tool_names = re.findall(pattern, res)
        return [tool.strip() for tool in tool_names if tool.strip()]

    
    def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Run full evaluation on the dataset
        
        Args:
            model_interface: Interface to the model being evaluated
            
        Returns:
            List of evaluation results
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call setup_dataset() first.")
        
        print(f"Starting evaluation on {len(self.dataset)} examples...")
        
        for i, example in enumerate(self.dataset):
            print(f"Evaluating example {i+1}/{len(self.dataset)}...")
            question = example["Question"]
            final_answer = example.get("Final answer", "")
            print(f"Question: {question}")
            print(f"Expected Answer: {final_answer}")
            file_path = os.path.join(self.data_dir, example["file_path"]) if example["file_path"] else ""
            print(f"File Path: {file_path}, file name: {example['file_name']}")
            if file_path:
                question += f"\nPlease refer to the file at {file_path} for necessary information to answer the question."
            
            # Get model prediction (this would call your model)
            if self.agent_interface:
                model_response, is_correct = self.agent_interface.judge(question, final_answer, file_path)
            else:
                # Placeholder for model prediction
                model_response = f"PLACEHOLDER_PREDICTION_FOR_TASK_{example['task_id']}"
                is_correct = False

            # Evaluate the prediction
            task_id = example.get("task_id", "")
            required_tools = self.get_required_tools(example)
            result = {
                "task_id": task_id,
                "question": question,
                "model_response": model_response,
                "expected_answer": final_answer,
                "is_correct": is_correct,
                "required_tools": required_tools,
                "level": example.get("Level", ""),
                "file_path": example.get("file_path", ""),
                "steps": example.get('Annotator Metadata', {}).get('Steps', ''),
                "timestamp": time.time()
            }
            self.results.append(result)
            # print("Agent Reponse:", result['model_response'])
            # print("Ground Truth:", result["expected_answer"])
            
            # Print progress
            status = "CORRECT" if result["is_correct"] else "INCORRECT"
            print(f"  Status: {status}")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall evaluation metrics

        Returns:
            Dictionary containing evaluation metrics
        """
        return self.calculate_detailed_metrics()

    def calculate_detailed_metrics(self) -> Dict[str, Any]:
        """
        Calculate detailed evaluation metrics

        Returns:
            Dictionary containing detailed evaluation metrics
        """
        if not self.results:
            return {}

        total_examples = len(self.results)
        correct_predictions = sum(1 for r in self.results if r["is_correct"])
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0

        # Calculate tool usage statistics
        all_tools = []
        correct_tool_usage = 0
        for result in self.results:
            all_tools.extend(result["required_tools"])
            # Count if the model used the correct tools for correct predictions
            if result["is_correct"]:
                correct_tool_usage += 1

        unique_tools = list(set(all_tools))

        # Calculate accuracy by level
        levels = {}
        for result in self.results:
            level = result.get("level", "unknown")
            if level not in levels:
                levels[level] = {"total": 0, "correct": 0}
            levels[level]["total"] += 1
            if result["is_correct"]:
                levels[level]["correct"] += 1

        # Calculate level-wise accuracy
        level_accuracies = {}
        for level, counts in levels.items():
            level_accuracies[level] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

        # Calculate tool effectiveness
        tool_effectiveness = {}
        for result in self.results:
            for tool in result["required_tools"]:
                if tool not in tool_effectiveness:
                    tool_effectiveness[tool] = {"used": 0, "successful": 0}
                tool_effectiveness[tool]["used"] += 1
                if result["is_correct"]:
                    tool_effectiveness[tool]["successful"] += 1

        # Calculate effectiveness percentage for each tool
        for tool in tool_effectiveness:
            count = tool_effectiveness[tool]
            tool_effectiveness[tool]["effectiveness"] = (
                count["successful"] / count["used"] if count["used"] > 0 else 0
            )

        detailed_metrics = {
            "total_examples": total_examples,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "unique_tools_used": unique_tools,
            "tool_count": len(unique_tools),
            "correct_tool_usage": correct_tool_usage,
            "accuracy_by_level": level_accuracies,
            "tool_effectiveness": tool_effectiveness,
            "average_tools_per_task": len(all_tools) / total_examples if total_examples > 0 else 0
        }

        return detailed_metrics
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to file
        
        Args:
            output_path: Path to save results
        """
        results_with_metrics = {
            "evaluation_config": {
                "model_name": self.model_name,
                "dataset": "GAIA",
                "timestamp": time.time(),
                "results_count": len(self.results)
            },
            "metrics": self.calculate_metrics(),
            "detailed_results": self.results
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """
        Print evaluation summary
        """
        metrics = self.calculate_metrics()

        print("\n" + "="*60)
        print("GAIA EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {self.model_name or 'UNSPECIFIED'}")
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Correct Predictions: {metrics['correct_predictions']}")
        print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Unique Tools Required: {metrics['tool_count']}")
        print(f"Average Tools Per Task: {metrics['average_tools_per_task']:.2f}")

        print("\nAccuracy by Level:")
        for level, acc in metrics['accuracy_by_level'].items():
            print(f"  {level}: {acc*100:.2f}%")

        print("\nMost Effective Tools:")
        # Sort tools by effectiveness
        sorted_tools = sorted(metrics['tool_effectiveness'].items(),
                             key=lambda x: x[1]['effectiveness'], reverse=True)
        for tool, stats in sorted_tools[:5]:  # Top 5 tools
            effectiveness = stats['effectiveness']
            print(f"  {tool}: {effectiveness*100:.2f}% success rate ({stats['used']} uses)")

        print("\nAll Required Tools:")
        for tool in metrics['unique_tools_used']:
            print(f"  - {tool}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GAIA benchmark")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to evaluate")
    parser.add_argument("--dataset_level", type=str, default="2023_level1", 
                       choices=["2023_level1", "2023_level2", "2023_level3"],
                       help="GAIA dataset level to evaluate")
    parser.add_argument("--output_path", type=str, default="./results/gaia_evaluation.json",
                       help="Path to save evaluation results")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache",
                       help="Directory to cache datasets and models")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GAIAEvaluator(model_name=args.model_name, cache_dir=args.cache_dir)
    
    # Setup dataset
    evaluator.setup_dataset(level=args.dataset_level)
    
    # Run evaluation (with placeholder model interface)
    # In a real scenario, you would pass an actual model interface
    results = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(args.output_path)