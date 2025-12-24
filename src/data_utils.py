"""Data preparation utilities for SFT training."""
from utils.drgrpo_grader import r1_zero_reward_fn


def extract_answer_from_solution(solution: str) -> str:
    """Extract the final answer from the solution (from \\boxed{} or ####).
    
    Args:
        solution: Solution text containing answer
        
    Returns:
        Extracted answer string
    """
    # Try to extract from \boxed{}
    if "\\boxed{" in solution:
        start = solution.rfind("\\boxed{")
        if start != -1:
            start += len("\\boxed{")
            # Find matching closing brace
            brace_count = 1
            i = start
            while i < len(solution) and brace_count > 0:
                if solution[i] == "{":
                    brace_count += 1
                elif solution[i] == "}":
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                return solution[start:i-1].strip()
    
    return ""


def prepare_sft_dataset(examples: list, prompt_template: str) -> list:
    """Convert examples to SFT format (prompt-response pairs) with grading.
    
    Args:
        examples: List of dataset examples with 'problem' and 'solution'
        prompt_template: Prompt template with {question} placeholder
        
    Returns:
        List of SFT data dictionaries with prompt, response, and grading info
    """
    sft_data = []
    
    for example in examples:
        problem = example['problem']
        solution = example['solution']
        
        # Create prompt by replacing {question} in template
        prompt = prompt_template.replace('{question}', problem)
        
        # Extract answer from solution
        answer_value = extract_answer_from_solution(solution)
        
        # Create response: solution (thinking) + closing tags with answer
        # Format must match drgrpo_grader.py expectation: "</think> <answer>"
        response = solution + "\n</think> <answer>" + f"{answer_value}</answer>"
        
        # Grade the response using drgrpo_grader
        try:
            reward_dict = r1_zero_reward_fn(response=response, ground_truth=solution, fast=True)
        except Exception as e:
            reward_dict = {
                'format_reward': 0.0,
                'answer_reward': 0.0,
                'reward': 0.0
            }
        
        sft_data.append({
            'prompt': prompt,
            'response': response,
            'problem': problem,
            'solution': solution,
            'answer': answer_value,
            'format_reward': reward_dict.get('format_reward', 0.0),
            'answer_reward': reward_dict.get('answer_reward', 0.0),
            'reward': reward_dict.get('reward', 0.0)
        })
    
    return sft_data
