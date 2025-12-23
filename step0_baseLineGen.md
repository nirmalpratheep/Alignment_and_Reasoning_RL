# Alignment and Reasoning RL - MATH Evaluation

Evaluation of Qwen 2.5 Math 1.5B base model on MATH dataset using RL zero prompt format.

**Based on**: [Stanford CS336 Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment)

## Quick Start

```bash
# Install dependencies
uv sync --extra vllm  # Linux/WSL only (vLLM not supported on Windows)

# Run evaluation
uv run python evaluate_model.py
```

## Evaluation Results (Qwen 2.5 Math 1.5B - Base Model)

**Dataset**: MATH (Hendrycks et al.) - 5,000 test examples across 7 subsets  
**Prompt Format**: RL zero (`<think>` reasoning + `<answer>` tags)

### Performance Metrics

| Category | Count | Percentage |
|----------|-------|------------|
| **Correct** (Format=1, Answer=1) | 142 | **2.84%** |
| **Wrong Answer** (Format=1, Answer=0) | 657 | 13.14% |
| **Format Failure** (Format=0, Answer=0) | 4,201 | 84.02% |

### Format Failure Analysis (4,201 cases)

| Issue | Count | Percentage |
|-------|-------|------------|
| Missing `</think>` tag | 2,397 | 57.1% |
| Missing `<answer>` tag | 2,277 | 54.2% |
| Missing `</answer>` tag | 2,567 | 61.1% |
| Incomplete generation | 2,567 | 61.1% |
| Wrong tag order | 904 | 21.5% |

**Conclusion**: Format failures are due to the **base model** not following instructions, **NOT parser issues**.

### Wrong Answer Analysis (657 cases)

The parser correctly extracted answers in all cases. Issues found:
- Incorrect mathematical reasoning (e.g., `x=2` instead of `x=4`)
- Calculation errors (e.g., wrong completing squares)
- Incomplete answers (e.g., described asymptotes but didn't count them)
- Domain/range errors (e.g., missing union in interval notation)

**Conclusion**: This is a **model capability issue** requiring better mathematical reasoning and domain knowledge.

## Key Findings

1. **Base model accuracy: 2.84%** - Expected for non-instruction-tuned models
2. **84% format failures** - Model needs format-aware training (SFT/RL)
3. **Among formatted responses (16%), only 17.8% correct** - Needs both format AND math improvement
4. **Parser/grader verified working correctly** - All issues stem from model outputs

## Files

```
.
├── evaluate_model.py           # Evaluation script with logging
├── drgrpo_grader.py           # Grading functions (from Dr. GRPO)
├── prompts/rl_zero.prompt     # RL zero prompt template
└── results/                    # Logs and JSON results
```

## Reference

This evaluation is based on methods from Stanford CS336 Assignment 5:
- **Repository**: https://github.com/stanford-cs336/assignment5-alignment
- **Grader**: Adapted from Dr. GRPO's math grading system
- **Dataset**: MATH dataset (Hendrycks et al. 2021)
- **Prompt Format**: RL zero-shot prompting with reasoning tags
