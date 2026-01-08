# Reliability Engineering Dataset Generator

A three-stage pipeline for extracting, augmenting, and solving reliability engineering exercises from OCR-processed textbooks.

## Pipeline Stages

1. **Extract**: Identify exercises from textbook chunks using GPT-4o-mini
2. **Augment**: Rewrite questions to be self-contained and standalone
3. **Solve**: Generate step-by-step reasoning with answer verification using DeepSeek R1

## Features

- Multi-threaded processing for faster execution
- Automatic retry with exponential backoff for API errors
- Quality validation filters (ghost questions, tautologies, context leaks)
- Cost tracking for API usage
- Resume capability (skips already processed files)

## Configuration

Edit the following variables in `new5.py`:

```python
OPENROUTER_API_KEY = "your-api-key"
TEXTBOOKS_FOLDER = "path/to/your/textbooks"
```

## Usage

```bash
pip install openai
python new5.py
```

## Output

- `dataset_reliability_augmented.jsonl`: Validated Q&A pairs
- `dataset_rejected.jsonl`: Rejected entries with failure reasons

## Models Used

- **Extraction & Augmentation**: `openai/gpt-4o-mini`
- **Reasoning**: `deepseek/deepseek-r1-distill-llama-70b`
