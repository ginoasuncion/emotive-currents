# Experiment Configuration Files

This directory contains JSON configuration files for running emotion classification experiments.

## Available Configs

- `gemma_3_27b.json` - Configuration for Google Gemma 3 27B experiments
- `template.json` - Template for creating new configurations

## Usage

```bash
# Run experiment with Google Gemma 3 27B config
python scripts/run_experiment.py --config configs/gemma_3_27b.json

# Run with config and override temperature
python scripts/run_experiment.py --config configs/gemma_3_27b.json --temperature 0.2
```

## Features

The experiment framework includes:

- **Nested MLflow experiments** based on prompt strategy and model
- **Backend logging** for progress tracking
- **API latency tracking** for performance monitoring
- **Prompt tracking** in MLflow
- **Per-emotion performance analysis** with detailed metrics
- **Comprehensive metrics** including Macro/Micro F1, Precision, Recall, Top-K metrics
- **Stratified sampling** ensuring representation of all emotions
- **Reproducible results** with fixed random seeds

## Config File Structure

```json
{
  "experiment_name": "experiment-name-here",
  "model": "model-name-here",
  "temperature": 0.1,
  "samples_per_emotion": 3,
  "prompt_strategy": "zero-shot",
  "random_seed": 42,
  "prompt": {
    "template": "Your prompt template here with {variables}",
    "variables": {
      "variable1": "value1",
      "variable2": "value2"
    },
    "output_format": "JSON",
    "instructions": "Additional instructions for the model"
  },
  "description": "Custom experiment description here...",
  "metadata": {
    "model_provider": "Provider name",
    "model_family": "Model family",
    "model_version": "Model version",
    "expected_performance": "Expected performance level",
    "use_case": "Use case description",
    "dataset": "Dataset name",
    "notes": "Additional notes about the experiment"
  }
}
```

## Creating New Configs

1. Copy `template.json` to a new file
2. Update the parameters for your experiment
3. Customize the prompt template and variables
4. Add appropriate metadata
5. Run with `--config your_config.json` 