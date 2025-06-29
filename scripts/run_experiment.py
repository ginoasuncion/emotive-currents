#!/usr/bin/env python3
"""
Script to run emotion classification experiments using config files.
"""

import json
import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from emotive_currents.experiments.emotion_classifier import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Run emotion classification experiment')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--samples-per-emotion', type=int, default=3, 
                       help='Number of samples per emotion (default: 3)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Running experiment with config: {args.config}")
    print(f"Model: {config['model']}")
    print(f"Samples per emotion: {args.samples_per_emotion}")
    
    # Handle both "prompt" and "prompt_config" keys
    prompt_config = config.get('prompt_config') or config.get('prompt')
    
    # Run experiment
    results = run_experiment(
        model=config['model'],
        temperature=config.get('temperature', 0.1),
        samples_per_emotion=args.samples_per_emotion,
        prompt_strategy=config.get('prompt_strategy', 'zero-shot'),
        random_seed=config.get('random_seed', 42),
        experiment_name=config.get('experiment_name'),
        description=config.get('description'),
        prompt_config=prompt_config
    )
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {results['results_file']}")
    print(f"Experiment ID: {results['experiment_id']}")


if __name__ == '__main__':
    main() 