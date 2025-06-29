#!/usr/bin/env python3
"""
Analyze per-emotion performance from experiment results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from emotive_currents.experiments.emotion_labels import EMOTION_LABELS

def analyze_per_emotion_performance(results_file):
    """Analyze performance for each emotion."""
    
    print("=" * 80)
    print("PER-EMOTION PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Load results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} samples from {results_file}")
    
    # Calculate per-emotion metrics
    emotion_performance = []
    
    for emotion in EMOTION_LABELS:
        true_col = f'true_{emotion}'
        pred_col = f'pred_{emotion}'
        
        if true_col in df.columns and pred_col in df.columns:
            # Get true and predicted values
            true_values = df[true_col].values
            pred_values = df[pred_col].values
            
            # Convert to binary (threshold 0.5 for predictions)
            true_binary = (true_values > 0).astype(int)
            pred_binary = (pred_values > 0.5).astype(int)
            
            # Calculate metrics
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate accuracy for this emotion
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            # Count samples where this emotion is present
            emotion_present = np.sum(true_binary)
            
            emotion_performance.append({
                'emotion': emotion,
                'samples_present': emotion_present,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
    
    # Create DataFrame and sort by F1 score
    perf_df = pd.DataFrame(emotion_performance)
    perf_df = perf_df.sort_values('f1', ascending=False)
    
    # Display results
    print("\n📊 PER-EMOTION PERFORMANCE RANKING (by F1 Score)")
    print("=" * 80)
    print(f"{'Emotion':<15} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10} {'Samples':<8}")
    print("-" * 80)
    
    for _, row in perf_df.iterrows():
        print(f"{row['emotion']:<15} {row['f1']:<8.3f} {row['precision']:<10.3f} {row['recall']:<8.3f} {row['accuracy']:<10.3f} {row['samples_present']:<8}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average F1 Score: {perf_df['f1'].mean():.3f}")
    print(f"Average Precision: {perf_df['precision'].mean():.3f}")
    print(f"Average Recall: {perf_df['recall'].mean():.3f}")
    print(f"Average Accuracy: {perf_df['accuracy'].mean():.3f}")
    
    # Top performers
    print(f"\n🏆 TOP 5 PERFORMING EMOTIONS:")
    for i, (_, row) in enumerate(perf_df.head(5).iterrows(), 1):
        print(f"{i}. {row['emotion']}: F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")
    
    # Bottom performers
    print(f"\n📉 BOTTOM 5 PERFORMING EMOTIONS:")
    for i, (_, row) in enumerate(perf_df.tail(5).iterrows(), 1):
        print(f"{i}. {row['emotion']}: F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")
    
    # Emotions with no samples
    zero_samples = perf_df[perf_df['samples_present'] == 0]
    if len(zero_samples) > 0:
        print(f"\n⚠️  EMOTIONS WITH NO SAMPLES ({len(zero_samples)}):")
        for emotion in zero_samples['emotion']:
            print(f"  - {emotion}")
    
    return perf_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze per-emotion performance")
    parser.add_argument('--results_file', type=str, help='Path to results CSV file')
    args = parser.parse_args()
    
    if args.results_file:
        results_file = args.results_file
    else:
        # Find the most recent results file
        results_dir = Path("experiment_results")
        csv_files = list(results_dir.glob("*.csv"))
        if not csv_files:
            print("No results files found!")
            return
        
        # Get the most recent file
        results_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Using most recent results file: {results_file}")
    
    analyze_per_emotion_performance(results_file)

if __name__ == "__main__":
    main() 