"""
GoEmotions Dataset Loader
Downloads and preprocesses the GoEmotions dataset for emotion classification.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple, Any
import os
import json


class GoEmotionsLoader:
    """Loads and preprocesses the GoEmotions dataset."""
    
    # 27 GoEmotions labels
    EMOTIONS = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise"
    ]
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.mlb = MultiLabelBinarizer()
        
    def download_dataset(self) -> Dict[str, Any]:
        """Download the GoEmotions dataset from HuggingFace.
        
        Returns:
            Dataset dictionary with train/validation/test splits
        """
        print("🔄 Downloading GoEmotions dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("go_emotions")
            
            print(f"✅ Dataset downloaded successfully!")
            print(f"📊 Train: {len(dataset['train'])} samples")
            print(f"📊 Validation: {len(dataset['validation'])} samples") 
            print(f"📊 Test: {len(dataset['test'])} samples")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            raise
    
    def preprocess_data(self, dataset: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Preprocess the dataset for multi-label classification.
        
        Args:
            dataset: Raw dataset from HuggingFace
            
        Returns:
            Dictionary of processed DataFrames
        """
        print("🔄 Preprocessing data...")
        
        processed_data = {}
        
        for split in ['train', 'validation', 'test']:
            print(f"Processing {split} split...")
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset[split])
            
            # Clean text
            df['text'] = df['text'].str.strip()
            df = df[df['text'].str.len() > 0]  # Remove empty texts
            
            # Convert labels to emotion names
            df['emotion_labels'] = df['labels'].apply(
                lambda x: [self.EMOTIONS[i] for i in x if i < len(self.EMOTIONS)]
            )
            
            # Create multi-label binary matrix
            if split == 'train':
                # Fit the MultiLabelBinarizer on training data
                emotion_matrix = self.mlb.fit_transform(df['emotion_labels'])
            else:
                emotion_matrix = self.mlb.transform(df['emotion_labels'])
            
            # Add binary columns for each emotion
            for i, emotion in enumerate(self.mlb.classes_):
                df[f'emotion_{emotion}'] = emotion_matrix[:, i]
            
            # Calculate emotion counts for analysis
            df['emotion_count'] = df['emotion_labels'].apply(len)
            
            processed_data[split] = df
            
            print(f"✅ {split}: {len(df)} samples processed")
            print(f"   Average emotions per sample: {df['emotion_count'].mean():.2f}")
        
        return processed_data
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files.
        
        Args:
            data: Dictionary of processed DataFrames
        """
        print("💾 Saving processed data...")
        
        for split, df in data.items():
            # Save as parquet for efficiency
            output_path = os.path.join(self.processed_dir, f"{split}.parquet")
            df.to_parquet(output_path, index=False)
            print(f"✅ Saved {split} data to {output_path}")
        
        # Save emotion mapping
        mapping = {
            "emotions": list(self.mlb.classes_),
            "num_emotions": len(self.mlb.classes_),
            "emotion_to_idx": {emotion: i for i, emotion in enumerate(self.mlb.classes_)}
        }
        
        mapping_path = os.path.join(self.processed_dir, "emotion_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✅ Saved emotion mapping to {mapping_path}")
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Load previously processed data.
        
        Returns:
            Dictionary of processed DataFrames
        """
        data = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = os.path.join(self.processed_dir, f"{split}.parquet")
            if os.path.exists(file_path):
                data[split] = pd.read_parquet(file_path)
            else:
                raise FileNotFoundError(f"Processed data not found: {file_path}")
        
        return data
    
    def get_emotion_statistics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate emotion distribution statistics.
        
        Args:
            data: Dictionary of processed DataFrames
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for split, df in data.items():
            emotion_cols = [col for col in df.columns if col.startswith('emotion_') and col != 'emotion_labels' and col != 'emotion_count']
            
            emotion_counts = df[emotion_cols].sum().sort_values(ascending=False)
            emotion_percentages = (emotion_counts / len(df) * 100).round(2)
            
            stats[split] = {
                'total_samples': len(df),
                'avg_emotions_per_sample': df['emotion_count'].mean(),
                'emotion_distribution': emotion_counts.to_dict(),
                'emotion_percentages': emotion_percentages.to_dict(),
                'most_common_emotions': emotion_counts.head(10).to_dict(),
                'least_common_emotions': emotion_counts.tail(10).to_dict()
            }
        
        return stats
    
    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Run the complete data loading and preprocessing pipeline.
        
        Returns:
            Dictionary of processed DataFrames
        """
        print("🌊 Starting GoEmotions data pipeline...")
        
        # Check if processed data already exists
        try:
            data = self.load_processed_data()
            print("✅ Found existing processed data!")
            return data
        except FileNotFoundError:
            print("📁 No processed data found, starting from scratch...")
        
        # Download and process data
        raw_dataset = self.download_dataset()
        processed_data = self.preprocess_data(raw_dataset)
        self.save_processed_data(processed_data)
        
        # Print statistics
        stats = self.get_emotion_statistics(processed_data)
        print("\n📊 Dataset Statistics:")
        for split, split_stats in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Total samples: {split_stats['total_samples']:,}")
            print(f"  Avg emotions per sample: {split_stats['avg_emotions_per_sample']:.2f}")
            print(f"  Top 5 emotions:")
            for emotion, count in list(split_stats['most_common_emotions'].items())[:5]:
                emotion_name = emotion.replace('emotion_', '')
                percentage = split_stats['emotion_percentages'][emotion]
                print(f"    {emotion_name}: {count:,} ({percentage}%)")
        
        print("\n✅ Data pipeline completed!")
        return processed_data


if __name__ == "__main__":
    # Example usage
    loader = GoEmotionsLoader()
    data = loader.run_pipeline()
    
    print(f"\n🎯 Data ready for training!")
    print(f"Train samples: {len(data['train']):,}")
    print(f"Validation samples: {len(data['validation']):,}")
    print(f"Test samples: {len(data['test']):,}")
