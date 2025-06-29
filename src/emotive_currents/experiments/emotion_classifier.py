import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from emotive_currents.ai_client import OpenRouterClient
from emotive_currents.experiments.emotion_labels import EMOTION_LABELS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionClassifierExperiment:
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        samples_per_emotion: int = 3,
        prompt_strategy: str = "zero-shot",
        random_seed: int = 42,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_config: Optional[Dict] = None
    ):
        self.model = model
        self.temperature = temperature
        self.samples_per_emotion = samples_per_emotion
        self.prompt_strategy = prompt_strategy
        self.random_seed = random_seed
        self.experiment_name = experiment_name or f"{model.replace('/', '_')}_emotion_experiment"
        self.description = description
        
        # Handle both "prompt" and "prompt_config" keys
        self.prompt_config = prompt_config
        self._original_prompt_config = prompt_config  # Store for logging
        
        self.client = OpenRouterClient()
        self.results = []
        self.latency_data = []
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Initialized experiment: {self.experiment_name}")
        logger.info(f"Model: {model}, Temperature: {temperature}, Samples per emotion: {samples_per_emotion}")

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the GoEmotions dataset."""
        logger.info("Loading GoEmotions dataset...")
        
        # Load the processed data (try parquet first, then CSV)
        data_path = Path("data/processed/validation.parquet")
        if not data_path.exists():
            data_path = Path("data/processed/goemotions_validation.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            df = pd.read_csv(data_path)
        else:
            df = pd.read_parquet(data_path)
        
        logger.info(f"Loaded {len(df)} samples from dataset")
        
        # Map EMOTION_LABELS to columns with 'emotion_' prefix
        emotion_col_map = {label: f'emotion_{label}' for label in EMOTION_LABELS}
        emotion_cols = list(emotion_col_map.values())
        
        # Handle neutral samples (samples with emotion_count = 0)
        neutral_indices = df[df['emotion_count'] == 0].index.tolist()
        if len(neutral_indices) >= self.samples_per_emotion:
            neutral_sampled = np.random.choice(neutral_indices, self.samples_per_emotion, replace=False)
        else:
            neutral_sampled = neutral_indices
            logger.warning(f"Only {len(neutral_indices)} neutral samples available, requested {self.samples_per_emotion}")
        
        # Create stratified sample ensuring at least samples_per_emotion per emotion
        sampled_indices = list(neutral_sampled)  # Start with neutral samples
        
        # Sample from the 27 emotions (exclude neutral from EMOTION_LABELS for this loop)
        emotion_labels_without_neutral = [label for label in EMOTION_LABELS if label != "neutral"]
        for emotion in emotion_labels_without_neutral:
            emotion_col = f'emotion_{emotion}'
            emotion_indices = df[df[emotion_col] > 0].index.tolist()
            if len(emotion_indices) >= self.samples_per_emotion:
                sampled = np.random.choice(emotion_indices, self.samples_per_emotion, replace=False)
                sampled_indices.extend(sampled)
            else:
                sampled_indices.extend(emotion_indices)
                logger.warning(f"Only {len(emotion_indices)} samples available for emotion '{emotion}', requested {self.samples_per_emotion}")
        
        # Remove duplicates and create final sample
        sampled_indices = list(set(sampled_indices))
        sampled_df = df.iloc[sampled_indices].copy()
        
        # Add neutral column for consistency
        sampled_df['emotion_neutral'] = (sampled_df['emotion_count'] == 0).astype(float)
        
        # Convert all emotion columns to float
        emotion_cols = [f'emotion_{label}' for label in EMOTION_LABELS]
        for col in emotion_cols:
            if col not in sampled_df.columns:
                sampled_df[col] = 0.0
            else:
                sampled_df[col] = sampled_df[col].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
        
        logger.info(f"Created stratified sample with {len(sampled_df)} unique samples")
        logger.info(f"Sample distribution: {sampled_df[emotion_cols].sum().to_dict()}")
        
        return sampled_df

    def generate_prompt(self, text: str) -> str:
        """Generate prompt for emotion classification."""
        if self.prompt_config:
            template = self.prompt_config.get("template", "")
            variables = self.prompt_config.get("variables", {})
            
            # Replace variables in template
            prompt = template
            
            # Handle variables as either list or dict
            if isinstance(variables, dict):
                for key, value in variables.items():
                    if key == "emotion_labels":
                        prompt = prompt.replace("{emotion_labels}", value)
            elif isinstance(variables, list):
                # For list variables, we just need to replace input_text
                pass
            
            # Always replace input_text placeholder with the actual text
            prompt = prompt.replace("{input_text}", text)
            
            return prompt
        else:
            # Fallback to default prompt
            emotion_labels_str = ", ".join(EMOTION_LABELS)
            return f"""You are an expert in emotion analysis. Given a sentence or paragraph, your task is to estimate the emotional intensity for each of 27 emotions defined in the GoEmotions dataset. Each emotion should be scored between 0.0 (not present) and 1.0 (very strongly present). Multiple emotions may be present at once.

Respond with a valid JSON object, where each key is an emotion label and each value is a float between 0.0 and 1.0.

Here are the 27 emotion labels:
{emotion_labels_str}

Input: "{text}"

Respond with:"""

    def call_api_with_latency_tracking(self, prompt: str) -> Tuple[Dict, float]:
        """Call API and track latency."""
        start_time = time.time()
        
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            logger.info(f"API call completed in {latency:.3f}s")
            
            return response, latency
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(f"API call failed after {latency:.3f}s: {str(e)}")
            raise

    def parse_response(self, response: Dict) -> Dict[str, float]:
        """Parse the API response to extract emotion scores with robust JSON handling."""
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Clean the content - remove markdown formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Try multiple JSON extraction strategies
            emotion_scores = None
            
            # Strategy 1: Direct JSON parsing
            try:
                emotion_scores = json.loads(content)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Find JSON object with { and }
            if emotion_scores is None:
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    try:
                        emotion_scores = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Try to fix common JSON issues
            if emotion_scores is None:
                # Fix common issues: single quotes, missing quotes, trailing commas
                fixed_content = content
                # Replace single quotes with double quotes
                fixed_content = fixed_content.replace("'", '"')
                # Remove trailing commas
                fixed_content = fixed_content.replace(',}', '}')
                fixed_content = fixed_content.replace(',]', ']')
                # Add quotes to unquoted keys
                import re
                fixed_content = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_content)
                
                try:
                    emotion_scores = json.loads(fixed_content)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Try to extract key-value pairs manually
            if emotion_scores is None:
                emotion_scores = {}
                # Look for patterns like "emotion": value or emotion: value
                import re
                patterns = [
                    r'"([^"]+)"\s*:\s*([0-9]*\.?[0-9]+)',  # "emotion": 0.5
                    r"'([^']+)'\s*:\s*([0-9]*\.?[0-9]+)",  # 'emotion': 0.5
                    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([0-9]*\.?[0-9]+)',  # emotion: 0.5
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for key, value in matches:
                        try:
                            emotion_scores[key] = float(value)
                        except ValueError:
                            continue
            
            # If we still don't have emotion scores, create default
            if emotion_scores is None:
                logger.warning(f"Could not parse JSON from response: {content[:200]}...")
                emotion_scores = {}
            
            # Validate that all emotions are present and normalize
            final_scores = {}
            for emotion in EMOTION_LABELS:
                if emotion in emotion_scores:
                    try:
                        score = float(emotion_scores[emotion])
                        final_scores[emotion] = max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        final_scores[emotion] = 0.0
                else:
                    final_scores[emotion] = 0.0
            
            # Log successful parsing
            if emotion_scores:
                logger.info(f"Successfully parsed {len(emotion_scores)} emotion scores")
            
            return final_scores
                
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Response content: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:200]}...")
            return {emotion: 0.0 for emotion in EMOTION_LABELS}

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive metrics from experiment results."""
        # Extract true and predicted labels
        y_true = []
        y_pred = []
        
        for result in results:
            # Extract true labels
            true_labels = [result[f'true_{emotion}'] for emotion in EMOTION_LABELS]
            y_true.append(true_labels)
            
            # Extract predicted scores and convert to binary predictions
            pred_scores = [result[f'pred_{emotion}'] for emotion in EMOTION_LABELS]
            # Convert scores to binary predictions (threshold at 0.5)
            pred_labels = [1 if score >= 0.5 else 0 for score in pred_scores]
            y_pred.append(pred_labels)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate basic metrics
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Calculate top-k metrics
        top_2_accuracy = self.calculate_top_k_accuracy(y_true, y_pred, k=2)
        top_2_precision = self.calculate_top_k_precision(y_true, y_pred, k=2)
        top_2_recall = self.calculate_top_k_recall(y_true, y_pred, k=2)
        top_2_f1 = self.calculate_top_k_f1(y_true, y_pred, k=2)
        
        top_3_accuracy = self.calculate_top_k_accuracy(y_true, y_pred, k=3)
        top_3_precision = self.calculate_top_k_precision(y_true, y_pred, k=3)
        top_3_recall = self.calculate_top_k_recall(y_true, y_pred, k=3)
        top_3_f1 = self.calculate_top_k_f1(y_true, y_pred, k=3)
        
        # Calculate per-emotion metrics
        per_emotion_metrics = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            emotion_true = y_true[:, i]
            emotion_pred = y_pred[:, i]
            
            f1 = f1_score(emotion_true, emotion_pred, zero_division=0)
            precision = precision_score(emotion_true, emotion_pred, zero_division=0)
            recall = recall_score(emotion_true, emotion_pred, zero_division=0)
            
            per_emotion_metrics[emotion] = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'micro_precision': micro_precision,
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'top_2_accuracy': top_2_accuracy,
            'top_2_precision': top_2_precision,
            'top_2_recall': top_2_recall,
            'top_2_f1': top_2_f1,
            'top_3_accuracy': top_3_accuracy,
            'top_3_precision': top_3_precision,
            'top_3_recall': top_3_recall,
            'top_3_f1': top_3_f1,
            'per_emotion_metrics': per_emotion_metrics
        }

    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy for multi-label classification."""
        correct_predictions = 0
        total_samples = len(y_true)
        
        for i in range(total_samples):
            true_labels = y_true[i]
            pred_scores = y_pred[i]
            
            # Get top-k predicted emotions
            top_k_indices = np.argsort(pred_scores)[-k:]
            top_k_pred = np.zeros_like(pred_scores)
            top_k_pred[top_k_indices] = 1
            
            # Check if top-k predictions match true labels
            if np.array_equal(true_labels, top_k_pred):
                correct_predictions += 1
        
        return correct_predictions / total_samples if total_samples > 0 else 0.0

    def calculate_top_k_precision(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate top-k precision for multi-label classification."""
        total_precision = 0.0
        valid_samples = 0
        
        for i in range(len(y_true)):
            true_labels = y_true[i]
            pred_scores = y_pred[i]
            
            # Get top-k predicted emotions
            top_k_indices = np.argsort(pred_scores)[-k:]
            top_k_pred = np.zeros_like(pred_scores)
            top_k_pred[top_k_indices] = 1
            
            # Calculate precision for this sample
            if np.sum(true_labels) > 0:  # Only if there are true positives
                intersection = np.sum(true_labels * top_k_pred)
                precision = intersection / np.sum(top_k_pred) if np.sum(top_k_pred) > 0 else 0.0
                total_precision += precision
                valid_samples += 1
        
        return total_precision / valid_samples if valid_samples > 0 else 0.0

    def calculate_top_k_recall(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate top-k recall for multi-label classification."""
        total_recall = 0.0
        valid_samples = 0
        
        for i in range(len(y_true)):
            true_labels = y_true[i]
            pred_scores = y_pred[i]
            
            # Get top-k predicted emotions
            top_k_indices = np.argsort(pred_scores)[-k:]
            top_k_pred = np.zeros_like(pred_scores)
            top_k_pred[top_k_indices] = 1
            
            # Calculate recall for this sample
            if np.sum(true_labels) > 0:  # Only if there are true positives
                intersection = np.sum(true_labels * top_k_pred)
                recall = intersection / np.sum(true_labels) if np.sum(true_labels) > 0 else 0.0
                total_recall += recall
                valid_samples += 1
        
        return total_recall / valid_samples if valid_samples > 0 else 0.0

    def calculate_top_k_f1(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate top-k F1 score for multi-label classification."""
        total_f1 = 0.0
        valid_samples = 0
        
        for i in range(len(y_true)):
            true_labels = y_true[i]
            pred_scores = y_pred[i]
            
            # Get top-k predicted emotions
            top_k_indices = np.argsort(pred_scores)[-k:]
            top_k_pred = np.zeros_like(pred_scores)
            top_k_pred[top_k_indices] = 1
            
            # Calculate F1 for this sample
            if np.sum(true_labels) > 0:  # Only if there are true positives
                intersection = np.sum(true_labels * top_k_pred)
                precision = intersection / np.sum(top_k_pred) if np.sum(top_k_pred) > 0 else 0.0
                recall = intersection / np.sum(true_labels) if np.sum(true_labels) > 0 else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                total_f1 += f1
                valid_samples += 1
        
        return total_f1 / valid_samples if valid_samples > 0 else 0.0

    def calculate_per_emotion_metrics(self, y_true: List[Dict], y_pred: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate per-emotion performance metrics."""
        per_emotion_metrics = {}
        
        for emotion in EMOTION_LABELS:
            true_labels = [1 if true_dict[emotion] > 0 else 0 for true_dict in y_true]
            predicted_labels = [1 if pred_dict[emotion] > 0.5 else 0 for pred_dict in y_pred]
            
            # Calculate metrics for this emotion
            f1 = f1_score(true_labels, predicted_labels, average='binary')
            precision = precision_score(true_labels, predicted_labels, average='binary')
            recall = recall_score(true_labels, predicted_labels, average='binary')
            
            per_emotion_metrics[emotion] = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        return per_emotion_metrics

    def run_experiment(self, experiment_name: str = None, description: str = None) -> Dict:
        """Run the emotion classification experiment."""
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED EMOTION CLASSIFICATION EXPERIMENT")
        logger.info("=" * 80)
        
        # Create comprehensive experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.model.replace('/', '_').replace(':', '_')}_{self.prompt_strategy}_{timestamp}"
        
        logger.info("Loading GoEmotions dataset...")
        sample_data = self.load_data()
        
        # Log sample distribution
        emotion_col_map = {label: f'emotion_{label}' for label in EMOTION_LABELS}
        emotion_cols = list(emotion_col_map.values())
        emotion_counts = sample_data[emotion_cols].sum()
        logger.info(f"Sample distribution: {emotion_counts.to_dict()}")
        
        # Set up MLflow
        mlflow_experiment_name = f"emotion-classification-{datetime.now().strftime('%Y%m%d')}"
        mlflow.set_experiment(mlflow_experiment_name)
        
        # Create run name with model, prompt strategy, and datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract model name (e.g., "claude-sonnet-4" from "anthropic/claude-sonnet-4")
        model_name = self.model.split('/')[-1].replace('-', '_')
        run_name = f"{model_name}_{self.prompt_strategy}_{timestamp}"
        
        # Register prompt if using config-based prompts
        if self.prompt_config:
            prompt_name = f"emotion-classification-{self.prompt_strategy}"
            variables = self.prompt_config.get("variables", {})
            if isinstance(variables, str):
                variables = {"emotion_labels": variables}
            mlflow.register_prompt(
                prompt_name,
                self.prompt_config["template"],
                variables
            )
            logger.info(f"Registered prompt: {prompt_name}")
        
        # Process samples
        results = []
        latencies = []
        
        for i, (idx, row) in enumerate(sample_data.iterrows(), 1):
            text = row['text']
            true_labels = {label: row[f'emotion_{label}'] for label in EMOTION_LABELS}
            
            logger.info(f"Processing sample {idx}/{len(sample_data)}: {text[:50]}...")
            
            # Generate prompt
            prompt = self.generate_prompt(text)
            
            # Make API call
            start_time = time.time()
            try:
                response, latency = self.call_api_with_latency_tracking(prompt)
                
                # Parse response
                predicted_scores = self.parse_response(response)
                logger.info(f"Successfully parsed {len(predicted_scores)} emotion scores")
                
                # Store results
                result = {
                    'sample_id': idx,
                    'text': text,
                    'prompt': prompt,
                    'latency': latency,
                    **{f'true_{emotion}': true_labels[emotion] for emotion in EMOTION_LABELS},
                    **{f'pred_{emotion}': predicted_scores[emotion] for emotion in EMOTION_LABELS}
                }
                results.append(result)
                latencies.append(latency)
                
                logger.info(f"Sample {idx} completed - Latency: {latency:.3f}s")
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                # Add failed result with zeros
                result = {
                    'sample_id': idx,
                    'text': text,
                    'prompt': prompt,
                    'latency': time.time() - start_time,
                    **{f'true_{emotion}': true_labels[emotion] for emotion in EMOTION_LABELS},
                    **{f'pred_{emotion}': 0.0 for emotion in EMOTION_LABELS}
                }
                results.append(result)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.calculate_metrics(results)
        
        # Calculate latency statistics
        latency_stats = {
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies)
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"experiment_results/{timestamp}_{self.model.replace('/', '_').replace(':', '_')}_emotion_experiment_results.csv"
        os.makedirs("experiment_results", exist_ok=True)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_filename, index=False)
        logger.info(f"Results saved to: {results_filename}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("model", self.model)
            mlflow.log_param("temperature", self.temperature)
            mlflow.log_param("samples_per_emotion", self.samples_per_emotion)
            mlflow.log_param("prompt_strategy", self.prompt_strategy)
            mlflow.log_param("random_seed", self.random_seed)
            
            # Log prompt details - handle both "prompt" and "prompt_config" keys
            prompt_config_to_log = self.prompt_config
            if prompt_config_to_log is None and hasattr(self, 'prompt_config'):
                # Try to get from the original config if it was passed as "prompt"
                prompt_config_to_log = getattr(self, '_original_prompt_config', None)
            
            if prompt_config_to_log:
                mlflow.log_param("prompt_template", prompt_config_to_log["template"])
                if "variables" in prompt_config_to_log:
                    mlflow.log_param("prompt_variables", str(prompt_config_to_log["variables"]))
                if "output_format" in prompt_config_to_log:
                    mlflow.log_param("prompt_output_format", prompt_config_to_log["output_format"])
                if "instructions" in prompt_config_to_log:
                    mlflow.log_param("prompt_instructions", prompt_config_to_log["instructions"])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric, value in metric_value.items():
                        if isinstance(value, dict):
                            for sub_sub_metric, sub_value in value.items():
                                mlflow.log_metric(f"{metric_name}_{sub_metric}_{sub_sub_metric}", float(sub_value))
                        else:
                            mlflow.log_metric(f"{metric_name}_{sub_metric}", float(value))
                else:
                    mlflow.log_metric(metric_name, float(metric_value))
            
            # Log latency stats
            for stat_name, stat_value in latency_stats.items():
                mlflow.log_metric(stat_name, stat_value)
            
            # Log artifacts
            mlflow.log_artifact(results_filename)
            
            # Log description
            if description:
                mlflow.log_param("description", description)
            
            # Log experiment metadata
            mlflow.log_param("total_samples", len(results))
            mlflow.log_param("successful_samples", len([r for r in results if r['latency'] > 0]))
            
            # Capture run_id before context ends
            experiment_run_id = run.info.run_id
        
        logger.info("=" * 80)
        logger.info("ENHANCED EXPERIMENT COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model}")
        logger.info(f"Samples tested: {len(results)}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.3f}")
        logger.info(f"Micro F1: {metrics['micro_f1']:.3f}")
        logger.info(f"Average Latency: {latency_stats['avg_latency']:.3f}s")
        
        # Log per-emotion performance
        logger.info("\nPER-EMOTION PERFORMANCE SUMMARY:")
        logger.info("-" * 80)
        for emotion in EMOTION_LABELS:
            emotion_metrics = metrics['per_emotion_metrics'][emotion]
            logger.info(f"{emotion}: F1={emotion_metrics['f1']:.3f}, Precision={emotion_metrics['precision']:.3f}, Recall={emotion_metrics['recall']:.3f}")
        
        return {
            'metrics': metrics,
            'latency_stats': latency_stats,
            'results_file': results_filename,
            'experiment_id': experiment_run_id
        }


def run_experiment(
    model: str,
    temperature: float = 0.1,
    samples_per_emotion: int = 3,
    prompt_strategy: str = "zero-shot",
    random_seed: int = 42,
    experiment_name: Optional[str] = None,
    description: Optional[str] = None,
    prompt_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Run an enhanced emotion classification experiment with comprehensive tracking."""
    experiment = EmotionClassifierExperiment(
        model=model,
        temperature=temperature,
        samples_per_emotion=samples_per_emotion,
        prompt_strategy=prompt_strategy,
        random_seed=random_seed,
        experiment_name=experiment_name,
        description=description,
        prompt_config=prompt_config
    )
    
    return experiment.run_experiment() 