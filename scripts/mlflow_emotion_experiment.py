#!/usr/bin/env python3
"""
Sample MLflow experiment logging for emotion analysis prompt.
"""

import mlflow
import json
from datetime import datetime

# Set the tracking URI to the running MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Example prompt and input
prompt = (
    "You are an expert in emotion analysis. Given a sentence or paragraph, your task is to estimate the emotional intensity for each of 27 emotions defined in the GoEmotions dataset. Each emotion should be scored between 0.0 (not present) and 1.0 (very strongly present). Multiple emotions may be present at once.\n"
    "Respond with a valid JSON object, where each key is an emotion label and each value is a float between 0.0 and 1.0.\n"
    "Here are the 27 emotion labels: [\"admiration\", \"amusement\", \"anger\", \"annoyance\", \"approval\", \"caring\", \"confusion\", \"curiosity\", \"desire\", \"disappointment\", \"disapproval\", \"embarrassment\", \"excitement\", \"fear\", \"gratitude\", \"grief\", \"joy\", \"love\", \"nervousness\", \"optimism\", \"pride\", \"realization\", \"relief\", \"remorse\", \"sadness\", \"surprise\", \"neutral\"]\n"
    "Example input: I'm really trying my best but everything is falling apart."
)

input_text = "I'm really trying my best but everything is falling apart."

# Example output (as would be returned by an LLM)
emotion_vector = {
  "admiration": 0.0,
  "amusement": 0.0,
  "anger": 0.6,
  "annoyance": 0.4,
  "approval": 0.0,
  "caring": 0.2,
  "confusion": 0.3,
  "curiosity": 0.0,
  "desire": 0.0,
  "disappointment": 0.8,
  "disapproval": 0.0,
  "embarrassment": 0.0,
  "excitement": 0.0,
  "fear": 0.3,
  "gratitude": 0.0,
  "grief": 0.7,
  "joy": 0.0,
  "love": 0.0,
  "nervousness": 0.2,
  "optimism": 0.0,
  "pride": 0.0,
  "realization": 0.4,
  "relief": 0.0,
  "remorse": 0.4,
  "sadness": 0.9,
  "surprise": 0.0,
  "neutral": 0.2
}

# Example metric (e.g., macro F1 score from evaluation)
macro_f1 = 0.78

# Log to MLflow
with mlflow.start_run(run_name="emotion-prompt-sample"):
    mlflow.log_param("prompt_version", "v1.0")
    mlflow.log_param("llm_name", "openai/gpt-4o")
    mlflow.log_param("input_text", input_text)
    mlflow.log_param("timestamp", datetime.now().isoformat())
    mlflow.log_metric("macro_f1", macro_f1)
    # Log the full emotion vector as an artifact
    with open("emotion_vector.json", "w") as f:
        json.dump(emotion_vector, f, indent=2)
    mlflow.log_artifact("emotion_vector.json")
    # Log the prompt as an artifact
    with open("prompt.txt", "w") as f:
        f.write(prompt)
    mlflow.log_artifact("prompt.txt")

print("Logged sample experiment to MLflow!") 