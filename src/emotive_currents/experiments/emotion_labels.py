"""
Emotion labels for the GoEmotions dataset.
"""

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Number of emotions
NUM_EMOTIONS = len(EMOTION_LABELS)

# Create a mapping from emotion to index for easy lookup
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
INDEX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTION_LABELS)} 