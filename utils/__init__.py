"""
emotion-music-recommender utilities package

This package contains all the helper functions and modules for:
- Emotion detection
- Data loading
- Music recommendation
"""

from .data_loader import load_data
from .emotion_detection import detect_emotions, load_emotion_model
from .recommender import preprocess_emotions, get_recommendations

__all__ = [
    'load_data',
    'detect_emotions',
    'load_emotion_model',
    'preprocess_emotions',
    'get_recommendations'
]

__version__ = '1.0.0'