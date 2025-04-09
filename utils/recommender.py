import pandas as pd
from collections import Counter

def preprocess_emotions(emotion_list):
    """Process detected emotions and return ordered unique list"""
    emotion_counts = Counter(emotion_list)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    
    unique_emotions = []
    for x in result:
        if x not in unique_emotions:
            unique_emotions.append(x)
    
    return unique_emotions

def get_recommendations(emotion_list, data):
    """Get music recommendations based on emotions"""
    data_frames = {
        'Neutral': data['neutral'],
        'Angry': data['angry'],
        'Fearful': data['fear'],
        'Happy': data['happy'],
        'Sad': data['sad']
    }
    
    recommendation_df = pd.DataFrame()
    
    if len(emotion_list) == 1:
        sample_sizes = [30]
    elif len(emotion_list) == 2:
        sample_sizes = [30, 20]
    elif len(emotion_list) == 3:
        sample_sizes = [55, 20, 15]
    elif len(emotion_list) == 4:
        sample_sizes = [30, 29, 18, 9]
    else:
        sample_sizes = [10, 7, 6, 5, 2]
    
    for i, emotion in enumerate(emotion_list):
        if emotion in data_frames:
            sample_size = sample_sizes[i] if i < len(sample_sizes) else 5
            recommendation_df = pd.concat([
                recommendation_df, 
                data_frames[emotion].sample(n=sample_size)
            ], ignore_index=True)
    
    return recommendation_df