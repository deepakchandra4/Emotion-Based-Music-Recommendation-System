import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load pre-trained model (using more efficient loading)
def load_enhanced_model():
    """Load enhanced emotion detection model"""
    model = load_model('models/enhanced_emotion_model.h5')  # Replace with your better model
    return model

# Emotion dictionary with confidence thresholds
EMOTION_DICT = {
    0: {"label": "Angry", "color": (255, 0, 0), "threshold": 0.7},
    1: {"label": "Disgust", "color": (0, 255, 0), "threshold": 0.6},
    2: {"label": "Fear", "color": (255, 0, 255), "threshold": 0.65},
    3: {"label": "Happy", "color": (255, 255, 0), "threshold": 0.75},
    4: {"label": "Neutral", "color": (200, 200, 200), "threshold": 0.6},
    5: {"label": "Sad", "color": (0, 0, 255), "threshold": 0.68},
    6: {"label": "Surprise", "color": (0, 255, 255), "threshold": 0.7}
}

def process_frame(frame, model, face_cascade, emotion_history):
    """Process a single frame for emotion detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_roi = cv2.resize(face_roi, (48, 48))
        normalized_roi = resized_roi / 255.0
        reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
        
        # Get emotion predictions
        predictions = model.predict(reshaped_roi)[0]
        max_index = np.argmax(predictions)
        confidence = predictions[max_index]
        
        # Only register high-confidence predictions
        if confidence > EMOTION_DICT[max_index]["threshold"]:
            emotion = EMOTION_DICT[max_index]["label"]
            emotion_history[emotion] += 1
            
            # Draw face rectangle and emotion label
            color = EMOTION_DICT[max_index]["color"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{emotion} {confidence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame, emotion_history