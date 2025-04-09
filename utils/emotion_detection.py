import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def load_emotion_model():
    """Load and return the emotion detection model"""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights('models/model.h5')
    return model

def detect_emotions(detection_time=10):
    """Detect emotions from webcam feed"""
    model = load_emotion_model()
    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful", 
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected_emotions = []
    
    for _ in range(detection_time * 10):  # 10 frames per second
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            emotion = emotion_dict[max_index]
            detected_emotions.append(emotion)
    
    cap.release()
    return detected_emotions