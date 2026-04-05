import os
# 1. GOLDEN LINES (Silence TensorFlow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# 2. SETUP & CONFIG
actions = np.array(['hello', 'thanks', 'iloveyou', 'dele_alli', 'peace', 'sorry', 'yes', 'no'])

# Load the Brain
try:
    model_path = os.path.join(os.path.dirname(__file__), 'action.h5')
    model = load_model(model_path)
    print("✅ Model Loaded! Expecting 60 frames.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 3. HELPER FUNCTION
def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            kp = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == 'Left': lh = kp
            else: rh = kp
    return np.concatenate([lh, rh])

# 4. MAIN LOOP
sequence = []
sentence = []
threshold = 0.8 

cap = cv2.VideoCapture(0)

print("Starting Camera... Perform signs slowly (2 seconds).")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # A. Process Image
    image = cv2.flip(frame, 1) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # B. Draw Hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # C. Prediction Logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    
    # === CRITICAL FIX IS HERE ===
    sequence = sequence[-60:] # CHANGED FROM 30 TO 60

    if len(sequence) == 60:   # CHANGED FROM 30 TO 60
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        best_guess_index = np.argmax(res)
        confidence = res[best_guess_index]
        
        if confidence > threshold: 
            sentence = actions[best_guess_index]
        else:
            sentence = "..."

    # D. Display UI
    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, str(sentence), (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('SilentVoice AI Tester (60 Frames)', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()