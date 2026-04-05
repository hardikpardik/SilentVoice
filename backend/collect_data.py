import cv2
import numpy as np
import os
import mediapipe as mp
import time

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 
# The new, distinct V1 list
actions = np.array(['hello', 'thanks', 'yes', 'dele_alli', 'salute', 'yoyohoneysingh', 'nothing'])
no_sequences = 30
sequence_length = 60

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create all folders upfront
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:
        # Give you 3 seconds to get ready for the new sign
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'GET READY FOR: {action.upper()} in {i}...', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            cv2.waitKey(1000)

        # Loop through videos
        for sequence in range(no_sequences):
            # Loop through frames
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Keypoints
                lh = np.zeros(21*3)
                rh = np.zeros(21*3)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label 
                        kp = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                        if label == 'Left': lh = kp
                        else: rh = kp
                
                keypoints = np.concatenate([lh, rh])
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # UI Feedback
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting {action} - Video {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1500) # 1.5 second pause between videos to reset hands
                else: 
                    cv2.putText(image, f'Collecting {action} - Video {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
cap.release()
cv2.destroyAllWindows()