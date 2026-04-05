from flask import Flask
from flask_socketio import SocketIO
from engineio.payload import Payload
import cv2
import numpy as np
import mediapipe as mp
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
# ping_timeout and ping_interval help prevent the "Too many packets" crash
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

print("🧠 Loading AI Brain...")
try:
    model = load_model('action.h5')
except Exception as e:
    print(f"❌ Error loading model: {e}")
    
actions = np.array(['hello', 'thanks', 'yes', 'dele_alli', 'salute', 'yoyohoneysingh', 'nothing'])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []
last_guess = "" 
frame_counter = 0
Payload.max_decode_packets = 500
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

@socketio.on('connect')
def test_connect():
    print("🟢 React Frontend Connected!")

@socketio.on('image')
def receive_image(image_data):
    global sequence, last_guess, frame_counter
    try:
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        frame_counter += 1

        # SCENARIO A: HANDS ARE VISIBLE
        if results.multi_hand_landmarks:
            lh = np.zeros(21*3)
            rh = np.zeros(21*3)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label 
                kp = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                if label == 'Left': lh = kp
                else: rh = kp
                
            keypoints = np.concatenate([lh, rh])
            sequence.append(keypoints)
            sequence = sequence[-60:]

            # THROTTLE: Only predict every 5th frame to prevent SocketIO crashes
            if len(sequence) == 60 and frame_counter % 5 == 0:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                best_index = np.argmax(res)
                confidence = res[best_index]

                if confidence > 0.98:
                    action_name = actions[best_index]
                    
                    # Ensure we don't spam the UI or predict 'nothing' when hands are clearly up
                    if action_name != last_guess and action_name != 'nothing':
                        socketio.emit('response_back', {'data': action_name})
                        print(f"\n✅ GUESS: {action_name.upper()} ({confidence*100:.1f}%)")
                        last_guess = action_name

        # SCENARIO B: HANDS ARE DROPPED
        else:
            # INSTANT FLUSH: Destroy the mutant half-and-half buffer
            sequence = []
            
            if last_guess != 'Waiting...':
                socketio.emit('response_back', {'data': 'Waiting...'})
                print("\n💤 HANDS DROPPED: Buffer flushed. UI set to Waiting...")
                last_guess = 'Waiting...'

        # Yield control to SocketIO so it doesn't choke on packets
        socketio.sleep(0)

    except Exception as e:
        pass 

if __name__ == '__main__':
    print("🚀 Starting Backend Server on port 5000...")
    socketio.run(app, port=5000, debug=False)