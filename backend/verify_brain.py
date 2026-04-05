import numpy as np
from tensorflow.keras.models import load_model
import os

# 1. Setup
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'yes', 'dele_alli', 'salute', 'yoyohoneysingh', 'nothing'])
sequence_length = 60
total_videos = 30

print("🧠 Loading AI Brain...")
try:
    model = load_model('action.h5')
except:
    print("❌ Error: Could not load action.h5. Make sure you are in the backend folder.")
    exit()

print("\n📊 --- FULL DATASET REPORT CARD ---")
print("Testing all 30 videos for each sign...\n")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"⚠️ Skipping {action.upper()}: Folder not found.")
        continue
        
    correct_guesses = 0
    
    for video_number in range(total_videos):
        sequence = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_path, str(video_number), f"{frame_num}.npy")
            if os.path.exists(file_path):
                sequence.append(np.load(file_path))
            else:
                sequence.append(np.zeros(126)) 
        
        sequence_array = np.expand_dims(sequence, axis=0)
        predictions = model.predict(sequence_array, verbose=0)[0]
        best_guess_index = np.argmax(predictions)
        
        if actions[best_guess_index] == action:
            correct_guesses += 1
            
    accuracy = (correct_guesses / total_videos) * 100
    
    if accuracy >= 80:
        status = "✅ STRONG"
    elif accuracy >= 50:
        status = "⚠️ WEAK (Might glitch in browser)"
    else:
        status = "❌ BROKEN"
        
    print(f"{action.upper()}: {accuracy:.1f}% accuracy ({correct_guesses}/{total_videos} correct) -> {status}")

print("\n🏁 DIAGNOSTIC COMPLETE.")