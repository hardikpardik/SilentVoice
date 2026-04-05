import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Setup Configuration
DATA_PATH = os.path.join('MP_Data')
# Your elite V1 lineup
actions = np.array(['hello', 'thanks', 'yes', 'dele_alli', 'salute', 'yoyohoneysingh', 'nothing'])
no_sequences = 30
sequence_length = 60
label_map = {label:num for num, label in enumerate(actions)}

# 2. Load the Dataset
print("📂 Loading data...")
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. Build the Neural Network Architecture
print("🧠 Building the AI Brain...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("🔥 Training started! Watch the accuracy numbers...")
model.fit(X_train, y_train, epochs=200, verbose=1)

# 5. Save the finished model
model.save('action.h5')
print("✅ Training Complete. New brain saved as action.h5!")