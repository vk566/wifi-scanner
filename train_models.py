import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

print("🚀 Training Advanced Real-World DNN Models...")

# ====================== 1. RSSI CLASSIFICATION MODEL ======================
print("Training RSSI Classification Model...")

# More realistic dataset with noise and environmental variation
X = []
y = []

for rssi in range(-105, -25):
    for _ in range(50):  # Multiple samples per RSSI value
        noisy_rssi = rssi + random.gauss(0, 3)  # Add realistic noise (±3 dBm)
        
        if noisy_rssi > -58:
            label = 0  # Strong
        elif noisy_rssi > -78:
            label = 1  # Medium
        else:
            label = 2  # Weak
            
        X.append([noisy_rssi])
        y.append(label)

X = np.array(X)
y = np.array(y)

# Advanced Model Architecture
model_rssi = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_rssi.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_rssi.fit(X, y, epochs=500, batch_size=64, verbose=1, validation_split=0.2)

model_rssi.save("models/rssi_model.h5")
print("✅ RSSI Model Trained and Saved")

# ====================== 2. DIRECTION MODEL ======================
print("Training Direction Prediction Model...")

X_dir = []
y_dir = []

for base in range(-95, -35):
    for _ in range(30):
        noise = random.gauss(0, 2.5)
        
        # Approaching (Getting closer)
        seq1 = [base + noise, base + 3 + noise, base + 6 + noise]
        X_dir.append(seq1)
        y_dir.append(0)   # Getting closer
        
        # Moving away
        seq2 = [base + 6 + noise, base + 3 + noise, base + noise]
        X_dir.append(seq2)
        y_dir.append(1)   # Moving away
        
        # Stable with small fluctuation
        seq3 = [base + noise, base + random.gauss(0, 1.5), base + noise]
        X_dir.append(seq3)
        y_dir.append(2)   # Stable

X_dir = np.array(X_dir)
y_dir = np.array(y_dir)

model_dir = keras.Sequential([
    layers.Dense(48, activation='relu', input_shape=(3,)),
    layers.Dropout(0.25),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_dir.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_dir.fit(X_dir, y_dir, epochs=600, batch_size=64, verbose=1, validation_split=0.2)

model_dir.save("models/direction_model.h5")
print("✅ Direction Model Trained and Saved")

print("\n🎉 Both models trained successfully!")
print("Models saved in 'models/' folder")