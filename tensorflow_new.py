import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ---- 1. Generate Historical Data ----
np.random.seed(0)
n_hist = 100
time_hist = np.arange(n_hist)

voltage_hist = 230 + np.sin(time_hist / 5) * 10 + np.random.normal(0, 3, n_hist)
current_hist = 5 + np.sin(time_hist / 6) * 0.5 + np.random.normal(0, 0.2, n_hist)
pf_hist = 0.95 + np.cos(time_hist / 8) * 0.03 + np.random.normal(0, 0.01, n_hist)

# ---- 2. Scaling ----
scaler = MinMaxScaler()
data_hist = np.stack([voltage_hist, current_hist, pf_hist], axis=1)
scaled_data = scaler.fit_transform(data_hist)

# ---- 3. Prepare Input for LSTM ----
X, y = [], []
window_size = 10

for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])

X = np.array(X)
y = np.array(y)

# ---- 4. LSTM Model ----
model = Sequential([
    LSTM(64, input_shape=(window_size, 3)),
    Dropout(0.2),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=16, verbose=0)

# ---- 5. Predict Future Values ----
n_future = 50
future_scaled = []
last_seq = scaled_data[-window_size:].reshape(1, window_size, 3)

for _ in range(n_future):
    next_pred = model.predict(last_seq, verbose=0)[0]
    future_scaled.append(next_pred)
    last_seq = np.append(last_seq[:, 1:, :], [[next_pred]], axis=1)

# ---- 6. Inverse Scaling ----
future_scaled = np.array(future_scaled)
future_pred = scaler.inverse_transform(future_scaled)

voltage_future = future_pred[:, 0]
current_future = future_pred[:, 1]
pf_future = future_pred[:, 2]

# ---- 7. Combine Full Data ----
voltage_total = np.concatenate([voltage_hist, voltage_future])
current_total = np.concatenate([current_hist, current_future])
pf_total = np.concatenate([pf_hist, pf_future])
time_total = np.arange(n_hist + n_future)

# ---- 8. Plotting ----
plt.figure(figsize=(15, 8))

# Voltage
plt.subplot(3, 1, 1)
plt.plot(time_hist, voltage_hist, 'b-', label='Historical Voltage')
plt.scatter(np.arange(n_hist, n_hist + n_future), voltage_future, color='red', s=20, label='Predicted Voltage')
for i, val in enumerate(voltage_future):
    plt.text(n_hist + i, val + 1.2, f"{val:.1f}", ha='center', fontsize=7, rotation=45)
plt.title("Voltage Forecast")
plt.ylabel("Voltage (V)")
plt.axvline(x=n_hist - 1, color='gray', linestyle='--')
plt.legend()
plt.grid()

# Current
plt.subplot(3, 1, 2)
plt.plot(time_hist, current_hist, 'g-', label='Historical Current')
plt.scatter(np.arange(n_hist, n_hist + n_future), current_future, color='orange', s=20, label='Predicted Current')
for i, val in enumerate(current_future):
    plt.text(n_hist + i, val + 0.02, f"{val:.2f}", ha='center', fontsize=7, rotation=45)
plt.title("Current Forecast")
plt.ylabel("Current (A)")
plt.axvline(x=n_hist - 1, color='gray', linestyle='--')
plt.legend()
plt.grid()

# Power Factor
plt.subplot(3, 1, 3)
plt.plot(time_hist, pf_hist, 'm-', label='Historical PF')
plt.scatter(np.arange(n_hist, n_hist + n_future), pf_future, color='black', s=20, label='Predicted PF')
for i, val in enumerate(pf_future):
    plt.text(n_hist + i, val + 0.003, f"{val:.3f}", ha='center', fontsize=7, rotation=45)
plt.title("Power Factor Forecast")
plt.xlabel("Time (samples)")
plt.ylabel("Power Factor")
plt.axvline(x=n_hist - 1, color='gray', linestyle='--')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
