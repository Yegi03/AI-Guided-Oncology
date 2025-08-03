import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import csv
import os

# Treatment info: (name, file, color, marker)
treatments = [
    ('Saline', 'datasets/saline85/saline85.xlsx', 'tab:blue', 'o', 's', 'D'),
    ('Untreated', 'datasets/untreated85/untreated85.xlsx', 'darkorange', 'o', 's', 'D'),
    ('MNPS', 'datasets/mnps85/mnps85.xlsx', 'forestgreen', 'o', 's', 'D'),
    ('MNFDG', 'datasets/mnfdg85/mnfdg85.xlsx', 'firebrick', 'o', 's', 'D'),
]

plt.figure(figsize=(12,8))
legend_handles = []

results = []

for name, path, color, train_marker, test_marker, pred_marker in treatments:
    # Read data
    df = pd.read_excel(path)
    X = df['var1'].values.reshape(-1, 1)
    y = df['var2'].values
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)
    # Standardize X
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    # Standardize y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    # LSTM expects 3D input: (samples, timesteps, features)
    X_train_lstm = X_train_scaled.reshape(-1, 1, 1)
    X_test_lstm = X_test_scaled.reshape(-1, 1, 1)
    # Build LSTM
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=(1, 1), return_sequences=True),
        LSTM(64, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train_lstm, y_train_scaled, epochs=500, verbose=0)
    # Predict
    preds_train_scaled = model.predict(X_train_lstm).flatten()
    preds_test_scaled = model.predict(X_test_lstm).flatten()
    preds_train = scaler_y.inverse_transform(preds_train_scaled.reshape(-1, 1)).flatten()
    preds_test = scaler_y.inverse_transform(preds_test_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test, preds_test)
    print(f'{name}: LSTM Test MSE = {mse:.2f}')
    # Save results for CSV
    for t, actual, pred in zip(X_train.flatten(), y_train, preds_train):
        results.append([name, 'train', t, actual, pred])
    for t, actual, pred in zip(X_test.flatten(), y_test, preds_test):
        results.append([name, 'test', t, actual, pred])
    # Plot
    h1 = plt.scatter(X_train, y_train, color=color, marker=train_marker, s=70, alpha=0.7, label=f'{name} Train (Actual)')
    h2 = plt.scatter(X_train, preds_train, color='black', marker='x', s=70, label=f'{name} Train (Pred)')
    h3 = plt.scatter(X_test, y_test, color=color, marker=test_marker, s=70, alpha=0.7, label=f'{name} Test (Actual)')
    if pred_marker == 'D':
        h4 = plt.scatter(X_test, preds_test, facecolor=color, edgecolor='black', linewidth=2, marker=pred_marker, s=70, label=f'{name} Test (Pred)')
    else:
        h4 = plt.scatter(X_test, preds_test, color=color, marker=pred_marker, s=70, edgecolor='black', label=f'{name} Test (Pred)')
    legend_handles.extend([h1, h2, h3, h4])

plt.xlabel('Time (days)')
plt.ylabel('Cancer Volume (mm³)')
plt.title('LSTM Regression Comparison: All Treatments (75/25 Split)')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=11, loc='best', ncol=2)
plt.tight_layout()
plt.savefig('result/lstm_all_treatments_comparison.png', dpi=300)

# Save predictions to CSV
os.makedirs('result', exist_ok=True)
with open('result/lstm_all_treatments_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['treatment', 'set', 'time', 'actual', 'predicted'])
    writer.writerows(results) 