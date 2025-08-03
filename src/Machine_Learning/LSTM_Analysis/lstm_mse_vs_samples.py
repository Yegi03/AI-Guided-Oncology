import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import make_interp_spline

# Treatment info: (name, file, color, marker, max_samples)
treatments = [
    ('Saline', 'datasets/saline85/saline85.xlsx', 'tab:blue', 'o', 85),
    ('Untreated', 'datasets/untreated85/untreated85.xlsx', 'darkorange', 's', 88),
    ('MNPS', 'datasets/mnps85/mnps85.xlsx', 'forestgreen', '^', 86),
    ('MNFDG', 'datasets/mnfdg85/mnfdg85.xlsx', 'firebrick', 'D', 90),
]

# Sample sizes to test (up to max for each treatment)
def get_sample_sizes(max_n):
    if max_n < 20:
        return [max_n]
    step = max(1, (max_n-20)//3)
    sizes = list(range(20, max_n, step))
    if max_n not in sizes:
        sizes.append(max_n)
    return sizes

plt.figure(figsize=(12, 8))

for name, path, color, marker, max_n in treatments:
    print(f"\nProcessing {name}...")
    df = pd.read_excel(path)
    X = df['var1'].values.reshape(-1, 1)
    y = df['var2'].values
    sample_sizes = get_sample_sizes(max_n)
    mean_mses = []
    median_mses = []
    for n_samples in sample_sizes:
        if n_samples > len(X):
            continue
        # Fixed test set: last 20% of the data
        test_size = max(5, int(0.2 * len(X)))
        X_train = X[:n_samples]
        y_train = y[:n_samples]
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        mses = []
        for seed in range(5):
            np.random.seed(seed)
            tf.random.set_seed(seed)
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
            model.fit(X_train_lstm, y_train_scaled, epochs=1000, verbose=0)
            preds_test_scaled = model.predict(X_test_lstm).flatten()
            preds_test = scaler_y.inverse_transform(preds_test_scaled.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_test, preds_test)
            mses.append(mse)
        mean_mses.append(np.mean(mses))
        median_mses.append(np.median(mses))
    # Interpolate for smooth curve
    if len(sample_sizes) > 2:
        xnew = np.linspace(min(sample_sizes), max(sample_sizes), 200)
        mean_spline = make_interp_spline(sample_sizes, mean_mses, k=2)(xnew)
        median_spline = make_interp_spline(sample_sizes, median_mses, k=2)(xnew)
        plt.plot(xnew, mean_spline, color=color, linestyle='-', label=f'{name} Mean')
        plt.plot(xnew, median_spline, color=color, linestyle='--', label=f'{name} Median')
    else:
        plt.plot(sample_sizes, mean_mses, color=color, linestyle='-', label=f'{name} Mean')
        plt.plot(sample_sizes, median_mses, color=color, linestyle='--', label=f'{name} Median')
    plt.scatter(sample_sizes, mean_mses, color=color, marker=marker, s=70)

plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Test MSE (Fixed Test Set)', fontsize=12)
plt.title('LSTM Performance vs Training Set Size\n(Fixed Test Set, Mean/Median, Smoothed)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('result/lstm_mse_vs_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nLSTM MSE vs Samples plot saved as 'result/lstm_mse_vs_samples.png'") 