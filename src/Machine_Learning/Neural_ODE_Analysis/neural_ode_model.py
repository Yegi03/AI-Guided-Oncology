import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import itertools
import csv
import os

# Treatment info: (name, file, color, marker)
treatments = [
    ('Saline', 'datasets/saline85/saline85.xlsx', 'tab:blue', 'o', 's', 'D'),
    ('Untreated', 'datasets/untreated85/untreated85.xlsx', 'darkorange', 'o', 's', 'D'),
    ('MNPS', 'datasets/mnps85/mnps85.xlsx', 'forestgreen', 'o', 's', 'D'),
    ('MNFDG', 'datasets/mnfdg85/mnfdg85.xlsx', 'firebrick', 'o', 's', 'D'),
]

def make_lagged(X, y, lag=6):
    X_lagged, y_lagged = [], []
    for i in range(lag, len(X)):
        X_lagged.append(np.concatenate([X[i-lag:i].flatten(), X[i]]))
        y_lagged.append(y[i])
    return np.array(X_lagged), np.array(y_lagged)

class ODEFunc(nn.Module):
    def __init__(self, dim, depth=2, hidden1=32, hidden2=16, hidden3=16, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(dim, hidden1), nn.Tanh(), nn.Dropout(dropout)]
        last_size = hidden1
        if depth > 1:
            layers += [nn.Linear(hidden1, hidden2), nn.Tanh(), nn.Dropout(dropout)]
            last_size = hidden2
        if depth > 2:
            layers += [nn.Linear(hidden2, hidden3), nn.Tanh(), nn.Dropout(dropout)]
            last_size = hidden3
        if depth > 3:
            layers += [nn.Linear(hidden3, hidden3), nn.Tanh()]
            last_size = hidden3

        layers += [nn.Linear(last_size, dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, t, x):
        return self.net(x)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, solver='rk4', rtol=1e-5, atol=1e-7):
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
    def forward(self, x):
        t = torch.tensor([0, 1], dtype=torch.float32).to(x.device)
        out = odeint(self.odefunc, x, t, method=self.solver, rtol=self.rtol, atol=self.atol)
        return out[1]

class NeuralODERegressor(nn.Module):
    def __init__(self, dim, depth=2, hidden1=32, hidden2=16, hidden3=16, dropout=0.1, solver='rk4', rtol=1e-5, atol=1e-7):
        super().__init__()
        self.odeblock = ODEBlock(ODEFunc(dim, depth, hidden1, hidden2, hidden3, dropout), solver=solver, rtol=rtol, atol=atol)
        self.readout = nn.Linear(dim, 1)
    def forward(self, x):
        x = self.odeblock(x)
        return self.readout(x).squeeze(-1)

def train_with_early_stopping(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, max_epochs=1000, patience=50):
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss

plt.figure(figsize=(12,8))
legend_handles = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter grid
lags = [6, 8, 10]
depths = [2, 3, 4]
hidden1s = [64, 128]
hidden2s = [32, 64]
hidden3s = [16, 32]
learning_rates = [1e-4, 5e-5]
solvers = ['rk4', 'dopri5']

results = []

for name, path, color, train_marker, test_marker, pred_marker in treatments:
    print(f"\nTuning {name}...")
    df = pd.read_excel(path)
    X = df['var1'].values.reshape(-1, 1)
    y = df['var2'].values.reshape(-1, 1)
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    best_scaler_X = None
    best_scaler_y = None
    best_X_train = best_X_test = best_y_train = best_y_test = None
    best_X_train_time = best_X_test_time = None
    for lag, depth, hidden1, hidden2, hidden3, lr, solver in itertools.product(lags, depths, hidden1s, hidden2s, hidden3s, learning_rates, solvers):
        # Make lagged features
        X_lagged, y_lagged = make_lagged(X, y, lag=lag)
        if len(X_lagged) < 10:
            continue  # skip if not enough data
        X_train, X_test, y_train, y_test = train_test_split(X_lagged, y_lagged, train_size=0.75, random_state=42)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train).flatten()
        y_test_scaled = scaler_y.transform(y_test).flatten()
        # Validation split from train
        val_size = int(0.2 * X_train_scaled.shape[0])
        if val_size < 2:
            continue
        X_val = X_train_scaled[:val_size]
        y_val = y_train_scaled[:val_size]
        X_train2 = X_train_scaled[val_size:]
        y_train2 = y_train_scaled[val_size:]
        X_train_t = torch.tensor(X_train2, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train2, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        model = NeuralODERegressor(dim=X_train.shape[1], depth=depth, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, dropout=0.1, solver=solver).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        model, val_loss = train_with_early_stopping(model, optimizer, loss_fn, X_train_t, y_train_t, X_val_t, y_val_t)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = (lag, depth, hidden1, hidden2, hidden3, lr, solver)
            best_model = model
            best_scaler_X = scaler_X
            best_scaler_y = scaler_y
            best_X_train = X_train_scaled
            best_X_test = X_test_scaled
            best_y_train = y_train_scaled
            best_y_test = y_test_scaled
            best_X_train_time = X_train[:, -1]
            best_X_test_time = X_test[:, -1]
    # Retrain on full train set with best config
    print(f"Best config for {name}: lag={best_config[0]}, depth={best_config[1]}, hidden1={best_config[2]}, hidden2={best_config[3]}, hidden3={best_config[4]}, lr={best_config[5]}, solver={best_config[6]}")
    print(f"Best val loss: {best_val_loss:.4f}")
    # Use best_model and scalers to predict
    X_train_t = torch.tensor(best_X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(best_y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(best_X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(best_y_test, dtype=torch.float32).to(device)
    best_model.eval()
    with torch.no_grad():
        preds_train = best_model(X_train_t).cpu().numpy()
        preds_test = best_model(X_test_t).cpu().numpy()
    preds_train_inv = best_scaler_y.inverse_transform(preds_train.reshape(-1, 1)).flatten()
    preds_test_inv = best_scaler_y.inverse_transform(preds_test.reshape(-1, 1)).flatten()
    y_train_inv = best_scaler_y.inverse_transform(best_y_train.reshape(-1, 1)).flatten()
    y_test_inv = best_scaler_y.inverse_transform(best_y_test.reshape(-1, 1)).flatten()
    # Save results for CSV
    for t, actual, pred in zip(best_X_train_time, y_train_inv, preds_train_inv):
        results.append([name, 'train', t, actual, pred])
    for t, actual, pred in zip(best_X_test_time, y_test_inv, preds_test_inv):
        results.append([name, 'test', t, actual, pred])
    mse = mean_squared_error(y_test_inv, preds_test_inv)
    print(f'{name}: Neural ODE Test MSE = {mse:.2f}')
    # Plot
    h1 = plt.scatter(best_X_train_time, y_train_inv, color=color, marker=train_marker, s=70, alpha=0.7, label=f'{name} Train (Actual)')
    h2 = plt.scatter(best_X_train_time, preds_train_inv, color='black', marker='x', s=70, label=f'{name} Train (Pred)')
    h3 = plt.scatter(best_X_test_time, y_test_inv, color=color, marker=test_marker, s=70, alpha=0.7, label=f'{name} Test (Actual)')
    if pred_marker == 'D':
        h4 = plt.scatter(best_X_test_time, preds_test_inv, facecolor=color, edgecolor='black', linewidth=2, marker=pred_marker, s=70, label=f'{name} Test (Pred)')
    else:
        h4 = plt.scatter(best_X_test_time, preds_test_inv, color=color, marker=pred_marker, s=70, edgecolor='black', label=f'{name} Test (Pred)')
    legend_handles.extend([h1, h2, h3, h4])

plt.xlabel('Time (days)')
plt.ylabel('Cancer Volume (mm³)')
plt.title('Neural ODE Regression Comparison: All Treatments (Auto-tuned, 75/25 Split, Lagged)')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=11, loc='best', ncol=2)
plt.tight_layout()
plt.savefig('result/neural_ode_all_treatments_comparison.png', dpi=300)

# Save predictions to CSV
os.makedirs('result', exist_ok=True)
with open('result/neural_ode_all_treatments_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['treatment', 'set', 'time', 'actual', 'predicted'])
    writer.writerows(results) 