import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# Treatment info: (name, file, color, marker)
treatments = [
    ('Saline', 'datasets/saline85/saline85.xlsx', 'tab:blue', 'o'),
    ('Untreated', 'datasets/untreated85/untreated85.xlsx', 'darkorange', 's'),
    ('MNPS', 'datasets/mnps85/mnps85.xlsx', 'forestgreen', '^'),
    ('MNFDG', 'datasets/mnfdg85/mnfdg85.xlsx', 'firebrick', 'D'),
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

def train_with_early_stopping(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, max_epochs=500, patience=30):
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
        if patience_counter > patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss

# Learning rates to test
learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Add random seeds for averaging, as in LSTM script
random_seeds = [0, 1, 2, 3, 4]

plt.figure(figsize=(12, 8))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for name, path, color, marker in treatments:
    print(f"\nProcessing {name}...")
    
    # Read data
    df = pd.read_excel(path)
    X = df['var1'].values.reshape(-1, 1)
    y = df['var2'].values.reshape(-1, 1)
    
    # Make lagged features (using fixed parameters for consistency)
    lag = 6
    X_lagged, y_lagged = make_lagged(X, y, lag=lag)
    n_samples = X_lagged.shape[0]
    if n_samples < 20:
        print(f"  Skipping {name}: insufficient data after lagging")
        continue
    
    # Contiguous time series split: first 75% train, last 25% test
    split_idx = int(0.75 * n_samples)
    X_train = X_lagged[:split_idx]
    y_train = y_lagged[:split_idx]
    X_test = X_lagged[split_idx:]
    y_test = y_lagged[split_idx:]
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train).flatten()
    y_test_scaled = scaler_y.transform(y_test).flatten()
    
    # Validation split from train
    val_size = int(0.2 * X_train_scaled.shape[0])
    if val_size < 2:
        print(f"  Skipping {name}: insufficient validation data")
        continue
    X_val = X_train_scaled[:val_size]
    y_val = y_train_scaled[:val_size]
    X_train2 = X_train_scaled[val_size:]
    y_train2 = y_train_scaled[val_size:]
    
    mean_mses = []
    median_mses = []
    for lr in learning_rates:
        mses = []
        print(f"  Testing learning rate: {lr}")
        for seed in random_seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Convert to tensors
            X_train_t = torch.tensor(X_train2, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train2, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)
            # Create model with fixed architecture
            model = NeuralODERegressor(
                dim=X_train.shape[1], 
                depth=2, 
                hidden1=64, 
                hidden2=32, 
                hidden3=16, 
                dropout=0.1, 
                solver='rk4'
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            loss_fn = nn.MSELoss()
            # Train
            model, val_loss = train_with_early_stopping(
                model, optimizer, loss_fn, X_train_t, y_train_t, X_val_t, y_val_t
            )
            # Test
            model.eval()
            with torch.no_grad():
                preds_test = model(X_test_t).cpu().numpy()
            preds_test_inv = scaler_y.inverse_transform(preds_test.reshape(-1, 1)).flatten()
            y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_test_inv, preds_test_inv)
            mses.append(mse)
            print(f"    Seed {seed}: MSE={mse:.4f}")
        mean_mses.append(np.mean(mses))
        median_mses.append(np.median(mses))
        print(f"  [Summary] LR={lr:.1e} Mean MSE={np.mean(mses):.4f}, Median MSE={np.median(mses):.4f}")
    # Smoothing/interpolation for presentation (optional, as in LSTM)
    from scipy.interpolate import make_interp_spline
    lrs_log = np.log10(learning_rates)
    lrs_smooth = np.logspace(np.log10(learning_rates[0]), np.log10(learning_rates[-1]), 200)
    mean_spline = make_interp_spline(lrs_log, mean_mses, k=2)
    median_spline = make_interp_spline(lrs_log, median_mses, k=2)
    plt.plot(lrs_smooth, mean_spline(np.log10(lrs_smooth)), color=color, label=f"{name} Mean")
    plt.plot(lrs_smooth, median_spline(np.log10(lrs_smooth)), color=color, linestyle='--', label=f"{name} Median")
    # Also plot original points
    plt.plot(learning_rates, mean_mses, color=color, marker=marker, linestyle='', markersize=8)
    plt.plot(learning_rates, median_mses, color=color, marker=marker, linestyle=':', markersize=8)

plt.xscale('log')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Test MSE', fontsize=12)
plt.title('Neural ODE Performance vs Learning Rate\n(Fixed Test Set, Mean/Median, Smoothed)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('result/neural_ode_mse_vs_lr.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nNeural ODE MSE vs Learning Rate plot saved as 'result/neural_ode_mse_vs_lr.png'") 