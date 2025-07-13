import json
import numpy as np
import torch
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os

def load_ngon_data(json_files):
    all_data = []
    for fname in json_files:
        with open(fname, 'r') as f:
            file_data = json.load(f)
            all_data.extend(file_data)
    all_ngons = list(range(3, 11)) 
    tensor_data = []
    for d in all_data:
        vec = [d.get(str(n), 0) for n in all_ngons]
        tensor_data.append(vec)
    tensor_data = np.array(tensor_data)
    data_scale = tensor_data.max() - tensor_data.min() if tensor_data.max() != tensor_data.min() else 1.0
    tensor_data = torch.tensor(tensor_data / data_scale, requires_grad=False, dtype=torch.float32)
    t = np.linspace(0, len(tensor_data), len(tensor_data), endpoint=False)
    tscale = t.max() - t.min() if t.max() != t.min() else 1.0
    t_torch = torch.tensor(t / tscale, requires_grad=False, dtype=torch.float32)
    return tensor_data, t_torch, len(all_ngons), all_ngons, data_scale, tscale

class ODEFunc(nn.Module):
    def __init__(self, layer_len, hidden_size=40, hidden_layers=3):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(layer_len, hidden_size),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) for _ in range(hidden_layers)],
            nn.Linear(hidden_size, layer_len)
        )
    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
    def forward(self, y0, t):
        return odeint(self.ode_func, y0, t)

def train_neural_ode(tensor_data, t_torch, layer_len, epochs=200, lr=0.01):
    ode_func = ODEFunc(layer_len)
    neural_ode = NeuralODE(ode_func)
    y0 = tensor_data[0, :]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(neural_ode.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = neural_ode(y0, t_torch)
        loss = criterion(y_pred, tensor_data)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return neural_ode

data_files = glob.glob('./neural_data/ngon_counts_per_cut*.json')  # or your pattern

models = []
for i, fname in enumerate(data_files):
    save_path='./nn_models/'+fname[14:-5]+'.pt'
    if os.path.exists(save_path):
        tensor_data, t_torch, layer_len, all_ngons, data_scale, tscale = load_ngon_data([fname])
        print("Loading saved model: ",save_path)
        ode_func = ODEFunc(layer_len)
        model = NeuralODE(ode_func)
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.eval()
        models.append(model)
    else:
        print(f"\nTraining model {i+1} on file: {fname}")
        
        tensor_data, t_torch, layer_len, all_ngons, data_scale, tscale = load_ngon_data([fname])
        model = train_neural_ode(tensor_data, t_torch, layer_len)
        models.append(model)
        
        # Evaluate the trained model
        y_pred = model(tensor_data[0, :], t_torch).detach().numpy()
        #plt.figure(figsize=(10, 5))
        #cmap = plt.get_cmap('tab10', layer_len)
        actual_fractions = tensor_data.numpy() / tensor_data.numpy().sum(axis=1, keepdims=True)
        pred_fractions = y_pred / y_pred.sum(axis=1, keepdims=True)
        torch.save(model.state_dict(), save_path)
        '''
        for j in range(layer_len):
            color = cmap(j)
            plt.plot(t_torch.numpy(), actual_fractions[:, j], color=color, label=f'{j+3}-gons')
            plt.plot(t_torch.numpy(), pred_fractions[:, j], '--', color=color, label=f'Predicted {j+3}-gons')
        plt.xlabel('Cuts')
        plt.ylabel('Fraction of n-gons')
        plt.legend()
        plt.title(f'Fraction of Each n-gon: Target vs Predicted (File {i+1})')
        plt.show()
    '''

# Load testing data (use the same all_ngons as in training for consistency)
test_file = './neural_data/ngon_counts_TESTING.json'
with open(test_file, 'r') as f:
    test_data = json.load(f)

# Use the all_ngons from the first model for consistent ordering
_, _, _, all_ngons, data_scale, tscale = load_ngon_data([data_files[0]])
test_tensor_data = []
for d in test_data:
    vec = [d.get(str(n), 0) for n in all_ngons]
    test_tensor_data.append(vec)
test_tensor_data = np.array(test_tensor_data)
test_tensor_data = torch.tensor(test_tensor_data / data_scale, requires_grad=False, dtype=torch.float32)
test_t = np.linspace(0, len(test_tensor_data), len(test_tensor_data), endpoint=False)
test_torch = torch.tensor(test_t / tscale, requires_grad=False, dtype=torch.float32)
layer_len = len(all_ngons)

# Plot predictions from all models on the same graph
plt.figure(figsize=(12, 6))
cmap =plt.get_cmap('tab10', layer_len)
for model_idx, model in enumerate(models):
    y0_test = test_tensor_data[0, :]
    y_pred_test = model(y0_test, test_torch).detach().numpy()
    pred_fractions = y_pred_test / y_pred_test.sum(axis=1, keepdims=True)
    for j in range(layer_len):
        color = cmap(j)
        plt.plot(
            test_torch.numpy(),
            pred_fractions[:, j],
            linestyle='--',
            color=color,
            alpha=0.25,  # Slightly different alpha for each model
            label=f'Model {model_idx+1} Pred {j+3}-gons' if model_idx == 0 else None  # Only label once per n-gon
        )

# Plot actual test data fractions for reference
actual_fractions = test_tensor_data.numpy() / test_tensor_data.numpy().sum(axis=1, keepdims=True)
for j in range(layer_len):
    color = cmap(j)
    plt.plot(
        test_torch.numpy(),
        actual_fractions[:, j],
        color=color,
        linewidth=2,
        label=f'Actual {j+3}-gons'
    )

# Collect all model predictions for averaging
all_model_preds = []
for model in models:
    y0_test = test_tensor_data[0, :]
    y_pred_test = model(y0_test, test_torch).detach().numpy()
    pred_fractions = y_pred_test / y_pred_test.sum(axis=1, keepdims=True)
    all_model_preds.append(pred_fractions)

# Convert to numpy array: shape (num_models, num_timesteps, num_ngon_types)
all_model_preds = np.array(all_model_preds)

# Compute the average across models (axis=0 is models)
avg_pred_fractions = np.mean(all_model_preds, axis=0)

# Plot the average predictions
for j in range(layer_len):
    color = cmap(j)
    plt.plot(
        test_torch.numpy(),
        avg_pred_fractions[:, j],
        linestyle='-',
        color=color,
        linewidth=3,
        label=f'Average Pred {j+3}-gons'
    )


plt.xlabel('Cuts')
plt.ylabel('Fraction of n-gons')
plt.title('Model Predictions on Test Data (All Models)')
#plt.legend()
#plt.show()

# Only plot 3, 4, and 5-gons
desired_ngons = [3, 4, 5]
desired_indices = [all_ngons.index(n) for n in desired_ngons if n in all_ngons]

plt.figure(figsize=(12, 6))
cmap = plt.get_cmap('tab10', len(desired_indices))

# Plot predictions from all models (faint lines)
for model_idx, model in enumerate(models):
    y0_test = test_tensor_data[0, :]
    y_pred_test = model(y0_test, test_torch).detach().numpy()
    pred_fractions = y_pred_test / y_pred_test.sum(axis=1, keepdims=True)
    for plot_idx, j in enumerate(desired_indices):
        color = cmap(plot_idx)
        plt.plot(
            test_torch.numpy(),
            pred_fractions[:, j],
            linestyle='--',
            color=color,
            alpha=0.25,
            label=f'Model {model_idx+1} Pred {all_ngons[j]}-gons' if model_idx == 0 else None
        )

# Plot actual test data fractions for reference
actual_fractions = test_tensor_data.numpy() / test_tensor_data.numpy().sum(axis=1, keepdims=True)
for plot_idx, j in enumerate(desired_indices):
    color = cmap(plot_idx)
    plt.plot(
        test_torch.numpy(),
        actual_fractions[:, j],
        color=color,
        linewidth=2,
        label=f'Actual {all_ngons[j]}-gons'
    )

# Plot the average predictions
avg_pred_fractions = np.mean(all_model_preds, axis=0)
for plot_idx, j in enumerate(desired_indices):
    color = cmap(plot_idx)
    plt.plot(
        test_torch.numpy(),
        avg_pred_fractions[:, j],
        linestyle='-',
        color=color,
        linewidth=3,
        label=f'Average Pred {all_ngons[j]}-gons'
    )

plt.xlabel('Cuts')
plt.ylabel('Fraction of n-gons')
plt.title('Model Predictions on Test Data (3, 4, 5-gons)')
plt.legend()
plt.show()