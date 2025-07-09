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
    all_ngons = sorted({int(k) for d in all_data for k in d.keys()})
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

def train_neural_ode(tensor_data, t_torch, layer_len, epochs=500, lr=0.01):
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

data_files = glob.glob('ngon_counts_per_cut*.json')  # or your pattern

models = []
for i, fname in enumerate(data_files):
    print(f"\nTraining model {i+1} on file: {fname}")
    tensor_data, t_torch, layer_len, all_ngons, data_scale, tscale = load_ngon_data([fname])
    model = train_neural_ode(tensor_data, t_torch, layer_len)
    models.append(model)

    # Evaluate the trained model
    y_pred = model(tensor_data[0, :], t_torch).detach().numpy()
    plt.figure(figsize=(10, 5))
    cmap = cm.get_cmap('tab10', layer_len)
    actual_fractions = tensor_data.numpy() / tensor_data.numpy().sum(axis=1, keepdims=True)
    pred_fractions = y_pred / y_pred.sum(axis=1, keepdims=True)
    for j in range(layer_len):
        color = cmap(j)
        plt.plot(t_torch.numpy(), actual_fractions[:, j], color=color, label=f'{j+3}-gons')
        plt.plot(t_torch.numpy(), pred_fractions[:, j], '--', color=color, label=f'Predicted {j+3}-gons')
    plt.xlabel('Cuts')
    plt.ylabel('Fraction of n-gons')
    plt.legend()
    plt.title(f'Fraction of Each n-gon: Target vs Predicted (File {i+1})')
    plt.show()

# Make a graph showing how the models made perform on a validation set