import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim

'''
Make a set of training data and a set of testing data
Plot the graph from notebook
'''


with open('ngon_counts_per_cut.json', 'r') as f:
    data=json.load(f)

#sorting the json data by number of sides
all_ngons = sorted({int(k) for d in data for k in d.keys()})
json_len=len(data)
tensor_data = []
t = np.linspace(0, json_len, json_len, endpoint=False)
tscale = t.max()-t.min()
t_torch = torch.tensor(t/tscale, requires_grad=False, dtype=torch.float32)

for d in data:
    vec = [d.get(str(n), 0) for n in all_ngons]
    tensor_data.append(vec)
tensor_data=np.array(tensor_data)
data_scale=tensor_data.max()-tensor_data.min()
tensor_data = torch.tensor(tensor_data/data_scale,requires_grad=False, dtype=torch.float32)
layer_len=len(tensor_data[0])


print("Tensor shape:", tensor_data.shape)
print(tensor_data)
print("T shape:", t_torch.shape)
print(t_torch)

# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        input_size = layer_len
        hidden_size = 40
        hidden_layers = 3
        output_size = layer_len

        # This is a complicated function - it will use lists to build as many hidden layers as we asked for
        # For an explanation - ask Gemini "explain how self.net is constructed"
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) for _ in range(hidden_layers)],
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, t, y):
        return self.net(y)

# Define the Neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, y0, t):
        return odeint(self.ode_func, y0, t)
    


# Create the ODE RHS function and Neural ODE model that solves dydt = f(y)
ode_func = ODEFunc()
neural_ode = NeuralODE(ode_func)

# Define initial condition
y0 = tensor_data[0,:]

# Define a simple loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(neural_ode.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    y_pred = neural_ode(y0, t_torch)
    loss = criterion(y_pred, tensor_data)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate the trained model
y_pred = neural_ode(y0, t_torch).detach().numpy()