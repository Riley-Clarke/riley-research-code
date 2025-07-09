import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import matplotlib.cm as cm

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

'''
# Plot the results
plt.figure(figsize=(10, 5))
cmap = cm.get_cmap('tab10', layer_len)  # or use another colormap if you have more than 10 n-gons
for i in range(layer_len):
    color = cmap(i)
    plt.plot(t_torch.numpy(), (tensor_data[:, i].numpy()), color=color, label=f'{i+3}-gons')
    plt.plot(t_torch.numpy(), y_pred[:, i], '--', color=color, label=f'Predicted {i+3}-gons')

plt.xlabel('Cuts')
plt.ylabel('Values')
plt.legend()
plt.title('Comparison of Target and Predicted Values')
plt.show()
'''
# --- Training data plot ---
plt.figure(figsize=(10, 5))
cmap = cm.get_cmap('tab10', layer_len)
# Compute fractions for actual and predicted
actual_fractions = tensor_data.numpy() / tensor_data.numpy().sum(axis=1, keepdims=True)
pred_fractions = y_pred / y_pred.sum(axis=1, keepdims=True)
for i in range(layer_len):
    color = cmap(i)
    plt.plot(t_torch.numpy(), actual_fractions[:, i], color=color, label=f'{i+3}-gons')
    plt.plot(t_torch.numpy(), pred_fractions[:, i], '--', color=color, label=f'Predicted {i+3}-gons')
plt.xlabel('Cuts')
plt.ylabel('Fraction of n-gons')
plt.legend()
plt.title('Fraction of Each n-gon: Target vs Predicted')
plt.show()

with open('ngon_counts_per_cut_TESTING.json', 'r') as f:
    test_data = json.load(f)

# Prepare test tensor (same as training)
test_tensor_data = []
for d in test_data:
    vec = [d.get(str(n), 0) for n in all_ngons]  # Use same all_ngons as training
    test_tensor_data.append(vec)
test_tensor_data = np.array(test_tensor_data)
test_tensor_data = torch.tensor(test_tensor_data/data_scale, requires_grad=False, dtype=torch.float32)
test_t = np.linspace(0, len(test_tensor_data), len(test_tensor_data), endpoint=False)
test_torch = torch.tensor(test_t/tscale, requires_grad=False, dtype=torch.float32)

# Use the first test value as initial condition
y0_test = test_tensor_data[0, :]
y_pred_test = neural_ode(y0_test, test_torch).detach().numpy()
'''
plt.figure(figsize=(10, 5))
for i in range(layer_len):
    color = cmap(i)
    plt.plot(test_torch.numpy(), test_tensor_data[:, i].numpy(), color=color, label=f'Test {i+3}-gons')
    plt.plot(test_torch.numpy(), y_pred_test[:, i], '--', color=color, label=f'Predicted {i+3}-gons')
plt.xlabel('Cuts')
plt.ylabel('Values')
plt.legend()
plt.title('Test Data: Target vs Predicted')
plt.show()
'''


# --- Test data plot ---
plt.figure(figsize=(10, 5))
test_actual_fractions = test_tensor_data.numpy() / test_tensor_data.numpy().sum(axis=1, keepdims=True)
test_pred_fractions = y_pred_test / y_pred_test.sum(axis=1, keepdims=True)
for i in range(layer_len):
    color = cmap(i)
    plt.plot(test_torch.numpy(), test_actual_fractions[:, i], color=color, label=f'Test {i+3}-gons')
    plt.plot(test_torch.numpy(), test_pred_fractions[:, i], '--', color=color, label=f'Predicted {i+3}-gons')
plt.xlabel('Cuts')
plt.ylabel('Fraction of n-gons')
plt.legend()
plt.title('Test Data: Fraction of Each n-gon')
plt.show()