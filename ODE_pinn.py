import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
device="cuda"
class pinn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(1,4096),
            nn.Tanh(),
            nn.Linear(4096,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
    def forward(self,x):
        return self.net(x)    
x_train= torch.linspace(0,2,10000).view(-1,1).requires_grad_(True).to(device)
model = pinn().to(device)
optimizer = optim.Adam(model.parameters(),lr = 3e-6)  
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train).to(device)
    dy_dx = torch.autograd.grad(y_pred, x_train, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    # ODE Residual Loss (Step 1 & 8)
    residual = dy_dx + 15* y_pred - 15* torch.sin(x_train) - torch.cos(x_train)
    loss_ode = torch.mean(residual**2)
    # Initial Condition Loss (Step 2 & 8)
    x0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    y0_pred = model(x0)
    loss_ic = torch.pow(y0_pred - 0, 2)
    
    # Total Loss
    total_loss = loss_ode + loss_ic
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {total_loss.item():.6f}')


import matplotlib.pyplot as plt
import numpy as np
model.eval()
x_test = torch.linspace(0, 2, 100).view(-1, 1).to(device)
with torch.no_grad():
    y_pinn = model(x_test).cpu().numpy()

x_plot = x_test.cpu().numpy()
y_true = np.sin(x_plot)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_true, label="Exact Solution (sin(x))", color='red', linestyle='dashed')
plt.plot(x_plot, y_pinn, label="PINN Prediction", color='blue', alpha=0.7)
plt.title("PINN vs Analytical Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()