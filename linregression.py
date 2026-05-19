# a linear regression model training & testing

# 1st step: preparing data y = X * W + b
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

bias = 0.4
weights = 0.7
X = torch.arange(0, 1, 0.01).unsqueeze(dim=1)
y = X * weights + bias


# 2nd step: spliting training & test sets (80/20%)
train_split = int(0.8 * len(X)) # 80
test_split = int(0.2 * len(X)) # 20
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[-test_split:], y[-test_split:]

# moving sets to gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_X_train = X_train.to(device)
gpu_y_train = y_train.to(device)
gpu_X_test = X_test.to(device)
gpu_y_test = y_test.to(device)


# 3rd step: creating the model/class
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # creating model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    # forwards passs aka computations in model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
model_1 = LinearRegression()
# since print(next(model_1.parameters()).device) is on cpu, we should move it to gpu
model_1.to(device)
# print(next(model_1.parameters()).device) # now it's on gpu!


# 4th step: defining loss & optimizer
loss_fn = nn.L1Loss()
optimizier = torch.optim.SGD(params=model_1.parameters(), lr=0.01)


# 5th step: loops
torch.manual_seed(42)
epochs = 1000
for epoch in range(epochs):
    # putting model into train mode
    model_1.train()
    # performing forward pass on training data
    y_pred = model_1(gpu_X_train)
    # calculating the loss
    loss = loss_fn(y_pred, gpu_y_train)
    # zero grads
    optimizier.zero_grad()
    # backpropagation
    loss.backward()
    # gradient descent/model update
    optimizier.step()

    # testing loop now
    # evaluation mode
    model_1.eval()
    # forward pass
    with torch.inference_mode():
        test_pred = model_1(gpu_X_test)
        # calculating test loss
        test_loss = loss_fn(test_pred, gpu_y_test)

    if epoch % 100 == 0: # REMEMBER: this if-statement must be inside the loop
        print(f"epoch {epoch}: train loss = {loss} | test loss = {test_loss}")


# visualization
model_1.cpu()
with torch.inference_mode():
    preds = model_1(X).squeeze()
plt.scatter(X, y, label="True data")
plt.plot(X, preds, label="Model prediction", color="red")
plt.legend()
plt.show()

    

