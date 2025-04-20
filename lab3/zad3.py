import torch
import pandas as pd
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_file = "train.csv"
test_file = "test.csv"

train_df = pd.read_csv(train_file)
train_df.dropna(inplace=True)
test_df = pd.read_csv(test_file)

X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y'].values

X_test = test_df['x'].values.reshape(-1, 1)
y_test = test_df['y'].values 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel().to(device)

loss_fn = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

epochs = 2000
for t in range(epochs + 1):
    model.train()

    y_pred = model(X_train_tensor)

    loss = loss_fn(y_pred, y_train_tensor)
    
    if t % 100 == 0:
        print(f'Epoch [{t}/{epochs}], Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 500 == 0:
        a, b = model.linear.bias.item(), model.linear.weight.item()
        y_graph = a + b * X_train

        plt.scatter(X_train, y_train, label='Original Data')
        plt.plot(X_train, y_graph, '-r', label=f'Prediction at t={t}')
        plt.title('Linear Regression - y = a + bx')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

a, b = model.linear.bias.item(), model.linear.weight.item()
print(f'Rezultat: y = {a:.2f} + {b:.2f} * x')

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, y_test_pred.cpu().numpy(), color='red', label='Fitted Line')
plt.title('Test Data and Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()