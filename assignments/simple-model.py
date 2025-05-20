# In this assignment, you will need to create a simple model from data processing to inference
# In order to train a model, we will need first to prepare the data.

# You can find the data inside of data/Housing.csv, please refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html to load the data
# You are expected to have basic knowledge about pytorch, you can find turorial here: https://docs.pytorch.org/tutorials/

import torch
import pandas as pd
import torch.nn as nn
import os

data = pd.read_csv("data/Housing.csv")

# First of all, you will need to understand the data, note the prices is what we want to predict
# Some data not align to number format, which computer wont know how to train a string, we need cast to a number.
yes_no_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
data[yes_no_columns] = data[yes_no_columns].replace({'yes': 1, 'no': 0})

furnishingstatus_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
data['furnishingstatus'] = data['furnishingstatus'].map(furnishingstatus_map)

print(data.head(n=10))

# Now we have solid data, we need to split the data into training and testing data
# We will use 80% of the data for training and 20% for testing using pandas

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Prepare features and target
target_col = 'price'
feature_cols = [col for col in data.columns if col != target_col]

X_train = train_data[feature_cols].values
y_train = train_data[target_col].values
X_test = test_data[feature_cols].values
y_test = test_data[target_col].values

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.simple_nn(x)

model = SimpleNN(input_dim=X_train.shape[1])

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 100000

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss = torch.log(loss)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# while we are trying the model, we can observe that the loss is decreasing, which is good.
# However, at some point, the loss will start to increase, which is the time we may want to stop the train.

# Save the model to assignments/output folder
os.makedirs('assignments/output', exist_ok=True)
torch.save(model.state_dict(), 'assignments/output/simple_nn.pth')

# Test the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")


