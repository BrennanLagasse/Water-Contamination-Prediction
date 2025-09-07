import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np

# ===== Load & Clean Data =====
merged = pd.read_csv("as_predictors_us.csv")

# Fill missing values in features (you could use mean imputation instead of 0)
merged = merged.fillna(0)

# Labels: binary classification for arsenic > 10 µg/L
y = (merged['As_ppb'] > 10).astype(int)

# Only keep the desired columns for X
X = merged[['pet', 'precip', 'aridity', 'temp', 'aet',
            'alpha_Priestley_Taylor', 'twi', 'clay_topsoil',
            'pH_subsoil', 'sand_subsoil', 'fluvisols']].fillna(0)


# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Check labels are valid
print("Unique train labels:", y_train_tensor.unique())
print("Unique test labels:", y_test_tensor.unique())

# Check for NaNs in tensors
print("NaNs in train features:", torch.isnan(X_train_tensor).any().item())
print("NaNs in test features:", torch.isnan(X_test_tensor).any().item())

# Create dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Finished preprocessing")

# ===== Model =====
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)  # No sigmoid here
        return x

input_size = X_train_scaled.shape[1]
model = BinaryClassifier(input_size)

# ===== Loss & Optimizer =====
criterion = nn.BCEWithLogitsLoss()  # handles sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Finished creating model")

# ===== Training Loop =====
num_epochs = 100
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Testing
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
# ===== Plot Results =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(train_losses, label='Training Loss')
ax1.plot(test_losses, label='Testing Loss')
ax1.set_title('Loss Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accuracies, label='Training Accuracy')
ax2.plot(test_accuracies, label='Testing Accuracy')
ax2.set_title('Accuracy Over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.show()