import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# --- 1. Device Configuration ---
# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Hyperparameters ---
num_epochs = 10  # Increased epochs for better training with CNN
batch_size = 64
learning_rate = 0.001

# --- 3. Data Loading and Preprocessing ---
# Define transformations for the MNIST dataset
# MNIST images are 28x28 grayscale.
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to FloatTensor and scales pixels to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize with MNIST's mean and std dev
])

# Download and load the MNIST training and test datasets
# This will automatically create a 'data' directory and download the dataset into it.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 4. CNN Model Definition ---
# Define a Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        # Input: 1 channel (grayscale), Output: 32 channels, Kernel size: 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        # Input size after two conv/pool layers: (28/2)/2 = 7 -> 7x7x64 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: 10 neurons for 10 digits (0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # -> n, 1, 28, 28 (batch_size, channels, height, width)
        x = self.pool(torch.relu(self.conv1(x))) # -> n, 32, 14, 14
        x = self.pool(torch.relu(self.conv2(x))) # -> n, 64, 7, 7
        x = x.view(-1, 64 * 7 * 7) # Flatten for fully connected layer -> n, 64*7*7
        x = torch.relu(self.fc1(x)) # -> n, 128
        x = self.fc2(x) # -> n, 10 (output logits for 10 classes)
        return x

# Instantiate the CNN model and move it to the configured device
model = CNN().to(device)
print("\nCNN Model Architecture:")
print(model)

# --- 5. Loss and Optimizer ---
criterion = nn.CrossEntropyLoss() # Standard for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. Training Loop ---
print("\nStarting Training...")
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move images and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

        total_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {total_loss / len(train_loader):.4f}")

# --- 7. Save the Trained Model ---
model_save_path = "cnn_digit_classifier.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved as {model_save_path}")

# --- 8. Testing and Evaluation ---
print("\nStarting Testing...")
model.eval() # Set model to evaluation mode
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad(): # Disable gradient calculation for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Get the predicted class (index of max logit)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# --- 9. Confusion Matrix and Classification Report ---
# Generate target names for the classification report (0-9)
target_names = [str(i) for i in range(10)]

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(all_labels, all_preds, target_names=target_names)
print("\nClassification Report (Accuracy per class, Precision, Recall, F1-Score):")
print(report)
