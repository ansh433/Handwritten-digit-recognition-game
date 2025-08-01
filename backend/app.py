import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
from waitress import serve # Import waitress here

# --- 1. Define the CNN Model Architecture ---
# This must be identical to the architecture used during training
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Load the Trained Model ---
MODEL_PATH = "cnn_digit_classifier.pth"
device = torch.device("cpu")

model = CNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"Model '{MODEL_PATH}' loaded successfully on {device}.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory as this script.")
    # In a real production environment, you might want to log this error and exit more gracefully
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. Define Image Preprocessing for Inference ---
transform_inference = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- 4. Flask App Setup ---
app = Flask(__name__)
# Set debug to False for production
app.config['DEBUG'] = False
CORS(app) # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(binary_data))
        input_tensor = transform_inference(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor.to(device))
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# --- 5. Production Server Setup with Waitress ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting Waitress server on {host}:{port}...")
    # This is the correct way to serve with Waitress
    serve(app, host=host, port=port)