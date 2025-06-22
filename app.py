from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Global model variable
model = None

def load_model():
    global model
    try:
        model = MNIST_CNN()
        # Load the model weights when the .pth file is available
        if os.path.exists('mnist_model.pth'):
            model.load_state_dict(torch.load('mnist_model.pth', map_location='cpu'))
            model.eval()
            print("Model loaded successfully!")
        else:
            print("Model file not found. Please upload 'mnist_model.pth' to the backend directory.")
    except Exception as e:
        print(f"Error loading model: {e}")

def generate_digit_image(digit, noise_factor=0.1):
    """Generate a synthetic digit image with some randomness"""
    # Create a base template for each digit
    digit_templates = {
        0: np.array([[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]]),
        1: np.array([[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]]),
        2: np.array([[0,1,1,1,0], [1,0,0,0,1], [0,0,1,1,0], [0,1,0,0,0], [1,1,1,1,1]]),
        3: np.array([[1,1,1,1,0], [0,0,0,0,1], [0,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]]),
        4: np.array([[1,0,0,1,0], [1,0,0,1,0], [1,1,1,1,1], [0,0,0,1,0], [0,0,0,1,0]]),
        5: np.array([[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]]),
        6: np.array([[0,1,1,1,0], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0]]),
        7: np.array([[1,1,1,1,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0]]),
        8: np.array([[0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0]]),
        9: np.array([[0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,1], [0,0,0,0,1], [0,1,1,1,0]])
    }
    
    # Get base template and resize to 28x28
    base = digit_templates.get(digit, digit_templates[0])
    
    # Create 28x28 image
    img = np.zeros((28, 28))
    
    # Place the template in the center with some offset variation
    start_row = np.random.randint(8, 15)
    start_col = np.random.randint(8, 15)
    
    for i in range(min(5, 28 - start_row)):
        for j in range(min(5, 28 - start_col)):
            if i < base.shape[0] and j < base.shape[1]:
                img[start_row + i, start_col + j] = base[i, j]
    
    # Add some noise and variations
    noise = np.random.normal(0, noise_factor, (28, 28))
    img = img + noise
    
    # Add some random stroke variations
    if np.random.random() > 0.5:
        # Add some random connected pixels
        for _ in range(np.random.randint(1, 4)):
            r, c = np.random.randint(5, 23), np.random.randint(5, 23)
            if img[r, c] > 0.5:
                # Add neighboring pixel
                dr, dc = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                if 0 <= r + dr < 28 and 0 <= c + dc < 28:
                    img[r + dr, c + dc] = max(img[r + dr, c + dc], 0.7)
    
    # Normalize to 0-1 range
    img = np.clip(img, 0, 1)
    
    # Convert to PIL Image
    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    
    return img_pil

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/generate', methods=['POST'])
def generate_images():
    try:
        data = request.get_json()
        digit = data.get('digit', 0)
        
        if not isinstance(digit, int) or digit < 0 or digit > 9:
            return jsonify({'error': 'Invalid digit. Must be between 0 and 9.'}), 400
        
        # Generate 5 different variations of the digit
        images = []
        for i in range(5):
            # Use different noise factors for variation
            noise_factor = 0.1 + (i * 0.05)
            img = generate_digit_image(digit, noise_factor)
            img_base64 = image_to_base64(img)
            images.append(img_base64)
        
        return jsonify({
            'digit': digit,
            'images': images,
            'count': len(images)
        })
    
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'MNIST Digit Generator API is running'
    })

if __name__ == '__main__':
    print("Starting MNIST Digit Generator API...")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)