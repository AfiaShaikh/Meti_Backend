# MNIST Digit Generator Backend

This is the Python backend for the MNIST digit generation application using PyTorch.

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your trained model:**
   - Place your trained model file as `mnist_model.pth` in this directory
   - The model should match the MNIST_CNN architecture provided

3. **Run the server:**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

- `POST /generate` - Generate 5 images of a specified digit
  - Body: `{"digit": 0-9}`
  - Returns: `{"digit": int, "images": [base64_strings], "count": int}`

- `GET /health` - Check server health and model status

## Model Architecture

The expected model architecture (MNIST_CNN):
- Conv2d(1, 32, 3x3) + ReLU + MaxPool2d(2x2)
- Conv2d(32, 64, 3x3) + ReLU + MaxPool2d(2x2)
- Flatten + Linear(3136, 128) + ReLU + Linear(128, 10)

## Notes

- The application includes a fallback digit generator that creates synthetic digits if the model file is not present
- Images are returned as base64-encoded PNG data
- CORS is enabled for frontend integration