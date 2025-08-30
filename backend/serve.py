import os
import json
import argparse
from PIL import Image
import numpy as np
import cv2  # Add OpenCV for eye detection
import torch
import torch.nn as nn
from torchvision import transforms
import tempfile  # For temp file handling

# Shared configurations
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = None

# Function to detect eyes in an image using OpenCV's Haar cascade
def is_eye_image(image, min_confidence=0.6):
    """
    Check if an image contains an eye using pre-trained Haar cascade classifier.
    Works with PIL Image or a file path.
    Returns True if an eye is detected, False otherwise.
    """
    # Load the eye detector
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    # Handle PIL Image by saving to a temporary file
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            img_path = tmp.name
        
        # Read the image and convert to grayscale (required for Haar cascades)
        img = cv2.imread(img_path)
        os.unlink(img_path)  # Clean up temp file
    else:
        # Assume it's a file path
        img = cv2.imread(image)
        
    if img is None:
        print(f"Warning: Could not read image")
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Check if any eyes were detected
    if len(eyes) == 0:
        print("No eyes detected")
        return False
    else:
        # additional checks here to ensure the detected eyes are valid
        # For example, check the size or position of the detected eyes
        for (x, y, w, h) in eyes:
            eye_area = img[y:y+h, x:x+w]
            eye_area_gray = cv2.cvtColor(eye_area, cv2.COLOR_BGR2GRAY)
            _, eye_area_binary = cv2.threshold(eye_area_gray, 30, 255, cv2.THRESH_BINARY)
            eye_area_confidence = np.mean(eye_area_binary) / 255.0
            
            if eye_area_confidence < min_confidence:
                print("Detected eye is not clear enough")
                return False
    return True
    

# Function to load class names from a JSON file
def load_class_names(filename='class_names.json'):
    """Load class names from a JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Class names file {filename} not found")
        exit(1)


# --------------------
# Models
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * (img_size//8) * (img_size//8), 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict_image(img: Image.Image, model, device, img_size=224, class_names=None):
    """
    img: PIL Image
    returns: (predicted_class:str, confidence:float)
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = np.argmax(probs)
    return class_names[idx], float(probs[idx])

# --------------------
# Flask App
# --------------------
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={r"/predict": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["POST"]
)

@app.route('/predict', methods=['POST'])
def predict_api():
    global model, class_names, args

    print("=== /predict called ===")
    # lazy‑load model once
    if model is None:
        print(f"Loading class names from {args.class_names}")
        class_names = load_class_names(args.class_names)
        print("Class names:", class_names)

        print(f"Loading model weights from {args.model_path}")
        model = SimpleCNN(num_classes=len(class_names))
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded.")

    file = request.files.get('image')
    if file is None:
        print("No image in request!")
        return jsonify({'error': 'No image uploaded'}), 400

    print(f"Received file: {file.filename}")
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        print("Error opening image:", e)
        return jsonify({'error': 'Invalid image file'}), 400

    # First validate that the image contains an eye, unless validation is skipped
    if not args.skip_eye_validation and not is_eye_image(img):
        print("Validation failed: No eye detected in the image")
        return jsonify({
            'error': 'No eye detected in the image',
            'valid_eye': False
        }), 400

    # Only proceed with prediction if eye is detected
    pred, conf = predict_image(
        img, model, device,
        img_size=img_size,
        class_names=class_names
    )
    print(f"Returning → class: {pred}, confidence: {conf:.4f}")
    return jsonify({
        'class': pred, 
        'confidence': conf,
        'valid_eye': True
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serve CNN for eye disease classification')
    parser.add_argument('--model_path', type=str, default='eye_model.pth',
                       help='Path to the trained model file')
    parser.add_argument('--class_names', type=str, default='class_names.json',
                       help='Path to the class names JSON file')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the service on')
    parser.add_argument('--skip_eye_validation', action='store_true',
                       help='Skip eye validation step (not recommended)')
    
    global args
    args = parser.parse_args()

    # print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        exit(1)
    if not os.path.exists(args.class_names):
        print(f"Error: Class names file {args.class_names} not found")
        exit(1)
                       
    # Start the Flask app
    print(f"Starting API server on port {args.port}...")
    app.run(host='127.0.0.1', port=args.port)