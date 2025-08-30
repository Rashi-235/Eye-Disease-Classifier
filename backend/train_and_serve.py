import os
import argparse
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report

img_size = 224

# --------------------
# Data Loading
# --------------------
def get_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.15, test_split=0.15):
    """
    Prepare train, validation, and test DataLoaders from directory of class folders.
    """
    # Transforms for training and evaluation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = full_dataset.classes
    print(f"Found {len(full_dataset)} images in {len(class_names)} classes.")
    print("Class names:", class_names)

    # Split indices
    total = len(full_dataset)
    val_size = int(val_split * total)
    test_size = int(test_split * total)
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Override transforms for val and test
    val_ds.dataset.transform = eval_transform
    test_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, class_names

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

# --------------------
# Training & Evaluation
# --------------------
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    train_loader, val_loader = dataloaders['train'], dataloaders['val']
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f"Best validation accuracy: {best_acc:.4f}")
    return model


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

# --------------------
# Inference util
# --------------------
def load_model(
    model_path:str,
    device,
    num_classes:int=5,
    use_pretrained:bool=True
):
    """
    Reconstructs the exact model architecture you trained (scratch vs pretrained),
    then loads weights from model_path.
    """
    if use_pretrained:
        # transfer‑learning ResNet18
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # scratch CNN
        model = SimpleCNN(num_classes=num_classes)
    # now load the weights you saved
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


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

# Load the trained model once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = None  # will be set after loading data
model = None

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
        print(f"Loading class names from {args.data_dir} …")
        _, _, _, class_names = get_data_loaders(
            args.data_dir,
            img_size=img_size
        )
        # print("Class names:", class_names)

        print(f"Loading model weights best_model.pth "
              f" (use_pretrained={args.use_pretrained}) …")
        model = load_model(
            'best_model.pth',
            device,
            num_classes=len(class_names),
            use_pretrained=args.use_pretrained
        )
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

    pred, conf = predict_image(
        img, model, device,
        img_size=img_size,
        class_names=class_names
    )
    print(f"Returning → class: {pred}, confidence: {conf:.4f}")
    return jsonify({'class': pred, 'confidence': conf})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or serve CNN for eye disease classification')
    parser.add_argument('--mode', type=str, choices=['train', 'serve'], default='train')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Path to root directory of class subfolders')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained ResNet18 instead of training CNN from scratch')
    global args
    args = parser.parse_args()

    if args.mode == 'train':
        # Prepare data
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
            args.data_dir, batch_size=args.batch_size)
        dataloaders = {'train': train_loader, 'val': val_loader}
        # Choose model
        if args.use_pretrained:
            model = load_model(model_path=None, device=device, use_pretrained=True)
            # replace final layer
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.to(device)
        else:
            # global img_size
            img_size = 224
            model = SimpleCNN(num_classes=len(class_names))
            model.to(device)
        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Train
        train_model(model, dataloaders, criterion, optimizer, device, num_epochs=args.epochs)
        # Evaluate on test set
        print("\nTest set performance:")
        evaluate_model(model, test_loader, device, class_names)
    else:
        # Serve via Flask
        app.run(host='127.0.0.1', port=5000)  
