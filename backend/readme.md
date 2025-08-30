# Eye Disease Classification Project

## Libraries Used

### Core Python Libraries
- **os**: File path handling in both train\.py and serve\.py
- **json**: Reading/writing class name mappings in JSON format
- **argparse**: Command-line argument parsing for configuration options
- **PIL (Python Imaging Library)**: Image loading and preprocessing

### Machine Learning Libraries
- **numpy**: Numerical operations and array manipulation
- **torch**: Core PyTorch library for deep learning
  - **torch.nn**: Neural network modules and building blocks
  - **torch.optim**: Optimization algorithms (Adam)
  - **torch.utils.data**: Data loading utilities
- **torchvision**: 
  - **datasets**: `ImageFolder` for organizing training data
  - **transforms**: Image preprocessing operations

### Model Evaluation
- **sklearn.metrics**: Evaluation metrics for model assessment
  - `confusion_matrix`: Displays class prediction errors
  - `classification_report`: Provides precision, recall, F1-score metrics

### Web Service (in serve\.py)
- **flask**: Web server for model deployment
- **flask_cors**: Cross-Origin Resource Sharing for API access

## Functions and Components

### Data Handling
- **save_class_names(class_names, filename)**: Saves class names to a JSON file
  - Parameters: class_names (list), filename (str)
  
- **load_class_names(filename)**: Retrieves class names from a JSON file
  - Parameters: filename (str)
  - Returns: List of class names

- **get_data_loaders(data_dir, batch_size, img_size, val_split, test_split)**:
  - Purpose: Prepares train, validation, and test data loaders
  - Parameters:
    - data_dir: Path to dataset
    - batch_size: Number of images per batch (default: 32)
    - img_size: Target image dimensions (default: 224)
    - val_split: Validation set proportion (default: 0.15) 
    - test_split: Test set proportion (default: 0.15)
  - Returns: train_loader, val_loader, test_loader, class_names

### Model Architecture
- **SimpleCNN**: Convolutional Neural Network for image classification
  - Architecture:
    - 3 convolutional layers with increasing channels (3→32→64→128)
    - ReLU activation and MaxPooling after each convolutional layer
    - Two fully connected layers with dropout (0.5) for regularization
  - Parameters: num_classes (default: 5)

### Training and Evaluation
- **train_model(model, dataloaders, criterion, optimizer, device, num_epochs)**:
  - Purpose: Trains the model on provided data
  - Parameters:
    - model: Neural network model
    - dataloaders: Dict with 'train' and 'val' data loaders
    - criterion: Loss function (CrossEntropyLoss)
    - optimizer: Adam optimizer
    - device: CPU or GPU
    - num_epochs: Training iterations (default: 10)
  - Returns: Trained model
  - Saves: Best model weights to 'eye_model.pth'

- **evaluate_model(model, dataloader, device, class_names)**:
  - Purpose: Evaluates model performance on test data
  - Parameters:
    - model: Trained neural network
    - dataloader: Test data loader
    - device: CPU or GPU
    - class_names: List of class names
  - Outputs: Confusion matrix and classification report


### Inference
- **predict_image(img, model, device, img_size, class_names)**:
  - Purpose: Makes prediction on a single image
  - Parameters:
    - img: PIL Image
    - model: Trained neural network
    - device: CPU or GPU
    - img_size: Target image dimensions
    - class_names: List of class names
  - Returns: predicted_class (str), confidence (float)

### Web API (in serve\.py)
- **Flask routes**:
  - `/predict`: Accepts image uploads and returns classification results
  - Implementation: Loads model and class names on first request (lazy loading)

## Machine Learning Tools and Techniques

- **Data Augmentation**: Random horizontal flips and rotations to improve model robustness
- **Transfer Learning**: Potential to use pre-trained models (structure is in place)
- **Model Regularization**: Dropout layers (0.5) to prevent overfitting
- **Validation Strategy**: Train/validation/test split for proper evaluation
- **Optimization**: Adam optimizer with configurable learning rate
- **Evaluation Metrics**: Confusion matrix and classification report with precision, recall, and F1-scores
- **Normalization**: Image pixel normalization using ImageNet statistics

> The project implements a complete machine learning pipeline for eye disease classification from training through evaluation to deployment as a web service.

---
# Steps to Run

Below are the step-by-step instructions to train the model, serve the API, and make predictions.

## Setup and Requirements

### 1. Install Required Libraries
```bash
pip install torch torchvision scikit-learn flask flask-cors numpy
```

### 2. Prepare Dataset
- Download the eye disease dataset and extract it
- Place it in a folder structure where each disease has its own subfolder containing images:
  ```
  data/
  ├── Cataract/
  ├── Conjunctivitis/
  ├── Eyelid/
  ├── Normal/
  └── Uveitis/
  ```

## Training the Model

### 1. Run the Training Script
```python
python train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 0.001
```


#### Parameters:
- `--data_dir`: Path to the dataset directory (default: `./data`)
- `--epochs`: Number of training epochs (default: `5`)
- `--batch_size`: Batch size for training (default: `32`)
- `--lr`: Learning rate for the optimizer (default: `0.0001`)

#### Outputs:
- `eye_model.pth`: Trained model weights
- `class_names.json`: Class names mapping file

#### Example:

- Train a model with default parameters
```py
python train.py
```


## Serving the API

### 1. Run the Server
```py
python serve.py --model_path eye_model.pth --class_names class_names.json --port 5000
```

#### Parameters:
- `--model_path`: Path to the trained model file (default: `eye_model.pth`)
- `--class_names`: Path to the class names JSON file (default: `class_names.json`)
- `--port`: Port to run the service on (default: `5000`)
- `--skip_eye_validation`: Skip eye validation step (not recommended)

#### Example:

- Start to serve a model with default parameters
```py
python serve.py
```


### 2. Access the API
- The API will be available at: `http://127.0.0.1:5000/predict`
- Send POST requests with image files in the `image` field

## Making Predictions

### 1. Run the Prediction Script
```py
python predict.py
```



#### Usage:
- When prompted, enter the path to an image file
- The script will send the image to the API and display the prediction results



### 3. Using the Web Interface
- Open the React frontend (refer to `eye_frontend` repo)
- Upload an image or use the camera
- View the prediction results

## Troubleshooting

### Common Issues:
1. **Model Loading Error**: Ensure the paths to model and class_names files are correct
2. **CORS Error**: When accessing the API from a different origin, check CORS settings in `serve.py`
3. **Memory Issues**: For large images or models, try reducing batch size or image size

### Memory Optimization:
- Use a smaller input image size (e.g., 160×160 instead of 224×224)
- Reduce model complexity by using LightweightCNN instead of SimpleCNN
- Apply quantization to reduce model size:
  ```python
  import torch
  model = torch.load('eye_model.pth')
  quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
  torch.save(quantized_model, 'eye_model_quantized.pth')
  ```

---
#### Dataset
This project uses the eye disease dataset from:
> [link to dataset](https://data.mendeley.com/datasets/n9zp473wfw/2)

Bitto, Abu Kowshir ; Ahmed, Marzia (2024), “Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation”, Mendeley Data, V2, doi: 10.17632/n9zp473wfw.2

