Here's a sample `README.md` that provides instructions and details for deploying your trained model as an API using Flask:

# CIFAR-10 Image Classifier API

This repository contains a simple Flask application that serves a pre-trained image classification model. The model is trained on the CIFAR-10 dataset and predicts one of ten classes, such as "airplane" or "cat."

## Features

- **Model Deployment**: The trained model is deployed as an API using the Flask web framework.
- **Image Classification**: Accepts images via a `/predict` endpoint and returns the predicted class.
- **Preprocessing**: Automatically preprocesses incoming images to match the model's input requirements.

## Dataset Information

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in ten classes. It has 50,000 training images and 10,000 test images. The dataset classes include:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

These classes represent real-world objects and are balanced evenly within the dataset.

## Model Information

The model used in this project is a Convolutional Neural Network (CNN). Its architecture includes the following layers:

1. **Convolutional Layers**: Extract feature maps from the input image.
2. **Pooling Layers**: Reduce spatial dimensions while preserving key features.
3. **Fully Connected Layers**: Process features globally and output the classification result.
4. **Regularization**:
   - **Batch Normalization**: Normalize features for stable learning.
   - **Dropout**: Prevent overfitting by randomly deactivating neurons.

The model was trained on the CIFAR-10 dataset to achieve high classification accuracy. The pre-trained model is saved in the `best_cnn_model.h5` file.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- Pillow (PIL)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/thetajwar2003/CIFAR-classifier.git
   cd cifar10-classifier-api
   ```

2. **Install Dependencies**:
   You can install the required packages using pip:

   ```bash
   pip install flask tensorflow pillow
   ```

3. **Download Pre-trained Model**:
   Ensure that you have a trained model named `best_cnn_model.h5` in the same directory as the application file.

### Usage

1. **Run the Flask App**:
   Start the application server by running:

   ```bash
   python app.py
   ```

   This will start a server at `http://0.0.0.0:3000`.

2. **Make Predictions**:
   The server exposes a `/predict` endpoint that accepts image files via a `POST` request.

   - **HTTP Method**: `POST`
   - **Endpoint**: `/predict`
   - **Body**: Form-data containing an image file, named `image`.

   **Example Using `curl`**:

   ```bash
   curl -X POST -F "image=@path/to/your/image.jpg" http://0.0.0.0:3000/predict
   ```

   This will return a JSON object containing the predicted class:

   ```json
   {
     "predicted_class": "airplane"
   }
   ```

   **Example Using Postman**:

   - Create a new `POST` request to `http://0.0.0.0:3000/predict`.
   - In the "Body" tab, select "form-data" and add a field named `image`, then choose an image file.
   - Send the request and review the predicted class in the JSON response.

### Project Structure

```
cifar10-classifier-api/
│
├── cifar-10-batches-py/   # Data files and batches
├── app.py                 # Flask application file
├── best_cnn_model.h5      # Pre-trained model
├── main.ipynb             # Model training and analysis notebook
└── README.md              # Instructions and documentation
```
