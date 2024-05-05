from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('best_cnn_model.h5')

# Initialize Flask app
app = Flask(__name__)


def preprocess_image(image):
    """ Preprocess image to fit model input requirements. """
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    # Load image from request
    file = request.files['image']
    image = Image.open(file)

    # Preprocess the image
    image_input = preprocess_image(image)

    # Get model prediction
    predictions = model.predict(image_input)
    predicted_class = np.argmax(predictions[0])

    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Return the predicted class
    return jsonify({'predicted_class': classes[predicted_class]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
