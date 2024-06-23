import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('E:\image classification\models\model.h5')

# Define the target image size expected by the model
target_size = (150, 150)

# Function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

# Function to make predictions on a single image
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])  # Get index of the highest probability class
    return predicted_class

# Example usage: Provide a path to a new image to make predictions
image_path = r'c:\Users\moham\Downloads\pexels-arts-1547813.jpg'
predicted_class = predict_image(image_path)

# Define class labels (assuming your classes are 0 to 5 as per your earlier description)
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Print the predicted class label
print('Predicted class:', class_labels[predicted_class])
