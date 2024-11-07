import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model("skin.h5")

# Load and preprocess the image
img_path = "img1-actinic keratosis.jpg"
img = image.load_img(img_path, target_size=(64, 64))  # Adjust target_size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if necessary

# Run inference
predictions = model.predict(img_array)

# Interpret the results
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class:", predicted_class)
