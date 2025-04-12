import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="fruit_classifier.tflite")
interpreter.allocate_tensors()

# Load an image and preprocess it
image = Image.open("datas/banana/banana_1.jpg")
image = image.resize((224, 224))
image = np.array(image) / 255.0  # Normalize to [0, 1]
image = np.expand_dims(image, axis=0).astype(np.float32)

# Set up input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)

# Get predicted class
predicted_class = np.argmax(output_data)
print(f"Predicted class: {predicted_class}")
