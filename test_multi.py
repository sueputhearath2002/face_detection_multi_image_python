
# =======================================
# import tensorflow as tf
# import numpy as np
# import cv2

# # Load class labels from labels.txt
# def load_labels(label_path='labels.txt'):
#     with open(label_path, 'r') as f:
#         return [line.strip() for line in f.readlines()]

# class_names = load_labels()

# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="fruit_classifier.tflite")
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Function to preprocess image for TFLite model
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))
#     img = img.astype(np.float32) / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Function to make predictions with TFLite model
# def predict_tflite(image_path, top_k=3):
#     img = preprocess_image(image_path)

#     # Set input tensor
#     interpreter.set_tensor(input_details[0]['index'], img)

#     # Run inference
#     interpreter.invoke()

#     # Get predictions
#     predictions = interpreter.get_tensor(output_details[0]['index'])[0]

#     # Get top K predictions
#     top_indices = np.argsort(predictions)[::-1][:top_k]
#     top_predictions = [(class_names[i], predictions[i]) for i in top_indices]

#     print(f"Predictions for {image_path}:")
#     for label, prob in top_predictions:
#         print(f"{label}: {prob:.6f}")

# # Test on an image
# predict_tflite("test_folder/more_fruite.jpg")  # Change to your image file
