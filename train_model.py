import tensorflow as tf
import matplotlib.pyplot as plt
# from keras_preprocessing import image_dataset_from_directory
from keras_preprocessing.image import ImageDataGenerator
import os

# Load images from the dataset folder
dataset_dir = 'datas'  # Path to your dataset folder
if not os.path.exists(dataset_dir):
    print(f"The dataset directory '{dataset_dir}' does not exist!")
else:
    print(f"The dataset directory '{dataset_dir}' exists.")

# List the contents of the dataset directory to check the structure
print(f"Contents of '{dataset_dir}':")
print(os.listdir(dataset_dir))

# Define data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training dataset with augmentation
train_ds = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    seed=123
)

# Load validation dataset
val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation
val_ds = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='sparse',  # Use integer labels for sparse categorical crossentropy
    # subset="validation",  # Specify this as validation data
    seed=123
)

# Get class names from class_indices attribute
class_names = list(train_ds.class_indices.keys())  # Extract the class names directly
print("Class Names: ", class_names)

# Save class names to labels.txt
with open('labels.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')
print("labels.txt created successfully!")

# Load a pre-trained MobileNetV2 model for better feature extraction
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model to prevent overfitting

# Build the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Output layer with the correct number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Calculate class weights to handle class imbalance
class_weights = {i: 1.0 for i in range(len(class_names))}  # Default equal weights

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,  # Ensure validation data is passed
    epochs=50,
    class_weight=class_weights
)

# Check available keys in history to debug
print("History keys:", history.history.keys())

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')

# Plot validation accuracy (check if 'val_accuracy' or 'val_acc' is available)
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
elif 'val_acc' in history.history:
    plt.plot(history.history['val_acc'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open('fruit_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to fruit_classifier.tflite successfully!")

# Model evaluation
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc}")
