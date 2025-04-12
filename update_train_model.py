import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import os

# Path to your dataset directory
dataset_dir = 'datas'

# Check if dataset directory exists
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The dataset directory '{dataset_dir}' does not exist!")

# Print contents of the dataset directory
print(f"Contents of '{dataset_dir}': {os.listdir(dataset_dir)}")

# Define data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Define validation data preprocessing
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training dataset
train_ds = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    seed=123,
)

# Load validation dataset
val_ds = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    seed=123,
)

# Get updated class names
class_names = list(train_ds.class_indices.keys())
print("Updated Class Names: ", class_names)

# Save class names to `labels.txt`
with open('labels.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')
print("Updated labels.txt created successfully!")

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

# Build the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax'),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
tflite_model_path = 'updated_fruit_classifier.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Updated model converted to {tflite_model_path} successfully!")

# Evaluate the model on the validation dataset
test_loss, test_acc = model.evaluate(val_ds)
print(f"Updated Test accuracy: {test_acc}")
