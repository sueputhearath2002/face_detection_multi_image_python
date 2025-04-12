import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Path to your dataset folder
dataset_dir = 'datas'
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The dataset directory '{dataset_dir}' does not exist!")
else:
    print(f"The dataset directory '{dataset_dir}' exists.")

# Get class names from the dataset
class_names = sorted(os.listdir(dataset_dir))
class_names = [cls for cls in class_names if cls != '.DS_Store']  # Filter out hidden files
print("Class Names:", class_names)

# Get the file paths for the images and their labels
image_paths = []
image_labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for image_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_file))
            image_labels.append(class_idx)

# Split data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, image_labels, test_size=0.2, random_state=123, stratify=image_labels)

# Create train and validation data generators with improved data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    rotation_range=30,      # Random rotation
    zoom_range=0.2,         # Random zoom
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    shear_range=0.2,        # Random shear transformation
    horizontal_flip=True,   # Random horizontal flip
    vertical_flip=True,     # Random vertical flip
    channel_shift_range=20.0, # Random color channel shift
    fill_mode='nearest'     # Fill the new pixels with the nearest values
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Use the ImageDataGenerator flow_from_directory method for the train and validation sets
train_ds = train_datagen.flow_from_directory(
    directory=dataset_dir,
    class_mode='sparse',
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=123
)

val_ds = val_datagen.flow_from_directory(
    directory=dataset_dir,
    class_mode='sparse',
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    seed=123
)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model layers

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model with a learning rate variable
learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Compute class weights to handle class imbalance
labels = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='./logs', histogram_freq=1)
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the validation dataset
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc}")

# Predict labels for the validation dataset
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=-1)

# Compute confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('face_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to face_detection_model.tflite successfully!")
