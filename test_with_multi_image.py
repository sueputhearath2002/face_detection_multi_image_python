
import cv2
import os
import numpy as np
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from ultralytics import YOLO

# === CONFIGURATION ===
dataset_dir = "datasv4"
input_shape = (224, 224, 3)
batch_size = 32
epochs = 50
validation_split = 0.2
seed = 123

# === CHECK DATASET DIRECTORY ===
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

print(f" Dataset directory '{dataset_dir}' found.")
print(f" Contents: {os.listdir(dataset_dir)}")

# === DATA LOADING & AUGMENTATION ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=validation_split
)

train_ds = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=seed
)

val_ds = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=seed
)

# === PRINT CLASS LABELS ===
class_names = list(train_ds.class_indices.keys())
print(" Class Names:", class_names)

with open("labels.txt", "w") as f:
    f.writelines("\n".join(class_names))
print("'labels.txt' created!")

# === MODEL ARCHITECTURE ===
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    include_preprocessing=True
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# === COMPILE MODEL ===
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# === TRAIN THE MODEL (Initial Phase) ===
print("Training classifier layers...")
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# === OPTIONAL: UNFREEZE AND FINE-TUNE ===
print("🔓 Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze early layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# === PLOT TRAINING ACCURACY ===
plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# === CONVERT TO TFLITE ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("face_detection.tflite", "wb") as f:
    f.write(tflite_model)
print(" Model converted to 'face_detection.tflite'")

# === YOLOv8 FACE DETECTION ===
yolo_model_path = "yolov8n-face.pt"
if not os.path.exists(yolo_model_path):
    print("Downloading YOLOv8 face model...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    response = requests.get(url, stream=True)
    with open(yolo_model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(" YOLOv8 model downloaded!")

def detect_and_classify_faces(image_path, model, class_names):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    model_yolo = YOLO(yolo_model_path)
    image = cv2.imread(image_path)
    results = model_yolo(image)

    face_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_boxes.append((x1, y1, x2, y2))

    detected_faces = {}

    for (fx1, fy1, fx2, fy2) in face_boxes:
        face = image[fy1:fy2, fx1:fx2]
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        face_resized = cv2.resize(face, (224, 224))
        face_preprocessed = preprocess_input(face_resized.astype(np.float32))
        face_array = np.expand_dims(face_preprocessed, axis=0)

        predictions = model.predict(face_array)[0]
        best_index = np.argmax(predictions)
        best_class = class_names[best_index]
        best_confidence = float(predictions[best_index])

        if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
            detected_faces[best_class] = best_confidence

        label_text = f"{best_class}: {best_confidence:.2f}"
        cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = "detected_faces.png"
    cv2.imwrite(output_path, image)
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ✅ Format: [{"name": "confidence"}, ...]
    formatted_faces = [{k: f"{v}"} for k, v in detected_faces.items()]
    print("Detected faces:", formatted_faces)

# === TEST FUNCTION ===
detect_and_classify_faces("test_folder/people_v4.png", model, class_names)

# =============================103%===================

# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from ultralytics import YOLO

# # === CONFIGURATION ===
# dataset_dir = "datasv4"
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 50
# validation_split = 0.2
# seed = 123

# # === CHECK DATASET DIRECTORY ===
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f" Dataset directory '{dataset_dir}' found.")
# print(f" Contents: {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.4,
#     brightness_range=[0.6, 1.4],
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=validation_split
# )

# train_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     seed=seed
# )

# val_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
#     seed=seed
# )

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print("'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV3Large(
#     input_shape=input_shape,
#     include_top=False,
#     weights="imagenet",
#     include_preprocessing=True
# )
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()

# # === TRAIN THE MODEL (Initial Phase) ===
# print("Training classifier layers...")
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# # === OPTIONAL: UNFREEZE AND FINE-TUNE ===
# print("🔓 Fine-tuning base model...")
# base_model.trainable = True
# for layer in base_model.layers[:100]:  # Freeze early layers
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# fine_tune_epochs = 10
# total_epochs = epochs + fine_tune_epochs

# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === YOLOv8 FACE DETECTION ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print("Downloading YOLOv8 face model...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(" YOLOv8 model downloaded!")

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     model_yolo = YOLO(yolo_model_path)
#     image = cv2.imread(image_path)
#     results = model_yolo(image)

#     face_boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_boxes.append((x1, y1, x2, y2))

#     detected_faces = {}

#     for (fx1, fy1, fx2, fy2) in face_boxes:
#         face = image[fy1:fy2, fx1:fx2]
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         face_resized = cv2.resize(face, (224, 224))
#         face_preprocessed = preprocess_input(face_resized.astype(np.float32))
#         face_array = np.expand_dims(face_preprocessed, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Format the detected_faces as requested
#     formatted_faces = [f"'{k}': {v}" for k, v in detected_faces.items()]
#     print("Detected faces:", formatted_faces)

# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)


# =================================102%======================
# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from ultralytics import YOLO

# # === CONFIGURATION ===
# dataset_dir = "datasv4"
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 50
# validation_split = 0.2
# seed = 123

# # === CHECK DATASET DIRECTORY ===
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f"✅ Dataset directory '{dataset_dir}' found.")
# print(f"📁 Contents: {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.4,
#     brightness_range=[0.6, 1.4],
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=validation_split
# )

# train_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     seed=seed
# )

# val_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
#     seed=seed
# )

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print("'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV3Large(
#     input_shape=input_shape,
#     include_top=False,
#     weights="imagenet",
#     include_preprocessing=True
# )
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()

# # === TRAIN THE MODEL (Initial Phase) ===
# print("Training classifier layers...")
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# # === OPTIONAL: UNFREEZE AND FINE-TUNE ===
# print("🔓 Fine-tuning base model...")
# base_model.trainable = True
# for layer in base_model.layers[:100]:  # Freeze early layers
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# fine_tune_epochs = 10
# total_epochs = epochs + fine_tune_epochs

# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === YOLOv8 FACE DETECTION ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print("Downloading YOLOv8 face model...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(" YOLOv8 model downloaded!")

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     model_yolo = YOLO(yolo_model_path)
#     image = cv2.imread(image_path)
#     results = model_yolo(image)

#     face_boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_boxes.append((x1, y1, x2, y2))

#     detected_faces = {}

#     for (fx1, fy1, fx2, fy2) in face_boxes:
#         face = image[fy1:fy2, fx1:fx2]
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         face_resized = cv2.resize(face, (224, 224))
#         face_preprocessed = preprocess_input(face_resized.astype(np.float32))
#         face_array = np.expand_dims(face_preprocessed, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # ✅ Format the detected_faces as requested
#     formatted_faces = [f"'{k}': {v}" for k, v in detected_faces.items()]
#     print("Detected faces:", formatted_faces)

# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)


# ============================101%==========================================
# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from ultralytics import YOLO

# # === CONFIGURATION ===
# dataset_dir = "datasv4"
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 30
# validation_split = 0.2
# seed = 123

# # === CHECK DATASET DIRECTORY ===
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f"✅ Dataset directory '{dataset_dir}' found.")
# print(f"📁 Contents: {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.4,
#     brightness_range=[0.6, 1.4],
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=validation_split
# )

# train_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     seed=seed
# )

# val_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
#     seed=seed
# )

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print("'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV3Large(
#     input_shape=input_shape,
#     include_top=False,
#     weights="imagenet",
#     include_preprocessing=True
# )
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()

# # === TRAIN THE MODEL (Initial Phase) ===
# print("Training classifier layers...")
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# # === OPTIONAL: UNFREEZE AND FINE-TUNE ===
# print("🔓 Fine-tuning base model...")
# base_model.trainable = True
# for layer in base_model.layers[:100]:  # Freeze early layers
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# fine_tune_epochs = 10
# total_epochs = epochs + fine_tune_epochs

# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === YOLOv8 FACE DETECTION ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print("Downloading YOLOv8 face model...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(" YOLOv8 model downloaded!")

# def non_max_suppression(boxes, scores, threshold=0.4):
#     indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
#     return [boxes[i] for i in indices.flatten()]

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     model_yolo = YOLO(yolo_model_path)
#     image = cv2.imread(image_path)
#     results = model_yolo(image)

#     face_boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_boxes.append((x1, y1, x2, y2))

#     detected_faces = {}

#     for (fx1, fy1, fx2, fy2) in face_boxes:
#         face = image[fy1:fy2, fx1:fx2]
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         face_resized = cv2.resize(face, (224, 224))
#         face_preprocessed = preprocess_input(face_resized.astype(np.float32))
#         face_array = np.expand_dims(face_preprocessed, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print(" Detection result saved to '{output_path}'")
#     print(" Detected faces:", detected_faces)


# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)
# ========================================100%==========================================
# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import torch
# import numpy
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from ultralytics import YOLO

# # === CONFIGURATION ===
# dataset_dir = "datasv4"
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 30
# validation_split = 0.2
# seed = 123

# # === CHECK DATASET DIRECTORY ===
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f"✅ Dataset directory '{dataset_dir}' found.")
# print(f"📁 Contents: {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.4,
#     brightness_range=[0.6, 1.4],
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=validation_split
# )

# train_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     seed=seed
# )

# val_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
#     seed=seed
# )

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print("'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV3Large(
#     input_shape=input_shape,
#     include_top=False,
#     weights="imagenet",
#     include_preprocessing=True
# )
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()

# # === TRAIN THE MODEL (Initial Phase) ===
# print("Training classifier layers...")
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# # === OPTIONAL: UNFREEZE AND FINE-TUNE ===
# print("🔓 Fine-tuning base model...")
# base_model.trainable = True
# for layer in base_model.layers[:100]:  # Freeze early layers
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# fine_tune_epochs = 10
# total_epochs = epochs + fine_tune_epochs

# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === YOLOv8 FACE DETECTION ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print("Downloading YOLOv8 face model...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(" YOLOv8 model downloaded!")

# def non_max_suppression(boxes, scores, threshold=0.4):
#     indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
#     return [boxes[i] for i in indices.flatten()]

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     model_yolo = YOLO(yolo_model_path)
#     image = cv2.imread(image_path)
#     results = model_yolo(image)

#     face_boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_boxes.append((x1, y1, x2, y2))

#     detected_faces = {}

#     for (fx1, fy1, fx2, fy2) in face_boxes:
#         face = image[fy1:fy2, fx1:fx2]
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         face_resized = cv2.resize(face, (224, 224))
#         face_preprocessed = preprocess_input(face_resized.astype(np.float32))
#         face_array = np.expand_dims(face_preprocessed, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print(" Detection result saved to '{output_path}'")
#     print(" Detected faces:", detected_faces)


# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)

# ====================================100% with mobielNetV3========================
# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from ultralytics import YOLO

# # === CONFIGURATION ===
# dataset_dir = "datasv4"
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 30
# validation_split = 0.2
# seed = 123

# # === CHECK DATASET DIRECTORY ===
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f"✅ Dataset directory '{dataset_dir}' found.")
# print(f"📁 Contents: {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.4,
#     brightness_range=[0.6, 1.4],
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=validation_split
# )

# train_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     seed=seed
# )

# val_ds = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
#     seed=seed
# )

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print("🏷️ Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print("✅ 'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV3Small(
#     input_shape=input_shape,
#     include_top=False,
#     weights="imagenet",
#     include_preprocessing=True
# )
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()

# # === TRAIN THE MODEL (Initial Phase) ===
# print("🚀 Training classifier layers...")
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# # === OPTIONAL: UNFREEZE AND FINE-TUNE ===
# print("🔓 Fine-tuning base model...")
# base_model.trainable = True
# for layer in base_model.layers[:100]:  # Freeze early layers
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# fine_tune_epochs = 10
# total_epochs = epochs + fine_tune_epochs

# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=history.epoch[-1] + 1)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print("✅ Model converted to 'face_detection.tflite'")

# # === YOLOv8 FACE DETECTION ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print("🔽 Downloading YOLOv8 face model...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print("✅ YOLOv8 model downloaded!")

# def non_max_suppression(boxes, scores, threshold=0.4):
#     indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
#     return [boxes[i] for i in indices.flatten()]

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     model_yolo = YOLO(yolo_model_path)
#     image = cv2.imread(image_path)
#     results = model_yolo(image)
    
#     face_boxes, face_scores = [], []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             if result.names[int(box.cls[0])] == "person" and confidence > 0.5:
#                 face_boxes.append([x1, y1, x2, y2])
#                 face_scores.append(confidence)

#     filtered_faces = non_max_suppression(face_boxes, face_scores)
#     detected_faces = {}

#     for (fx1, fy1, fx2, fy2) in filtered_faces:
#         face = image[fy1:fy2, fx1:fx2]
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         face_resized = cv2.resize(face, (224, 224))
#         face_preprocessed = preprocess_input(face_resized.astype(np.float32))
#         face_array = np.expand_dims(face_preprocessed, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     print(f"🖼️ Detection result saved to '{output_path}'")
#     print("📌 Detected faces:", detected_faces)

# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)


# ========================================100%==========================================

# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from ultralytics import YOLO

# # === CHECK DATASET DIRECTORY ===
# dataset_dir = "datasv4"
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f" Dataset directory '{dataset_dir}' found.")
# print(f" Contents of '{dataset_dir}': {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     rescale=1./255, rotation_range=40, width_shift_range=0.3, height_shift_range=0.3,
#     shear_range=0.3, zoom_range=0.4, brightness_range=[0.6, 1.4], horizontal_flip=True, fill_mode="nearest"
# )
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_ds = train_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", seed=123)
# val_ds = val_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", seed=123)

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print(" 'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),  # Increased dropout for better generalization
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=["accuracy"])
# model.summary()

# # === TRAIN THE MODEL ===
# history = model.fit(train_ds, validation_data=val_ds, epochs=50)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT MODEL TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === DOWNLOAD YOLOv8 FACE MODEL IF MISSING ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print(f"🔽 Downloading '{yolo_model_path}'...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(f" '{yolo_model_path}' downloaded successfully!")

# # === YOLOv8 FACE DETECTION & CLASSIFICATION ===
# def non_max_suppression(boxes, scores, threshold=0.4):
#     """Apply Non-Maximum Suppression (NMS) to remove overlapping face detections."""
#     indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
#     return [boxes[i] for i in indices.flatten()]

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     # Load YOLO model
#     model_yolo = YOLO(yolo_model_path)

#     # Read input image
#     image = cv2.imread(image_path)
#     results = model_yolo(image)
    
#     face_boxes = []
#     face_scores = []
    
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])

#             if result.names[int(box.cls[0])] == "person" and confidence > 0.5:
#                 face_boxes.append([x1, y1, x2, y2])
#                 face_scores.append(confidence)

#     # Apply Non-Maximum Suppression to filter overlapping faces
#     filtered_faces = non_max_suppression(face_boxes, face_scores)

#     detected_faces = {}  # Dictionary to store unique detections

#     for (fx1, fy1, fx2, fy2) in filtered_faces:
#         face = image[fy1:fy2, fx1:fx2]

#         # Ensure face is valid
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         # Preprocess face for classification
#         face_resized = cv2.resize(face, (224, 224)) / 255.0
#         face_array = np.expand_dims(face_resized, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         # Store only the highest confidence per class
#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         # Draw rectangle and label
#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     print(f" Face detection saved as '{output_path}'")
#     print(f" Detected faces: {detected_faces}")

# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v4.png", model, class_names)

# ==================================================================================90========================

# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# import requests
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from ultralytics import YOLO 

# # === CHECK DATASET DIRECTORY ===
# dataset_dir = "datav3"
# if not os.path.exists(dataset_dir):
#     raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found!")

# print(f" Dataset directory '{dataset_dir}' found.")
# print(f" Contents of '{dataset_dir}': {os.listdir(dataset_dir)}")

# # === DATA LOADING & AUGMENTATION ===
# train_datagen = ImageDataGenerator(
#     rescale=1./255, rotation_range=40, width_shift_range=0.3, height_shift_range=0.3,
#     shear_range=0.3, zoom_range=0.4, brightness_range=[0.6, 1.4], horizontal_flip=True, fill_mode="nearest"
# )
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_ds = train_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", seed=123)
# val_ds = val_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", seed=123)

# # === PRINT CLASS LABELS ===
# class_names = list(train_ds.class_indices.keys())
# print(" Class Names:", class_names)

# with open("labels.txt", "w") as f:
#     f.writelines("\n".join(class_names))
# print(" 'labels.txt' created!")

# # === MODEL ARCHITECTURE ===
# base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
# base_model.trainable = False

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.6),  # Increased dropout for better generalization
#     tf.keras.layers.Dense(len(class_names), activation="softmax")
# ])

# # === COMPILE MODEL ===
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1000, 0.9)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=["accuracy"])
# model.summary()

# # === TRAIN THE MODEL ===
# history = model.fit(train_ds, validation_data=val_ds, epochs=50)

# # === PLOT TRAINING ACCURACY ===
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # === CONVERT MODEL TO TFLITE ===
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("face_detection.tflite", "wb") as f:
#     f.write(tflite_model)
# print(" Model converted to 'face_detection.tflite'")

# # === DOWNLOAD YOLOv8 FACE MODEL IF MISSING ===
# yolo_model_path = "yolov8n-face.pt"
# if not os.path.exists(yolo_model_path):
#     print(f"🔽 Downloading '{yolo_model_path}'...")
#     url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
#     response = requests.get(url, stream=True)
#     with open(yolo_model_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(f" '{yolo_model_path}' downloaded successfully!")

# # === YOLOv8 FACE DETECTION & CLASSIFICATION ===
# def non_max_suppression(boxes, scores, threshold=0.4):
#     """Apply Non-Maximum Suppression (NMS) to remove overlapping face detections."""
#     indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
#     return [boxes[i] for i in indices.flatten()]

# def detect_and_classify_faces(image_path, model, class_names):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image '{image_path}' not found.")

#     # Load YOLO model
#     model_yolo = YOLO(yolo_model_path)

#     # Read input image
#     image = cv2.imread(image_path)
#     results = model_yolo(image)
    
#     face_boxes = []
#     face_scores = []
    
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])

#             if result.names[int(box.cls[0])] == "person" and confidence > 0.5:
#                 face_boxes.append([x1, y1, x2, y2])
#                 face_scores.append(confidence)

#     # Apply Non-Maximum Suppression to filter overlapping faces
#     filtered_faces = non_max_suppression(face_boxes, face_scores)

#     detected_faces = {}  # Dictionary to store unique detections

#     for (fx1, fy1, fx2, fy2) in filtered_faces:
#         face = image[fy1:fy2, fx1:fx2]

#         # Ensure face is valid
#         if face.shape[0] == 0 or face.shape[1] == 0:
#             continue

#         # Preprocess face for classification
#         face_resized = cv2.resize(face, (224, 224)) / 255.0
#         face_array = np.expand_dims(face_resized, axis=0)

#         predictions = model.predict(face_array)[0]
#         best_index = np.argmax(predictions)
#         best_class = class_names[best_index]
#         best_confidence = float(predictions[best_index])

#         # Store only the highest confidence per class
#         if best_class not in detected_faces or best_confidence > detected_faces[best_class]:
#             detected_faces[best_class] = best_confidence

#         # Draw rectangle and label
#         label_text = f"{best_class}: {best_confidence:.2f}"
#         cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#         cv2.putText(image, label_text, (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     output_path = "detected_faces.png"
#     cv2.imwrite(output_path, image)
#     cv2.imshow("Detected Faces", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     print(f" Face detection saved as '{output_path}'")
#     print(f" Detected faces: {detected_faces}")

# # === TEST FUNCTION ===
# detect_and_classify_faces("test_folder/people_v2.png", model, class_names)




