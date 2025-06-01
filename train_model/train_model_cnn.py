import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import img_to_array, load_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIG ===
DATASET_PATH = "/Users/y.slupitskyi/Work/item_recognition/dataset.zip"
EXTRACTED_PATH = "/Users/y.slupitskyi/Work/item_recognition/dataset"
MODEL_SAVE_PATH = "best_custom_cnn.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# === UNZIP ===
if not os.path.exists(EXTRACTED_PATH):
    with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(EXTRACTED_PATH))

# === LOAD IMAGES ===
def load_dataset(path):
    X, y, labels = [], [], sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    label_map = {name: idx for idx, name in enumerate(labels)}
    for label in labels:
        folder = os.path.join(path, label)
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                img = load_img(os.path.join(folder, file), target_size=IMG_SIZE)
                img = img_to_array(img) / 255.0
                X.append(img)
                y.append(label_map[label])
    return np.array(X), np.array(y), labels

X, y, label_names = load_dataset(EXTRACTED_PATH)
y_cat = to_categorical(y, num_classes=len(label_names))
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, stratify=y, test_size=0.2, random_state=42)
y_train = np.argmax(y_train_cat, axis=1)
y_test = np.argmax(y_test_cat, axis=1)

# === CLASS WEIGHTS ===
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

# === AUGMENTATION ===
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === METRIC PLOTTING ===
def plot_metrics(history, title):
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(["accuracy", "loss"]):
        plt.subplot(1, 2, i + 1)
        if metric in history.history:
            plt.plot(history.history[metric], label=f"train {metric}")
            plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
            plt.title(f"{metric}")
            plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# === MODEL DEFINITIONS ===
def build_cnn(num_classes):
    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# === TRAINING FUNCTION ===
def train_and_evaluate(model, model_name):
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test_cat),
                        epochs=EPOCHS, verbose=2,
                        callbacks=[early_stop],
                        class_weight=class_weights)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(f"\n=== Classification Report: {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=label_names))
    plot_conf_matrix(y_test, y_pred, model_name)
    plot_metrics(history, model_name)

    # === SAVE MODEL ===
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# === RUN TRAINING ===
model = build_cnn(len(label_names))
train_and_evaluate(model, "Custom_CNN")
