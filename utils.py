import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.backend import clear_session


IMG_SIZE = (224, 224)
LABELS = ['books', 'coins', 'drumsticks', 'stamps', 'vinyls']

clear_session()
model = load_model("best_custom_cnn.keras")

def predict_category(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    label_idx = np.argmax(prediction)
    label = LABELS[label_idx]
    return label, prediction[0][label_idx]
