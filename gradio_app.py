
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("image_classifier.h5")

def classify_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return f"Prediction: Class {class_id} ({confidence:.2f}% confidence)"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🔍 AI Image Classifier",
    description="Upload an image and let AI predict the class!"
)

interface.launch()
