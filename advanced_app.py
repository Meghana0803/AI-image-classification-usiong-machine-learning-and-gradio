import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tf_explain.core.grad_cam import GradCAM

# Load Model
model = tf.keras.models.load_model("image_classifier.h5")

def classify_multiple(imgs):
    results = []
    for img in imgs:
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)
        prediction = model.predict(img_array)
        confidence = np.max(prediction) * 100
        class_id = np.argmax(prediction)
        results.append(f"Class {class_id} (Confidence: {confidence:.2f}%)")
    return results

def explain(img):
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    explainer = GradCAM()
    grad_cam_result = explainer.explain((img_array, None), model)

    return grad_cam_result

interface = gr.Interface(
    fn=classify_multiple,
    inputs=gr.Image(type="pil", multiple=True),
    outputs="text",
    title="🔍 AI Image Classifier with Grad-CAM",
    description="Upload images for classification & explanation!"
)

interface.launch(share=True, debug=True)
