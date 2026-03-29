
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

concrete_model = load_model("concrete_detector_v2.h5")
crack_model = tf.keras.models.load_model("crack_detector_v3.keras")

def detect_cracks(image):
    # Resize for classification
    img_resized = cv2.resize(image, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Step 1: Is it concrete?
    concrete_pred = concrete_model.predict(img_input, verbose=0)[0][0]
    is_concrete = concrete_pred < 0.5

    if not is_concrete:
        return image, "NOT A CONCRETE SURFACE\nThis image does not appear to be a concrete or infrastructure surface."

    # Step 2: Does it have cracks? (EfficientNetB0 expects 224x224, raw 0-255)
    img_224 = cv2.resize(image, (224, 224)).astype(np.float32)
    img_224_input = np.expand_dims(img_224, axis=0)
    img_224_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_224_input)
    
    crack_pred = crack_model.predict(img_224_preprocessed, verbose=0)[0][0]
    # Cracked = class 0, NotCracked = class 1
    has_crack = crack_pred < 0.5
    confidence = (1 - crack_pred) if has_crack else crack_pred

    if not has_crack:
        return image, f"NO CRACK DETECTED\nConcrete surface identified. No cracks found.\nConfidence: {confidence*100:.1f}%"

    # Step 3: Visualize cracks
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (512, 512))
    img_display = cv2.resize(image, (512, 512))

    blurred = cv2.GaussianBlur(img_gray, (15, 15), 3)
    blurred_normalized = ((blurred.astype(np.float32) - blurred.min()) /
                         (blurred.max() - blurred.min()) * 255).astype(np.uint8)

    edges = cv2.Canny(blurred_normalized, 20, 80)
    img_highlighted = img_display.copy()
    img_highlighted[edges > 0] = [255, 0, 0]

    return img_highlighted, f"CRACK DETECTED\nCracks highlighted in red.\nConfidence: {confidence*100:.1f}%"

demo = gr.Interface(
    fn=detect_cracks,
    inputs=gr.Image(type="numpy", label="Upload Infrastructure Image"),
    outputs=[
        gr.Image(type="numpy", label="Result"),
        gr.Textbox(label="Detection Result")
    ],
    title="CUDA-Accelerated Infrastructure Crack Detection",
    description="Upload an image of a concrete surface. The system uses EfficientNetB0 transfer learning trained on 126,000+ real infrastructure images to detect cracks with 98.14% accuracy.",
)

demo.launch()
