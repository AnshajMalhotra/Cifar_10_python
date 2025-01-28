import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# Load the ONNX model
onnx_model_path = "cifar10_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# CIFAR-10 class labels
cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

st.title("CIFAR-10 Image Classifier")

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess the image
    img = Image.open(uploaded_file).resize((32, 32))  # Resize to 32x32
    img = np.array(img) / 255.0  # Normalize pixel values

    # Ensure the image has 3 channels (RGB)
    if len(img.shape) == 2:  # If grayscale
        img = np.stack((img,) * 3, axis=-1)
    img = np.expand_dims(img.astype(np.float32), axis=0)  # Add batch dimension

    # Debugging input details
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Input Details:")
    for input in session.get_inputs():
        print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    try:
        # Run inference
        predictions = session.run([output_name], {input_name: img})[0]
        predicted_index = np.argmax(predictions)
        predicted_label = cifar10_labels[predicted_index]

        # Display the image and prediction
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write(f"Predicted Label: {predicted_label}")
        st.bar_chart(predictions)
    except Exception as e:
        st.error(f"Error during inference: {e}")
