import onnxruntime as ort
from PIL import Image
import numpy as np
from scipy.io import savemat  # Add this import to save .mat files

# Load the ONNX model
onnx_model_path = "cifar10_model.onnx"  # Path to the ONNX file
session = ort.InferenceSession(onnx_model_path)

# CIFAR-10 class labels
cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load and preprocess the image
image_path = "bird.jpg"  # Replace with the path to your test image
img = Image.open(image_path).resize((32, 32))  # Resize to 32x32 pixels
img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

# Ensure the image is in the correct format (batch_size, height, width, channels)
if len(img.shape) == 2:  # Convert grayscale to RGB
    img = np.stack((img,) * 3, axis=-1)
img = np.expand_dims(img.astype(np.float32), axis=0)  # Add batch dimension

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predictions = session.run([output_name], {input_name: img})[0]

# Get the predicted label
predicted_index = np.argmax(predictions)
predicted_label = cifar10_labels[predicted_index]

# Save the results to a .mat file
result = {
    "predicted_index": int(predicted_index),
    "predicted_label": predicted_label,
    "predictions": predictions.tolist(),  # Save all class probabilities
    "cifar10_labels": cifar10_labels      # Save class labels for reference
}
savemat("prediction_result.mat", result)  # Save results to 'prediction_result.mat'

# Display the result in the Python console
print(f"Predicted Label: {predicted_label}")
print("Results saved to 'prediction_result.mat'")
