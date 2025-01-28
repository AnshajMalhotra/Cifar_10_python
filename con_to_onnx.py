import tensorflow as tf
import tf2onnx

# Load the Keras model
keras_model = tf.keras.models.load_model('cifar10_model.h5')

# Convert to ONNX format
onnx_model_path = "cifar10_model.onnx"
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, opset=13)  # Unpack the tuple

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model successfully converted and saved to {onnx_model_path}")
