# https://chatgpt.com/c/6734c8e2-965c-8006-92a8-22bc6961e561

import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare


# Load PyTorch model
model = torch.load("/Users/ivanursul/git/fall-detection-phd/training/variations/performer/performer_model.pt")
model.eval()  # Set model to evaluation mode

try:
    # Try a standard shape, like for image data (batch size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust dimensions based on your needs
    model(dummy_input)
    print("This shape is compatible with the model.")
except Exception as e:
    print("Error:", e)

# Define a dummy input based on the model's expected input shape
# Replace `input_shape` with the correct input dimensions of your model
input_shape = (3, 224, 224)
dummy_input = torch.randn(1, *input_shape)

# Step 1: Convert PyTorch model to ONNX format
onnx_path = "performer_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True)

# Step 2: Convert ONNX model to TensorFlow SavedModel format
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_saved_model_path = "performer_model.pb"
tf_rep.export_graph(tf_saved_model_path)

# Step 3: Convert TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
# Optional: Optimize the model for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite format
tflite_model = converter.convert()
tflite_model_path = "performer_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted to TFLite format: {tflite_model_path}")