import tensorflow as tf
import cv2
import numpy as np
import json

# Load and parse the JSON data
with open('processed_image.json', 'r') as json_file:
    data = json.load(json_file)

# Convert the list data to a tensor
tensor = tf.constant(data, dtype=tf.float32)

reconstructed_tensor = tf.reshape(tensor, [128, 128, 3])

# Convert tensor to numpy array
array = reconstructed_tensor.numpy()

# Rescale the pixel values from [0, 1] to [0, 255]
array = (array * 255).astype(np.uint8)

# PC SPECIFIC
# Save the image using OpenCV
cv2.imwrite('ProcessedImagesVisualized/output_image21_128.png', cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
