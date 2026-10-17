
import numpy as np
from src.utils.preprocessing import preprocess_image

image_path = "data/raw/test_leaf.jpg.png"

result = preprocess_image(image_path)

print("Image processed successfully!")
print("Array shape:", result.shape)
print("Data type:", result.dtype)
print("Minimum pixel value:", result.min())
print("Maximum pixel value:", result.max())

assert result.shape == (224, 224, 3)
assert result.min() >= 0.0
assert result.max() <= 1.0

print("All tests passed!")
