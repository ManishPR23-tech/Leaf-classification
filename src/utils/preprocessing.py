import os
import numpy as np
from PIL import Image

def preprocess_image(image_path, size=(224, 224)):
    with Image.open(image_path) as original:
        rgba = original.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background.resize(size)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    return image_array
