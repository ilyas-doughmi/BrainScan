from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (224, 224)


def load_image(image_path):
    try:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("image not found")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype(np.float32) / 255.0

        return image

    except (ValueError, cv2.error) as error:
        print(f"Error loading {image_path}: {error}")
        return None
