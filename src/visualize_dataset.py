from pathlib import Path
from collections import Counter
import random 

import cv2 
import matplotlib.pyplot as plt


Data_dir = Path("Data")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class_count = Counter()

samples = {}

for class_dir in sorted(Data_dir.iterdir()):
    if not class_dir.is_dir():
        continue

    images = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]
    class_count[class_dir.name] = len(images)
    samples[class_dir.name] = random.sample(
        images, min(4, len(images))
    )

plt.figure(figsize=(9,5))
plt.bar(class_count.keys(), class_count.values())
plt.title("Number of MRI")
plt.xlabel("Class")
plt.ylabel("Numbers of images")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    len(samples), 4, figsize=(12, 3 * len(samples))
)

for row, (class_name, paths) in enumerate(samples.items()):
    for col in range(4):
        ax = axes[row, col]

        if col >= len(paths):
            ax.axis("off")
            continue

        image = cv2.imread(str(paths[col]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

plt.tight_layout()
plt.show()
