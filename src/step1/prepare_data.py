from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

DATA_DIR = Path("Data")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = []
labels = []

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            image_paths.append(str(image_path))
            labels.append(class_dir.name)

image_paths = np.array(image_paths)
labels = np.array(labels)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

train_paths, temp_paths, y_train, y_temp = train_test_split(
    image_paths,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

val_paths, test_paths, y_val, y_test = train_test_split(
    temp_paths,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Classes:", list(encoder.classes_))
print("Total images:", len(image_paths))
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))
print("Image size for CNN: 224x224")
