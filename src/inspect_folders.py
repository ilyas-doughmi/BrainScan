from pathlib import Path
from collections import Counter
import cv2


Extensions = {".jpg",".jpeg",".bmp",".png"}
Dir = Path("Data")


if not Dir.is_dir():
    raise FileNotFoundError("Dataset folder not found \n")


clases=  sorted(path for path in Dir.iterdir() if path.is_dir())


if not clases:
    raise ValueError("No class found")

total_readable = 0
total_corrupted = 0
total_unsupporter = 0 

extensions_count = Counter()

for class_dir in clases:
    files = [p for p in class_dir.rglob("*") if p.is_file()]

    images = [
        p for p in files
        if p.suffix.lower() in Extensions
    ]

    unspported = [
        p for p in files
        if p.suffix.lower() not in Extensions
    ]

    corrupted = []

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            corrupted.append(image_path)

    working = len(images) - len(corrupted)

    total_readable += working
    total_corrupted += len(corrupted)
    total_unsupporter += len(unspported)

    extensions_count.update(
        p.suffix.lower() for p in images
    )

    print(f"Class : {class_dir.name}")
    print(f"Readable : {total_readable}")
    print(f"Corrupted : {total_corrupted}")
    print(f"Unsupported : {total_unsupporter}")