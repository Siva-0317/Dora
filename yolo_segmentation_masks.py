import cv2
import numpy as np
import os
from pathlib import Path

def yolo_to_mask(img_path, label_path, mask_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.split())
            # Convert YOLO (cx, cy, bw, bh) → pixel coordinates
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    cv2.imwrite(mask_path, mask)

def process_ip102(img_dir, label_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for img_file in Path(img_dir).rglob("*.jpg"):
        rel_path = img_file.relative_to(img_dir)
        label_file = Path(label_dir) / rel_path.with_suffix(".txt")
        mask_file = Path(mask_dir) / rel_path.with_suffix(".png")
        os.makedirs(mask_file.parent, exist_ok=True)
        yolo_to_mask(str(img_file), str(label_file), str(mask_file))

if __name__ == "__main__":
    process_ip102("datasets/ip102/images/train", "datasets/ip102/labels/train", "datasets/ip102_masks/train")
    process_ip102("datasets/ip102/images/val", "datasets/ip102/labels/val", "datasets/ip102_masks/val")
    #process_ip102("datasets/ip102/images/test", "datasets/ip102/labels/test", "datasets/ip102_masks/test")

    print("✅ IP102 masks generated!")
