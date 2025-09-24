import cv2
import numpy as np
import os
from pathlib import Path

def create_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold ranges for disease spots (brown/yellow/black)
    lower = np.array([5, 40, 40])     # lower HSV bound (adjustable)
    upper = np.array([30, 255, 255])  # upper HSV bound

    mask = cv2.inRange(hsv, lower, upper)

    # Smoothen mask
    mask = cv2.medianBlur(mask, 5)

    cv2.imwrite(mask_path, mask)

def process_dataset(img_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for img_file in Path(img_dir).rglob("*.jpg"):
        rel_path = img_file.relative_to(img_dir)
        mask_file = Path(mask_dir) / rel_path
        os.makedirs(mask_file.parent, exist_ok=True)
        create_mask(str(img_file), str(mask_file))

if __name__ == "__main__":
    plantvillage_path = "datasets/plantvillage/train"
    masks_output = "datasets/plantvillage_masks/train"
    process_dataset(plantvillage_path, masks_output)

    plantvillage_path = "datasets/plantvillage/val"
    masks_output = "datasets/plantvillage_masks/val"
    process_dataset(plantvillage_path, masks_output)

    plantvillage_path = "datasets/plantvillage/test"
    masks_output = "datasets/plantvillage_masks/test"
    process_dataset(plantvillage_path, masks_output)

    print("✅ PlantVillage masks generated!")
