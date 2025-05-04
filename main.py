from Facial_size.size_calculations import *
import csv
import os
from loguru import logger

def process_directory(directory_path, csv_path):
    header = [
        "Image",
        "Right Eye Width", "Right Eye Height",
        "Left Eye Width", "Left Eye Height",
        "Forehead Size",
        "Nose to Lip", "Lip to Chin", "Golden Ratio"
    ]

    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for filename in os.listdir(directory_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(directory_path, filename)
                try:
                    re_w, re_h, le_w, le_h = eye_size(img_path)
                    nose_lip, lip_chin, golden = nose_lips_chin(img_path)
                    forehead_size = forehead(img_path)
                    writer.writerow([filename, re_w, re_h, le_w, le_h, forehead_size, nose_lip, lip_chin, golden])
                    logger.debug(f"Processed {filename}")
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
if __name__ == "__main__":