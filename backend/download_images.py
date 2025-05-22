import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np

# Argument parser for input TSV file
parser = argparse.ArgumentParser(description='Image Downloader for Fake News Dataset')
parser.add_argument('file', type=str, help='D:/MultimodalFakeNewsAI/data/multimodal_train.tsv')
args = parser.parse_args()

# Load and clean DataFrame
df = pd.read_csv(args.file, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# Image folder (relative to script location or full path)
image_dir = "D:/MultimodalFakeNewsAI/data/images"
os.makedirs(image_dir, exist_ok=True)

num_failed = 0
pbar = tqdm(total=len(df), desc="Downloading images")

for index, row in df.iterrows():
    has_image = row.get("hasImage", False)
    image_url = row.get("image_url", "")
    image_id = row.get("id", "")

    if has_image and image_url and image_id:
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        try:
            if not os.path.exists(image_path):
                urllib.request.urlretrieve(image_url, image_path)
        except Exception as e:
            num_failed += 1
            print(f"❌ Failed: {image_url} ({e})")
    pbar.update(1)

pbar.close()
print("✅ Download complete.")
print(f"❌ Failed downloads: {num_failed}")
