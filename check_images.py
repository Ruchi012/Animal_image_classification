import os
from PIL import Image
import tensorflow as tf

DATASET_DIR = "dataset"
LOG_FILE = "image_check_log.txt"

lines = []

def log(msg):
    print(msg, flush=True)
    lines.append(str(msg))

log("Starting image check (using TensorFlow's own decoder this time)...")
log(f"Dataset folder: {os.path.abspath(DATASET_DIR)}")

bad_files = []
checked = 0

for root, _, files in os.walk(DATASET_DIR):
    for name in files:
        path = os.path.join(root, name)
        checked += 1
        try:
            raw = tf.io.read_file(path)
            # channels=0 means "keep whatever the file declares" -- this is
            # what image_dataset_from_directory does internally, so it will
            # fail on exactly the same files that break train.py
            img = tf.io.decode_image(raw, channels=0, expand_animations=False)
            n_channels = img.shape[-1]
            if n_channels not in (1, 3, 4):
                bad_files.append((path, n_channels))
        except Exception as e:
            bad_files.append((path, f"DECODE ERROR: {e}"))

        if checked % 200 == 0:
            log(f"  ...checked {checked} so far")

log(f"Checked {checked} files total.")

if not bad_files:
    log("TensorFlow decoder found no problem files. (Unexpected -- see note below.)")
else:
    log(f"Found {len(bad_files)} problem file(s) according to TensorFlow's decoder:")
    for path, info in bad_files:
        log(f"  {path}  ->  {info}")

    log("Force-fixing all flagged files by re-saving as clean RGB JPEG...")
    fixed = 0
    for path, info in bad_files:
        try:
            with Image.open(path) as im:
                im.convert("RGB").save(path)
            fixed += 1
        except Exception as e:
            log(f"  Could not fix {path}: {e}")
    log(f"Fixed {fixed} of {len(bad_files)} flagged file(s).")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nFull log written to {LOG_FILE}")