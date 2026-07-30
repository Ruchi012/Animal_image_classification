import os
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --- Config -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.keras")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
TARGET_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

print("Looking for model at:", MODEL_PATH)
print("Exists?", os.path.exists(MODEL_PATH))

# --- Load model once at startup ----------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path: str):
    """Runs the same preprocessing pipeline used during training / main.py."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=TARGET_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0

    raw_score = float(model.predict(arr, verbose=0)[0][0])

    # dataset/ folders are "cat" then "dog" alphabetically -> class 0 = cat, class 1 = dog
    label = "Dog" if raw_score > 0.5 else "Cat"
    confidence = raw_score if raw_score > 0.5 else 1 - raw_score
    return label, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Choose an image first."
        elif not allowed_file(file.filename):
            error = "That file type isn't supported. Use JPG, PNG, or WEBP."
        else:
            ext = file.filename.rsplit(".", 1)[1].lower()
            filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                label, confidence = predict(save_path)
                result = {"label": label, "confidence": round(confidence * 100, 1)}
                image_url = f"/static/uploads/{filename}"
            except Exception as exc:  # keep the app alive on a bad file
                error = f"Couldn't read that image: {exc}"

    return render_template("index.html", result=result, image_url=image_url, error=error)


if __name__ == "__main__":
    app.run(debug=True)