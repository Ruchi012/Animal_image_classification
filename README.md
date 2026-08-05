# Animal Image Classification — Cat vs Dog 🐱🐶

A web app that classifies uploaded photos as **Cat** or **Dog** using a
transfer-learning model built on **MobileNetV2**, served through a
**Flask** front end and deployed on **Render**.

Upload a photo → the model predicts the animal → the result and
confidence score are shown in the UI.

---

## Live Demo

[https://animal-image-classification-qc9u.onrender.com](https://animal-image-classification-qc9u.onrender.com)

> Hosted on Render's free tier — the app may take a few seconds to wake
> up if it's been idle.

---

## How It Works

1. **Training** (`train.py`) — A MobileNetV2 backbone (pretrained on
   ImageNet) is frozen, and a small classifier head is trained on top of
   it using a labeled `dataset/cat` / `dataset/dog` image folder. Data
   augmentation (flip, rotation, zoom, contrast) and dropout help the
   model generalize despite a modest dataset size. Training uses
   `EarlyStopping` to stop once validation accuracy plateaus, restoring
   the best-performing weights.
2. **Export** — The trained Keras model (`cat_dog_model.keras`) is
   converted to a **TFLite** model (`cat_dog_model.tflite`) for a much
   smaller runtime memory footprint — this lets the app run inference on
   Render's free-tier instance without needing the full TensorFlow
   package installed in production.
3. **Serving** (`app.py`) — A Flask app loads the `.tflite` model once
   at startup via `tflite-runtime`, accepts an uploaded image, resizes
   and preprocesses it, runs inference, and renders the result.

---

## Project Structure

```
Animal_image_classification/
├── app.py                   # Flask web app (serves predictions)
├── train.py                 # Model training script (MobileNetV2 transfer learning)
├── convert_to_tflite.py     # Converts the trained .keras model to .tflite
├── cat_dog_model.keras      # Trained Keras model (used locally / for conversion)
├── cat_dog_model.tflite     # Lightweight model used in production
├── requirements.txt         # Python dependencies
├── runtime.txt              # Pinned Python version for deployment
├── dataset/                 # Training images (cat/, dog/ subfolders)
├── static/
│   ├── style.css
│   └── uploads/             # User-uploaded images (created at runtime)
├── templates/
│   └── index.html           # Main page template
└── test.jpg                 # Sample image for local testing
```

---

## Model Details

| | |
|---|---|
| **Base architecture** | MobileNetV2 (ImageNet weights, frozen) |
| **Input size** | 128 × 128 × 3 |
| **Classifier head** | GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid) |
| **Loss** | Binary crossentropy |
| **Optimizer** | Adam (lr = 1e-4) |
| **Classes** | `cat` (0), `dog` (1) — alphabetical order |
| **Decision threshold** | 0.5 (score > 0.5 → Dog, else → Cat) |
| **Best validation accuracy** | ~98.3% |

**Important:** the model's first layer (`Rescaling(scale=1/127.5, offset=-1)`)
performs its own pixel normalization to match what MobileNetV2 expects.
Inference code must pass **raw 0–255 pixel values** — do not divide by
255 before calling the model, or predictions will be badly skewed.

---

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Ruchi012/AI_RESUME_SCREENING.git
cd Animal_image_classification
```

### 2. Set up a virtual environment (Python 3.11 recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```
Visit **http://127.0.0.1:5000** in your browser.

---

## Retraining the Model

If you want to train on your own dataset:

1. Organize your images as:
   ```
   dataset/
   ├── cat/
   │   ├── img001.jpg
   │   └── ...
   └── dog/
       ├── img001.jpg
       └── ...
   ```
2. Run training:
   ```bash
   python train.py
   ```
   This saves the best model to `cat_dog_model.keras` (based on
   validation accuracy).
3. Convert it for deployment:
   ```bash
   python convert_to_tflite.py
   ```
   This produces `cat_dog_model.tflite`, which `app.py` actually loads
   at runtime.
4. Commit and push both files if deploying:
   ```bash
   git add cat_dog_model.keras cat_dog_model.tflite
   git commit -m "Retrain model"
   git push
   ```

---

## Deployment (Render)

This app is deployed as a **Web Service** (not a Static Site — the app
needs a running Python process to serve predictions).

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120` |
| **Root Directory** | *(blank)* |
| **Python version** | Pinned via `runtime.txt` / `PYTHON_VERSION` env var |

### Why TFLite instead of full TensorFlow in production?
Render's free tier caps instances at 512 MB RAM. Loading full
TensorFlow plus a Keras model at startup can exceed that limit,
crashing the app with an out-of-memory error. Using `tflite-runtime`
for inference dramatically reduces the memory footprint, allowing the
app to run reliably on the free tier.

### Key dependency notes
- `tflite-runtime` requires **NumPy < 2** — it was compiled against the
  NumPy 1.x ABI and will fail to import under NumPy 2.x.
- Pin `PYTHON_VERSION` (via `runtime.txt` or Render's Environment tab)
  to a version TensorFlow/TFLite actually supports — very new Python
  releases often don't have compatible wheels yet.

---

## Tech Stack

- **Model**: TensorFlow / Keras, MobileNetV2 transfer learning
- **Inference runtime**: `tflite-runtime`
- **Backend**: Flask, Gunicorn
- **Frontend**: HTML/CSS (Jinja2 templates)
- **Hosting**: Render

---

## Known Limitations

- Trained on a modest dataset — accuracy may drop on unusual poses,
  lighting, or breeds underrepresented in training data.
- Free-tier hosting may sleep after inactivity; the first request after
  idle time can take longer to respond.

---

## License

MIT
