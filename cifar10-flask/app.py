"""
PixelSense - CIFAR-10 image classifier (Flask POC)

Run:   python app.py      ->  http://127.0.0.1:5000
Needs: category_intel.h5 sitting next to this file.
"""
import base64
import io
import os

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from keras.layers import Dense
from keras.models import load_model
from PIL import Image, UnidentifiedImageError

# ------------------------------------------------------------------ config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "category_intel.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE = (32, 32)  # CIFAR-10 input size (width, height)

# IMPORTANT: this must match how the model was TRAINED.
# If you trained on x_train / 255.0, keep True. If you fed raw 0-255 pixels, set False.
NORMALIZE = True

# The position in this list == the class id the model predicts (CIFAR-10 order).
CLASSES = [
    {"name": "airplane",   "emoji": "✈️", "color": "#38bdf8"},
    {"name": "automobile", "emoji": "🚗", "color": "#f43f5e"},
    {"name": "bird",       "emoji": "🐦", "color": "#facc15"},
    {"name": "cat",        "emoji": "🐱", "color": "#fb923c"},
    {"name": "deer",       "emoji": "🦌", "color": "#a3e635"},
    {"name": "dog",        "emoji": "🐶", "color": "#c084fc"},
    {"name": "frog",       "emoji": "🐸", "color": "#34d399"},
    {"name": "horse",      "emoji": "🐴", "color": "#f472b6"},
    {"name": "ship",       "emoji": "🚢", "color": "#22d3ee"},
    {"name": "truck",      "emoji": "🚚", "color": "#818cf8"},
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-production")  # flash() needs this
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # reject uploads over 5 MB

class CompatDense(Dense):
    """Lets an OLDER Keras open a model saved by a NEWER Keras (e.g. trained in Colab).

    Newer Keras writes an extra 'quantization_config' key into every Dense layer's saved
    config; older Keras rejects it ("Unrecognized keyword arguments ... quantization_config").
    The value is always None for a normal model, so it is safe to just drop it.
    """

    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


# Load the model ONCE when the server starts (not on every request).
# compile=False skips the "compiled metrics" warning - we only need predict().
model = load_model(MODEL_PATH, compile=False, custom_objects={"Dense": CompatDense})


# ----------------------------------------------------------------- helpers
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(img: Image.Image):
    """PIL image -> (batch of shape (1, 32, 32, 3), the 32x32 PIL image)."""
    small = img.resize(IMG_SIZE)
    arr = np.array(small, dtype="float32")
    if NORMALIZE:
        arr = arr / 255.0  # assign it back, otherwise the division is thrown away
    return np.expand_dims(arr, axis=0), small


def to_data_uri(img: Image.Image) -> str:
    """Turn a PIL image into a base64 string an <img src="..."> can show directly."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Someone typed /predict in the address bar (a GET) - send them home.
    if request.method == "GET":
        return redirect(url_for("index"))

    file = request.files.get("image")  # "image" = the name="" of the <input type="file">

    if file is None or file.filename == "":
        flash("Please choose an image first.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, WEBP or BMP.", "error")
        return redirect(url_for("index"))

    try:
        # convert("RGB") forces 3 colour channels (drops alpha, expands grayscale)
        img = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        flash("That file doesn't look like a valid image.", "error")
        return redirect(url_for("index"))

    batch, small = preprocess(img)
    probs = model.predict(batch, verbose=0)[0]

    # Safety net: if the model outputs raw logits instead of softmax, convert them.
    if not np.isclose(probs.sum(), 1.0, atol=1e-3):
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    order = np.argsort(probs)[::-1]  # class ids, most likely first
    results = [{**CLASSES[i], "pct": round(float(probs[i]) * 100, 1)} for i in order]

    preview = img.copy()
    preview.thumbnail((512, 512))  # keep the page light

    return render_template(
        "result.html",
        prediction=results[0],
        results=results,
        original_uri=to_data_uri(preview),
        model_uri=to_data_uri(small),
    )


@app.errorhandler(413)
def file_too_large(_err):
    flash("That file is too big - please keep it under 5 MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
