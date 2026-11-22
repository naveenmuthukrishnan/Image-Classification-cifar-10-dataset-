# PixelSense

A small Flask web app that classifies an uploaded image into one of the ten CIFAR-10 categories using a trained Keras model. Upload a picture, and the app shows what it thinks the image is, along with a confidence breakdown across all ten classes.

This is a proof of concept for deploying a CIFAR-10 image classifier behind a simple web interface.

## Features

- Upload an image (PNG, JPG, JPEG, WEBP or BMP) from the browser
- Predicts one of 10 CIFAR-10 classes and shows a confidence percentage for every class, sorted most likely first
- Shows your original image next to the 32x32 version the model actually saw
- Friendly error messages for missing files, unsupported types, corrupt images and files over 5 MB
- Model is loaded once at server start, not on every request
- Handles models that output raw logits by converting them to probabilities

## Classes

The model predicts one of these ten classes (CIFAR-10 order):

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

## Tech stack

- Python 3
- Flask for the web server and templates
- Keras / TensorFlow for model loading and inference
- Pillow for image handling
- NumPy for array processing

## Project structure

```
cifar10-flask/
├── app.py               # Flask app: config, model loading, routes
├── category_intel.h5    # Trained Keras model (must be in the same folder as app.py)
├── requirements.txt     # Python dependencies
├── static/
│   ├── css/             # Stylesheets
│   └── js/              # JavaScript
└── templates/
    ├── base.html        # Shared layout that the other pages extend
    ├── index.html       # Upload page
    └── result.html      # Prediction result page
```

## Getting started

### 1. Install dependencies

Using a virtual environment is recommended:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Add the model

Place your trained model file, `category_intel.h5`, in the same folder as `app.py`.

### 3. Run the app

```bash
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## How it works

1. The user uploads an image on the home page (`/`).
2. The form posts to `/predict`, which validates the file (present, allowed extension, readable as an image).
3. The image is converted to RGB, resized to 32x32 and, if `NORMALIZE` is on, scaled to the 0-1 range.
4. The model predicts class probabilities for the image.
5. The result page (`result.html`, which extends `base.html`) displays the top prediction, the full ranked list of classes, and both the original and 32x32 images (embedded as base64 data URIs, so nothing is saved to disk).

Visiting `/predict` directly with a GET request redirects back to the home page.

## Configuration

All settings are at the top of `app.py`.

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_PATH` | `category_intel.h5` next to `app.py` | Location of the Keras model |
| `ALLOWED_EXTENSIONS` | png, jpg, jpeg, webp, bmp | Accepted upload types |
| `IMG_SIZE` | `(32, 32)` | Model input size (width, height) |
| `NORMALIZE` | `True` | Divide pixels by 255 before prediction |
| `MAX_CONTENT_LENGTH` | 5 MB | Maximum upload size |
| `SECRET_KEY` | `change-me-in-production` | Needed for flash messages; set the `SECRET_KEY` environment variable |

**Important:** `NORMALIZE` must match how the model was trained. If you trained on `x_train / 255.0`, keep it `True`. If you trained on raw 0-255 pixels, set it to `False`. A mismatch is the most common cause of bad predictions.

## Troubleshooting

**`Unrecognized keyword arguments ... quantization_config` when loading the model**
This happens when a model saved with a newer Keras version (for example, trained in Google Colab) is opened with an older Keras. `app.py` already handles this with a `CompatDense` layer that drops the extra key, so no action is needed. If you see the error, make sure `custom_objects={"Dense": CompatDense}` is still passed to `load_model`.

**Predictions are always wrong or always the same class**
Check that `NORMALIZE` matches your training setup, and that the `CLASSES` list order matches the class ids your model was trained with.

**Model file not found**
Confirm `category_intel.h5` is in the same folder as `app.py`.

## Limitations

- The model only knows the 10 CIFAR-10 classes, so any image is forced into one of them, even if it shows something else entirely.
- Images are downscaled to 32x32, so accuracy is limited, especially for photos with small or off-centre subjects.
- Confidence percentages show how the model distributes probability across the classes, not a guarantee of correctness.

## Deployment notes

This is a proof of concept. Before using it beyond local testing:

- Turn off `debug=True` in `app.run(...)`
- Set a strong `SECRET_KEY` through the environment
- Serve the app with a production WSGI server such as Gunicorn or Waitress rather than the Flask development server
