Installation (Windows PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
# Activate the venv in PowerShell
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install runtime deps

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional) If you want DeepFace emotion detection, install a backend — TensorFlow is common:

```powershell
# CPU-only TensorFlow (smaller):
pip install tensorflow-cpu
# or full TensorFlow (may require GPU/CUDA):
pip install tensorflow
```

4) Pin the tested versions (recommended before deploying):

```powershell
pip freeze > requirements.txt
```

Notes
- `utils.py` is a local helper module and does not appear in `requirements.txt`.
- If `opencv-python` gives trouble on Windows, try installing pre-built wheels or use `pip install opencv-python-headless` if you don't need GUI features.
- If you need a dev/test dependency file, create `requirements-dev.txt` and list packages like `pytest`, `black`, etc.
