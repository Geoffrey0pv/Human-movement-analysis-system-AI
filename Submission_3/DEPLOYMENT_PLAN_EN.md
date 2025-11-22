# Human Movement Analysis – Deployment Plan

This document describes the deployment plan for the human–movement classification system developed in the project. The goal is to provide a stable, reproducible and easy–to–maintain web application based on Hugging Face Spaces and on an XGBoost classifier trained on pose data extracted with MediaPipe.

---

## 1. Deployment goal

- Provide a **web application** that allows users to:
  - Record or upload a short video.
  - Process the video frame by frame to extract body pose.
  - Classify each frame into one of five actions: `caminar_adelante`, `caminar_atras`, `girar`, `pararse`, `sentarse`.
  - Display a **summary of the results** (dominant action and class distribution).

- Target environment: **academic / laboratory**, for demonstration and experimentation. The system is **not** intended for clinical or production use.

---

## 2. High–level architecture

### 2.1 Main components

1. **Code repository (Space)**  
   - Hugging Face Space: `RaulQode/human-movement-analysis`  
   - Contains:
     - `app.py` – inference logic and Gradio interface definition.
     - `requirements.txt` – Python dependencies.
     - `README.md` – documentation (this file).
     - Optional auxiliary files (notebooks, figures). No heavy binary files.

2. **Model / dataset repository**  
   - Hugging Face dataset: `RaulQode/model_humanmov`  
   - Contains:
     - `modelo_acciones.pkl` – pickled dictionary with:
       - `model` (trained XGBoost classifier),
       - `label_encoder` (label encoder),
       - basic metadata (classes, version).

3. **Inference service**  
   - The Space runs `app.py`, which:
     - Downloads the model from the dataset using `hf_hub_download`.
     - Exposes a Gradio interface based on **video upload/recording + Submit button**.
     - Processes videos with OpenCV + MediaPipe and applies the model.

4. **End users**  
   - Access the public URL of the Space via a web browser.
   - No local dependency installation is required.

### 2.2 Functional flow

1. The user opens the Space URL.
2. The user records or uploads a short video containing movements.
3. The user clicks **Submit**.
4. `app.py`:
   - Reads the video with `cv2.VideoCapture`.
   - Extracts frames and applies MediaPipe Pose.
   - Builds a feature vector for each frame.
   - Obtains predictions from the XGBoost model.
   - Aggregates results (class counts, dominant action).
5. Gradio displays:
   - The input video (optional).
   - A text summary with the dominant action and the class distribution.

This flow avoids the complexity and fragility of real–time webcam streaming and is more robust across browsers and environments.

---

## 3. Software requirements

Dependencies are managed through `requirements.txt`. A recommended configuration is:

```txt
gradio==5.1.0
mediapipe
opencv-python-headless
numpy
joblib
huggingface_hub>=0.30
scikit-learn
xgboost
```

Notes:

- `opencv-python-headless` is used because the Hugging Face runtime does not provide a graphical server.
- A recent version of `gradio` is fixed to avoid incompatibilities with `huggingface_hub`.
- For local development, Python 3.10 and a virtual environment are recommended.

Local installation example:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Recommended repository structure (Space)

A minimal structure for the Space repository is:

```text
human-movement-analysis/
├─ app.py
├─ requirements.txt
├─ README.md
└─ notebooks/              # (optional) notebooks and exploratory analysis
```

- Training data and the trained model are stored **outside** this repo, in the Hugging Face dataset `RaulQode/model_humanmov`.
- Heavy `.pkl` files and raw videos should not be committed to this repository to avoid size limits and policy issues.

---

## 5. Model loading from the dataset

Inside `app.py`, the model is loaded with `hf_hub_download`:

```python
from huggingface_hub import hf_hub_download
import joblib

MODEL_REPO = "RaulQode/model_humanmov"
MODEL_FILE = "modelo_acciones.pkl"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="dataset",
)
model_data = joblib.load(model_path)
model = model_data["model"]
label_encoder = model_data["label_encoder"]
```

This design allows updating the model in the dataset without changing the Space structure, as long as the filename or the `MODEL_FILE` value is updated accordingly.

---

## 6. Deployment on Hugging Face Spaces

### 6.1 Basic deployment workflow

1. **Clone the Space repository locally**  
   ```bash
   git clone https://huggingface.co/spaces/RaulQode/human-movement-analysis
   cd human-movement-analysis
   ```

2. **Develop and test locally**  
   - Edit `app.py`, `requirements.txt` and other necessary files.
   - Run the application locally:
     ```bash
     python app.py
     ```
   - Open the URL printed by Gradio (typically `http://localhost:7860`).

3. **Commit and push to the Space**  
   ```bash
   git add .
   git commit -m "Describe the change"
   git push
   ```

4. **Automatic rebuild**  
   - Hugging Face rebuilds the container using the new `requirements.txt` and code.
   - Build status can be inspected in the **“Build logs”** tab of the Space.

5. **Verification in production**  
   - Open the Space in the browser.
   - Test with several sample videos, checking that:
     - Upload and processing complete without errors.
     - Response time is reasonable for short videos.
     - The output is coherent with the performed movement.

### 6.2 Staging environment

For major changes, it is recommended to:

1. Create a **staging Space** (private), e.g. `RaulQode/human-movement-analysis-staging`.
2. Connect the staging Space to the same model dataset.
3. Apply and test changes there before promoting them to the public Space.
4. Once stable, replicate changes in the main Space.

This reduces the risk of leaving the public Space in a broken state.

---

## 7. Logging and monitoring

### 7.1 Logging in `app.py`

Adding log messages helps diagnose errors and measure processing time. Example:

```python
import logging, time
logging.basicConfig(level=logging.INFO)

def analyze_video(video_path):
    t0 = time.time()
    logging.info(f"New video: {video_path}")
    # processing...
    logging.info(f"Processed in {time.time() - t0:.2f} s, frames={num_frames}")
    return summary
```

In Hugging Face, these messages appear in the Space **“Logs”** tab.

### 7.2 Functional validation

Each time a new version is deployed, the following tests are recommended:

- One video per class (five simple videos).
- One mixed video containing several actions in sequence.
- Tests on different browsers (Chrome, Firefox, Edge) and operating systems.

---

## 8. Model version management

A simple versioning scheme in the model dataset is suggested, e.g.:

- `modelo_acciones_v1.pkl`
- `modelo_acciones_v2.pkl`

Recommended steps when training a new model:

1. Save the new model with a versioned filename and upload it to the dataset.
2. Update `MODEL_FILE` in `app.py` to point to the new file.
3. Test the new model in the staging Space.
4. If results are satisfactory, promote the change to the production Space and record the model version in `README.md` and/or in `model_data["version"]`.

If a new version causes issues, rollback consists simply of restoring the previous `MODEL_FILE` value and pushing a new commit.

---

## 9. Security and ethical considerations

Even in an academic setting, several points must be considered:

1. **Data privacy**  
   - Avoid uploading raw videos with identifiable faces to public repositories.
   - When storing data for analysis, prefer anonymised representations (e.g., pose landmarks only).

2. **System access**  
   - The Space can be public for demonstration purposes.
   - If more sensitive data are used, consider making the Space private and managing access via Hugging Face organisations.

3. **Responsible use**  
   - Clearly state in this `README` that the system is a proof–of–concept academic prototype and must not be used for clinical decision–making, mass surveillance or disciplinary control.

Example disclaimer:

> This project is an academic demonstration of the use of computer–vision and machine–learning techniques for recognising human movements. It has not been validated for clinical, security or decision–making applications and must not be used in such contexts.

---

## 10. Possible future extensions

In the medium term, the deployment plan could be extended in the following directions:

1. **Standalone REST API**  
   - Extract the video–analysis logic into a microservice (e.g., FastAPI) and expose an endpoint `/predict_video` that receives a file and returns a JSON response.
   - Use Gradio solely as a frontend calling the API.

2. **Result persistence**  
   - Store prediction summaries in a lightweight database (e.g., SQLite or PostgreSQL) for further analysis.

3. **Scaling**  
   - If the number of users increases significantly, migrate the backend to a more flexible infrastructure (Render, Railway, GCP, etc.), while keeping the Space as a public demo.

---

## 11. Summary

The deployment plan relies on a lightweight architecture fully supported by Hugging Face Spaces and datasets. The separation between application code and trained model enables incremental updates, while the use of a staging Space reduces the risk associated with changes in production. The video–upload–centric interaction flow, instead of real–time streaming, prioritises stability and reproducibility of the system in the academic context for which it was designed.
