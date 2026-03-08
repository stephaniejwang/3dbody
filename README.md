# 3D Body Measurement Tool

Browser-based 3D body measurement tool for fashion/apparel sizing. Upload a video with a reference object, get an interactive 3D body mesh with real-world measurements.

## Licensing

All dependencies are **Apache 2.0, MIT, CC0, or BSD** licensed. No SMPL/SMPL-X. Safe for commercial use.

| Component | License |
|-----------|---------|
| Anny v0.3 (body model) | Apache 2.0 + CC0 |
| MediaPipe (pose estimation) | Apache 2.0 |
| OpenCV ArUco (calibration) | Apache 2.0 |
| RT-DETR via HuggingFace (object detection) | Apache 2.0 |
| Three.js (3D rendering) | MIT |
| FastAPI (backend) | MIT |

## Quick Start (Local)

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run (serves both API and frontend)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/stephaniejwang/3dbody)

Or manually:
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo `stephaniejwang/3dbody`
3. Settings will auto-detect from `render.yaml`
4. Click **Deploy**

## Deploy with Docker

```bash
docker build -t 3dbody .
docker run -p 8000:8000 3dbody
```

## Setup Details

### No model downloads needed

Unlike SMPL-based tools, this project uses:
- **Anny** — installs via pip, assets bundled in the package
- **MediaPipe** — downloads pose model automatically on first run
- **RT-DETR** — downloads from HuggingFace automatically on first run (only needed for A4/credit card reference modes)

No manual model weight downloads or `models/` directory setup required.

If `anny` fails to install from PyPI:

```bash
pip install anny@git+https://github.com/naver/anny.git
```

## Usage

1. **Enter your height** — Improves accuracy with any reference method
2. **Print an ArUco marker** (optional) — Visit the ArUco page for a printable 10cm marker
3. **Record a video** — Wear form-fitting clothing, hold the reference object visible, and walk slowly in a circle
4. **Upload** — Select the video and reference method, then upload
5. **View results** — Interact with the 3D mesh and review measurements
6. **Refine** — Enter your height on the results page to recalibrate

### Reference Methods (ranked by accuracy)

1. **ArUco Marker + Height** — Print the provided marker, hold it in frame + enter height. Most accurate.
2. **ArUco Marker** — Print the provided marker, hold it in frame.
3. **A4/Letter Paper** — Hold a standard sheet of paper in frame.
4. **Credit Card** — Hold a credit card in frame.
5. **Height Only** — Select "No Reference Object" and enter your height.

## Architecture

```
Video Upload
  → Extract keyframes (every 10 frames, max 30)
  → Scale calibration (ArUco / RT-DETR / height)
  → MediaPipe pose detection on best frame
  → Estimate body phenotype from pose proportions
  → Anny generates parametric body mesh (T-pose)
  → Filter eyeball/mouth geometry from mesh
  → Anny Anthropometry + geometric measurements
  → Return mesh + measurements to frontend

Frontend renders in Three.js with measurement overlays.
Height recalibration available post-processing.
```

### Key Design: Anny as Parametric Body Model

Anny is a differentiable human body model that takes **phenotype parameters** (age, gender, height, weight, muscle, proportions) in [0, 1] range and outputs a 3D mesh. Unlike SMPL, Anny:
- Is fully Apache 2.0 licensed (commercial use OK)
- Has 13,718 vertices and 13,710 quad faces (native topology)
- Includes built-in `Anthropometry` class for height, mass, waist, BMI
- Outputs in **meters**, with **Z-axis up**

MediaPipe detects 2D/3D pose landmarks from video, which we use to estimate body proportions and map them to Anny phenotype parameters.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model load status |
| GET | `/` | Frontend (index.html) |
| POST | `/upload` | Upload video + reference mode, returns `{ job_id }` |
| GET | `/status/{job_id}` | Job status + results when done |
| POST | `/recalibrate` | Re-scale mesh using known height |

## Notes

- SMPL and SMPL-X are intentionally **not** used due to licensing restrictions
- Anny v0.3 uses its own native topology (not the smplx variant, which is non-commercial only)
- No YOLOv8/Ultralytics (AGPL) — RT-DETR via HuggingFace Transformers is used instead
- No auth, no database — stateless per job
- Video files are deleted after processing (privacy)
- Eyeball and inner-mouth geometry filtered from mesh for clean rendering
