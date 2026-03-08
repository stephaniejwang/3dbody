"""
FastAPI backend for 3D Body Measurement Tool.

All dependencies: Apache 2.0, MIT, or BSD licensed.
NO SMPL, NO SMPL-X, NO PromptHMR, NO AGPL libraries.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calibration import calibrate_from_frames, CalibrationResult
from inference import InferenceEngine, InferenceResult
from measure import extract_measurements

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Body Measurement Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend — static files mounted at root as catch-all (must be after all API routes).
# This is configured at module level after all @app decorators, see bottom of file.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


class ReferenceMode(str, Enum):
    aruco = "aruco"
    a4 = "a4"
    credit_card = "credit_card"
    height_cm = "height_cm"


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


# In-memory job store
jobs: dict = {}

# Global inference engine (loaded once at startup)
engine: Optional[InferenceEngine] = None
engine_load_error: Optional[str] = None

# Async job queue
job_queue: asyncio.Queue = asyncio.Queue()


@app.on_event("startup")
async def startup():
    global engine, engine_load_error
    try:
        engine = InferenceEngine()
        logger.info("Inference engine loaded successfully.")
    except Exception as e:
        engine_load_error = str(e)
        logger.error(f"Engine loading failed: {engine_load_error}")

    # Start background worker
    asyncio.create_task(_job_worker())


@app.get("/health")
async def health():
    if engine is not None:
        return {"status": "ready", "models_loaded": True}
    return {
        "status": "degraded",
        "models_loaded": False,
        "error": engine_load_error,
        "instructions": "pip install anny mediapipe — see README.md for details.",
    }


@app.post("/upload")
async def upload(
    video: UploadFile = File(...),
    reference_mode: ReferenceMode = Form(...),
    height_cm: Optional[float] = Form(None),
):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Engine not loaded: {engine_load_error}. See /health.",
        )

    if reference_mode == ReferenceMode.height_cm and height_cm is None:
        raise HTTPException(
            status_code=400,
            detail="height_cm is required when reference_mode is 'height_cm'.",
        )

    filename = video.filename or ""
    if not filename.lower().endswith((".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Only mp4/mov files are accepted.")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(filename)[1]
    ) as tmp:
        content = await video.read()
        if len(content) > 200 * 1024 * 1024:
            os.unlink(tmp.name)
            raise HTTPException(status_code=400, detail="File exceeds 200MB limit.")
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.queued,
        "progress": "Queued for processing",
        "result": None,
        "error": None,
    }

    await job_queue.put({
        "job_id": job_id,
        "video_path": tmp_path,
        "reference_mode": reference_mode.value,
        "height_cm": height_cm,
    })

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    response = {"status": job["status"], "progress": job["progress"]}
    if job["status"] == JobStatus.done:
        response["result"] = job["result"]
    elif job["status"] == JobStatus.error:
        response["error"] = job["error"]
    return response


class RecalibrateRequest(BaseModel):
    job_id: str
    height_cm: float


@app.post("/recalibrate")
async def recalibrate(req: RecalibrateRequest):
    """Re-scale an existing mesh result using a known height.

    Takes the mesh from a completed job, scales it so the mesh height
    matches height_cm, then recomputes all measurements. This is the
    single biggest accuracy improvement — it anchors the proportional
    mesh to an absolute real-world scale.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded.")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[req.job_id]
    if job["status"] != JobStatus.done or job["result"] is None:
        raise HTTPException(status_code=400, detail="Job not completed yet.")

    if req.height_cm < 50 or req.height_cm > 250:
        raise HTTPException(status_code=400, detail="Height must be between 50 and 250 cm.")

    result = job["result"]

    # Get current mesh vertices (in cm)
    vertices_cm = np.array(result["mesh_vertices"], dtype=np.float32)
    faces = np.array(result["faces"], dtype=np.int32)

    # Use Anny's anthropometric height (landmark-based: crown to sole),
    # NOT the bounding box (Z-max - Z-min) which includes hair/fingertips
    # and would over-scale the mesh, inflating all measurements.
    vertices_m = (vertices_cm / 100.0).astype(np.float32)
    current_anthro = engine.get_anthropometry(vertices_m)
    current_height_cm = current_anthro["height_m"] * 100.0

    if current_height_cm <= 0:
        # Fallback to bounding box if Anny returns zero
        current_height_cm = float(vertices_cm[:, 2].max() - vertices_cm[:, 2].min())
    if current_height_cm <= 0:
        raise HTTPException(status_code=500, detail="Mesh has zero height.")

    # Uniform scale to match provided height
    scale_factor = req.height_cm / current_height_cm
    vertices_cm = vertices_cm * scale_factor

    # Recompute anthropometry on scaled mesh
    vertices_m = (vertices_cm / 100.0).astype(np.float32)
    anny_anthro = engine.get_anthropometry(vertices_m)

    # Recompute all measurements
    measurements = extract_measurements(vertices_cm, faces, anny_anthropometry=anny_anthro)

    # Update stored result
    new_result = {
        "mesh_vertices": vertices_cm.tolist(),
        "faces": result["faces"],
        "measurements": measurements,
        "phenotypes": result["phenotypes"],
        "calibration": {
            "method_used": "height_cm",
            "confidence": 0.9,
            "warning": None,
        },
    }
    jobs[req.job_id]["result"] = new_result

    return {"result": new_result}


async def _job_worker():
    """Background worker that processes jobs from the queue."""
    while True:
        job_data = await job_queue.get()
        job_id = job_data["job_id"]
        try:
            jobs[job_id]["status"] = JobStatus.processing
            result = await asyncio.get_event_loop().run_in_executor(
                None, _process_job, job_data, job_id
            )
            jobs[job_id]["status"] = JobStatus.done
            jobs[job_id]["result"] = result
        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            jobs[job_id]["status"] = JobStatus.error
            jobs[job_id]["error"] = str(e)
        finally:
            try:
                os.unlink(job_data["video_path"])
            except OSError:
                pass
            job_queue.task_done()


def _update_progress(job_id: str, msg: str):
    if job_id in jobs:
        jobs[job_id]["progress"] = msg


def _process_job(job_data: dict, job_id: str) -> dict:
    """Synchronous job processing pipeline."""
    video_path = job_data["video_path"]
    reference_mode = job_data["reference_mode"]
    height_cm = job_data["height_cm"]

    # Step 1: Extract keyframes
    _update_progress(job_id, "Extracting frames...")
    frames = _extract_keyframes(video_path)
    if not frames:
        raise ValueError("Could not extract any frames from the video.")
    logger.info(f"Extracted {len(frames)} keyframes")

    # Step 2: Scale calibration
    if reference_mode in ("a4", "credit_card"):
        _update_progress(job_id, "Loading object detection model...")
    else:
        _update_progress(job_id, "Detecting scale...")
    cal_result: CalibrationResult = calibrate_from_frames(
        frames, reference_mode, height_cm=height_cm,
    )
    logger.info(f"Calibration done: method={cal_result.method_used}, confidence={cal_result.confidence}")
    _update_progress(job_id, "Detecting scale...")

    # Step 3: Run inference on the best frame
    _update_progress(job_id, "Detecting pose...")
    best_frame = _select_best_frame(frames)
    logger.info(f"Selected best frame, running MediaPipe + Anny...")
    inf_result: InferenceResult = engine.run(best_frame)
    logger.info(f"Inference done: {inf_result.vertices.shape[0]} vertices")
    _update_progress(job_id, "Recovering body mesh...")

    # Step 4: Get Anny's built-in anthropometry (in meters)
    _update_progress(job_id, "Applying scale calibration...")
    anny_anthro = engine.get_anthropometry(inf_result.vertices)

    # Step 5: Scale mesh to real-world dimensions.
    # Two scaling paths:
    #   A) Reference object (ArUco/A4/credit_card): use pixels_per_cm from
    #      calibration + person's pixel height from MediaPipe to compute
    #      real height in cm, then scale mesh to match.
    #   B) Height input: user provides height directly.
    # Both can be combined — height input always takes priority if provided.
    vertices = inf_result.vertices.copy()
    derived_height_cm = None

    # Path A: derive real height from reference object + person pixel height
    if cal_result.pixels_per_cm > 0 and reference_mode in ("aruco", "a4", "credit_card"):
        person_height_cm = inf_result.person_height_px / cal_result.pixels_per_cm
        if 50 < person_height_cm < 250:
            derived_height_cm = person_height_cm
            logger.info(
                f"Derived person height from {reference_mode}: "
                f"{inf_result.person_height_px:.0f}px / {cal_result.pixels_per_cm:.1f}px/cm "
                f"= {person_height_cm:.1f} cm"
            )

    # Path B: user-provided height (takes priority over derived)
    effective_height_cm = None
    if height_cm is not None and height_cm > 0:
        effective_height_cm = height_cm
        method_note = "user_height"
    elif derived_height_cm is not None:
        effective_height_cm = derived_height_cm
        method_note = "ref_object"

    if effective_height_cm is not None:
        mesh_height_m = anny_anthro["height_m"]
        if mesh_height_m > 0:
            scale_factor = (effective_height_cm / 100.0) / mesh_height_m
            vertices = vertices * scale_factor
            # Re-compute anthropometry with scaled vertices
            anny_anthro = engine.get_anthropometry(vertices)
            logger.info(f"Mesh scaled by {scale_factor:.3f}x ({method_note}: {effective_height_cm:.1f} cm)")

            # Boost confidence when both height and reference object are used
            if height_cm is not None and height_cm > 0 and reference_mode != "height_cm":
                cal_result = CalibrationResult(
                    pixels_per_cm=cal_result.pixels_per_cm,
                    confidence=min(0.95, cal_result.confidence + 0.2),
                    method_used=f"{cal_result.method_used}+height",
                    warning=None,
                )

    # Convert to cm for frontend display
    vertices_cm = vertices * 100.0

    # Step 6: Extract measurements
    _update_progress(job_id, "Computing measurements...")
    measurements = extract_measurements(
        vertices_cm, inf_result.faces, anny_anthropometry=anny_anthro
    )

    return {
        "mesh_vertices": vertices_cm.tolist(),
        "faces": inf_result.faces.tolist(),
        "measurements": measurements,
        "phenotypes": inf_result.phenotypes,
        "calibration": {
            "method_used": cal_result.method_used,
            "confidence": round(cal_result.confidence, 2),
            "warning": cal_result.warning,
        },
    }


def _extract_keyframes(video_path: str, every_n: int = 10, max_frames: int = 30) -> list:
    """Extract keyframes from video: every N frames, up to max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def _select_best_frame(frames: list) -> np.ndarray:
    """Select the sharpest frame (highest Laplacian variance)."""
    best_score = -1
    best_frame = frames[0]
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_frame = frame
    return best_frame


# ---------- Serve frontend static files (must be after all API routes) ----------
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Catch-all: serve .html, .js, .css etc. from frontend/
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
