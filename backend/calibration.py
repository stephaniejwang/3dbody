"""
Scale calibration from reference objects.

Supported methods:
  - ArUco marker detection (cv2.aruco, Apache 2.0)
  - Known object detection via HuggingFace RT-DETR (Apache 2.0) — NOT YOLOv8 (AGPL)
  - Manual height input fallback

NO SMPL, NO SMPL-X, NO YOLOv8/Ultralytics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Known object dimensions in cm
KNOWN_OBJECTS = {
    "a4": {"width_cm": 21.0, "height_cm": 29.7, "label": "A4 paper"},
    "credit_card": {"width_cm": 8.56, "height_cm": 5.4, "label": "Credit card"},
}

# Default ArUco marker size in cm
DEFAULT_ARUCO_SIZE_CM = 10.0

# RT-DETR model singleton (loaded lazily)
_rtdetr_model = None
_rtdetr_processor = None


@dataclass
class CalibrationResult:
    pixels_per_cm: float
    confidence: float  # 0.0–1.0
    method_used: str
    warning: Optional[str] = None


def calibrate_from_frames(
    frames: list,
    reference_mode: str,
    height_cm: Optional[float] = None,
    aruco_size_cm: float = DEFAULT_ARUCO_SIZE_CM,
) -> CalibrationResult:
    """Run scale calibration on frames using the specified reference mode."""
    if reference_mode == "aruco":
        return _calibrate_aruco(frames, aruco_size_cm)
    elif reference_mode in ("a4", "credit_card"):
        return _calibrate_known_object(frames, reference_mode)
    elif reference_mode == "height_cm":
        return _calibrate_height(height_cm)
    else:
        raise ValueError(f"Unknown reference_mode: {reference_mode}")


def _calibrate_aruco(
    frames: list, marker_size_cm: float
) -> CalibrationResult:
    """Detect ArUco marker (DICT_4X4_50) and compute pixels_per_cm."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    detections = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            for marker_corners in corners:
                pts = marker_corners[0]  # shape (4, 2)
                perimeter_px = 0.0
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    perimeter_px += np.linalg.norm(p2 - p1)
                perimeter_cm = 4.0 * marker_size_cm
                ppc = perimeter_px / perimeter_cm
                detections.append(ppc)

    if not detections:
        logger.warning("No ArUco markers detected in any frame.")
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="aruco",
            warning="No ArUco marker detected. Try holding the marker more steadily in frame.",
        )

    median_ppc = float(np.median(detections))
    if len(detections) >= 5:
        std = float(np.std(detections))
        cv_coeff = std / median_ppc if median_ppc > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - cv_coeff * 5))
    else:
        confidence = min(0.6, len(detections) * 0.15)

    warning = None
    if confidence < 0.5:
        warning = "Low ArUco detection confidence. Ensure the marker is flat and well-lit."

    return CalibrationResult(
        pixels_per_cm=median_ppc,
        confidence=round(confidence, 2),
        method_used="aruco",
        warning=warning,
    )


def _get_rtdetr():
    """Lazily load HuggingFace RT-DETR model (Apache 2.0). NOT YOLOv8 (AGPL)."""
    global _rtdetr_model, _rtdetr_processor
    if _rtdetr_model is not None:
        return _rtdetr_model, _rtdetr_processor

    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        import torch

        logger.info("Loading RT-DETR from HuggingFace (PekingU/rtdetr_r50vd)...")
        _rtdetr_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        _rtdetr_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        _rtdetr_model.eval()
        logger.info("RT-DETR loaded successfully.")
        return _rtdetr_model, _rtdetr_processor
    except Exception as e:
        logger.warning(f"Could not load RT-DETR: {e}. Falling back to contour detection.")
        return None, None


def _calibrate_known_object(
    frames: list,
    object_type: str,
) -> CalibrationResult:
    """Detect a known object (A4 paper or credit card) using RT-DETR.

    Uses HuggingFace RT-DETR (Apache 2.0) for object detection.
    NOT YOLOv8 — YOLOv8/Ultralytics is AGPL licensed.
    """
    obj = KNOWN_OBJECTS[object_type]

    model, processor = _get_rtdetr()
    if model is not None:
        return _detect_with_rtdetr(frames, obj, model, processor)
    else:
        return _detect_with_contours(frames, obj, object_type)


def _detect_with_rtdetr(
    frames: list,
    obj: dict,
    model,
    processor,
) -> CalibrationResult:
    """Detect known object using HuggingFace RT-DETR (Apache 2.0)."""
    import torch
    from PIL import Image

    expected_ratio = max(obj["width_cm"], obj["height_cm"]) / min(
        obj["width_cm"], obj["height_cm"]
    )

    # Cap to 10 evenly-spaced frames for better confidence without processing all 30
    max_sample = 10
    if len(frames) > max_sample:
        indices = [int(i * (len(frames) - 1) / (max_sample - 1)) for i in range(max_sample)]
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames
    logger.info(f"RT-DETR: processing {len(sampled_frames)} of {len(frames)} frames")

    detections = []
    for idx, frame in enumerate(sampled_frames):
        logger.info(f"RT-DETR: frame {idx + 1}/{len(sampled_frames)}")
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process detections
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.3
            )[0]

            for box, score in zip(results["boxes"], results["scores"]):
                box = box.cpu().numpy()
                w = abs(box[2] - box[0])
                h = abs(box[3] - box[1])
                if w < 20 or h < 20:
                    continue

                ratio = max(w, h) / min(w, h)
                if abs(ratio - expected_ratio) < 0.5:
                    longest_px = max(w, h)
                    longest_cm = max(obj["width_cm"], obj["height_cm"])
                    ppc = longest_px / longest_cm
                    detections.append(float(ppc))
        except Exception as e:
            logger.warning(f"RT-DETR inference failed on frame: {e}")
            continue

    if not detections:
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="rtdetr",
            warning=f"Could not detect {obj['label']} in any frame. Try holding it more visibly.",
        )

    median_ppc = float(np.median(detections))
    confidence = min(0.75, len(detections) * 0.1)
    if len(detections) >= 3:
        std = float(np.std(detections))
        cv_coeff = std / median_ppc if median_ppc > 0 else 1.0
        confidence = min(0.75, max(0.2, 0.75 - cv_coeff * 3))

    return CalibrationResult(
        pixels_per_cm=median_ppc,
        confidence=round(confidence, 2),
        method_used="rtdetr",
        warning="Object detection is less accurate than ArUco markers."
        if confidence < 0.6
        else None,
    )


def _detect_with_contours(
    frames: list,
    obj: dict,
    object_type: str,
) -> CalibrationResult:
    """Fallback: detect rectangular object via contour analysis."""
    expected_ratio = max(obj["width_cm"], obj["height_cm"]) / min(
        obj["width_cm"], obj["height_cm"]
    )

    detections = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w < 30 or h < 30:
                    continue
                ratio = max(w, h) / min(w, h)
                if abs(ratio - expected_ratio) < 0.4:
                    area = cv2.contourArea(approx)
                    frame_area = frame.shape[0] * frame.shape[1]
                    if 0.005 < area / frame_area < 0.3:
                        longest_px = max(w, h)
                        longest_cm = max(obj["width_cm"], obj["height_cm"])
                        ppc = longest_px / longest_cm
                        detections.append(ppc)

    if not detections:
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="contour_fallback",
            warning=f"Could not detect {obj['label']}. Ensure it's clearly visible and well-lit.",
        )

    median_ppc = float(np.median(detections))
    confidence = min(0.5, len(detections) * 0.08)

    return CalibrationResult(
        pixels_per_cm=median_ppc,
        confidence=round(confidence, 2),
        method_used="contour_fallback",
        warning="Using contour fallback — low confidence. Consider using an ArUco marker.",
    )


def _calibrate_height(height_cm: Optional[float]) -> CalibrationResult:
    """Height-based calibration: scale is applied after mesh recovery in main.py."""
    if height_cm is None or height_cm <= 0:
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="height_cm",
            warning="Invalid height value.",
        )

    # Sentinel value — actual scaling happens in main.py using Anny's mesh height
    return CalibrationResult(
        pixels_per_cm=1.0,
        confidence=0.7,
        method_used="height_cm",
        warning="Height-based calibration is less accurate than marker-based methods.",
    )
