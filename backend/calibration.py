"""
Scale calibration from reference objects.

Supported methods:
  - ArUco marker detection (cv2.aruco, Apache 2.0)
  - A4/credit card detection: contour-based with white/light color
    filtering, perspective correction, and aspect ratio matching.
    RT-DETR is used as a secondary signal when available.
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


def _calibrate_known_object(
    frames: list,
    object_type: str,
) -> CalibrationResult:
    """Detect a known object (A4 paper or credit card) using contour analysis.

    Primary method: find white/light rectangular contours with the expected
    aspect ratio. This is more reliable than RT-DETR for paper/card detection
    because RT-DETR has no "paper" class and detects unrelated objects.

    The contour method:
      1. Converts to grayscale, thresholds for light regions (paper is white/light)
      2. Finds rectangular contours with 4 corners
      3. Uses perspective-corrected dimensions (not bounding box) for accuracy
      4. Matches aspect ratio tightly against A4 (1.414) or credit card (1.585)
      5. Filters by size relative to frame
    """
    obj = KNOWN_OBJECTS[object_type]
    expected_ratio = max(obj["width_cm"], obj["height_cm"]) / min(
        obj["width_cm"], obj["height_cm"]
    )

    detections = _detect_with_contours(frames, obj, expected_ratio)

    if not detections:
        logger.warning(f"Contour detection found no {obj['label']}. Trying RT-DETR fallback.")
        model, processor = _get_rtdetr()
        if model is not None:
            detections = _detect_with_rtdetr(frames, obj, expected_ratio, model, processor)

    if not detections:
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="contour",
            warning=f"Could not detect {obj['label']}. Ensure it's clearly visible, flat, and well-lit.",
        )

    median_ppc = float(np.median(detections))

    # Compute confidence from consistency of detections
    if len(detections) >= 3:
        std = float(np.std(detections))
        cv_coeff = std / median_ppc if median_ppc > 0 else 1.0
        confidence = min(0.75, max(0.3, 0.75 - cv_coeff * 3))
    else:
        confidence = min(0.5, len(detections) * 0.15)

    logger.info(
        f"Calibration: {len(detections)} detections, "
        f"median {median_ppc:.2f} px/cm, confidence {confidence:.2f}"
    )

    return CalibrationResult(
        pixels_per_cm=median_ppc,
        confidence=round(confidence, 2),
        method_used="contour",
        warning=None if confidence >= 0.4 else
            f"Low confidence detecting {obj['label']}. Consider using an ArUco marker.",
    )


def _detect_with_contours(
    frames: list,
    obj: dict,
    expected_ratio: float,
) -> list:
    """Detect rectangular light-colored object via contour analysis.

    Tuned for white/light paper and cards against typical backgrounds.
    Uses perspective-corrected side lengths for accurate pixel/cm conversion.
    """
    detections = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        frame_area = frame_h * frame_w

        # Threshold for white/light regions (paper is typically bright)
        # Use adaptive threshold to handle varying lighting
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Also try adaptive threshold for uneven lighting
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -10
        )

        for mask in [bright_mask, adaptive_mask]:
            # Clean up: close small gaps, remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

                if len(approx) != 4:
                    continue

                # Filter by area: paper should be a reasonable fraction of frame
                area = cv2.contourArea(approx)
                area_fraction = area / frame_area
                if area_fraction < 0.005 or area_fraction > 0.25:
                    continue

                # Check convexity (paper should be convex)
                if not cv2.isContourConvex(approx):
                    continue

                # Compute side lengths using the 4 corner points
                # (perspective-corrected, not axis-aligned bounding box)
                pts = approx.reshape(4, 2).astype(np.float64)
                side_lengths = []
                for i in range(4):
                    side_lengths.append(np.linalg.norm(pts[(i+1) % 4] - pts[i]))

                # Group into two pairs of opposite sides
                s0, s1, s2, s3 = side_lengths
                pair_a = (s0 + s2) / 2  # sides 0 and 2 are opposite
                pair_b = (s1 + s3) / 2  # sides 1 and 3 are opposite

                # Check that opposite sides are roughly equal (not a trapezoid)
                if pair_a > 0 and pair_b > 0:
                    opp_ratio_a = abs(s0 - s2) / max(s0, s2)
                    opp_ratio_b = abs(s1 - s3) / max(s1, s3)
                    if opp_ratio_a > 0.25 or opp_ratio_b > 0.25:
                        continue

                longest_px = max(pair_a, pair_b)
                shortest_px = min(pair_a, pair_b)

                if shortest_px < 20:
                    continue

                aspect = longest_px / shortest_px

                # Tight aspect ratio check: A4 = 1.414, credit card = 1.585
                # Allow ±0.15 tolerance (accounts for slight perspective)
                if abs(aspect - expected_ratio) > 0.15:
                    continue

                # Check that region inside the contour is predominantly light
                rect_mask = np.zeros_like(gray)
                cv2.drawContours(rect_mask, [approx], -1, 255, -1)
                mean_brightness = cv2.mean(gray, mask=rect_mask)[0]
                if mean_brightness < 140:
                    continue

                # Compute pixels_per_cm using both dimensions for accuracy
                # Longest side corresponds to longest real dimension
                longest_cm = max(obj["width_cm"], obj["height_cm"])
                shortest_cm = min(obj["width_cm"], obj["height_cm"])

                ppc_long = longest_px / longest_cm
                ppc_short = shortest_px / shortest_cm
                ppc = (ppc_long + ppc_short) / 2  # average both dimensions

                logger.debug(
                    f"Contour detection: {longest_px:.0f}x{shortest_px:.0f}px, "
                    f"ratio={aspect:.3f} (expected {expected_ratio:.3f}), "
                    f"brightness={mean_brightness:.0f}, ppc={ppc:.2f}"
                )
                detections.append(float(ppc))

    return detections


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
        logger.warning(f"Could not load RT-DETR: {e}. Using contour detection only.")
        return None, None


def _detect_with_rtdetr(
    frames: list,
    obj: dict,
    expected_ratio: float,
    model,
    processor,
) -> list:
    """Fallback: detect known object using HuggingFace RT-DETR (Apache 2.0).

    Only uses COCO "book" class (index 73) as the closest match to paper/card.
    Much tighter filtering than before to avoid false positives.
    """
    import torch
    from PIL import Image

    # COCO class IDs that could be paper-like objects
    # 73 = "book" — closest to paper in COCO
    paper_like_classes = {73}

    # Cap to 10 evenly-spaced frames
    max_sample = 10
    if len(frames) > max_sample:
        indices = [int(i * (len(frames) - 1) / (max_sample - 1)) for i in range(max_sample)]
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames
    logger.info(f"RT-DETR fallback: processing {len(sampled_frames)} frames")

    detections = []
    for idx, frame in enumerate(sampled_frames):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.4
            )[0]

            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                label_id = int(label.item())

                # Only consider paper-like COCO classes
                if label_id not in paper_like_classes:
                    continue

                box = box.cpu().numpy()
                w = abs(box[2] - box[0])
                h = abs(box[3] - box[1])
                if w < 30 or h < 30:
                    continue

                ratio = max(w, h) / min(w, h)
                # Tighter ratio check: ±0.2
                if abs(ratio - expected_ratio) > 0.2:
                    continue

                # Size sanity: paper should be < 30% of frame
                frame_h, frame_w = frame.shape[:2]
                if (w * h) / (frame_w * frame_h) > 0.3:
                    continue

                longest_px = max(w, h)
                shortest_px = min(w, h)
                longest_cm = max(obj["width_cm"], obj["height_cm"])
                shortest_cm = min(obj["width_cm"], obj["height_cm"])

                ppc = ((longest_px / longest_cm) + (shortest_px / shortest_cm)) / 2
                detections.append(float(ppc))
                logger.debug(
                    f"RT-DETR: class={label_id}, score={score:.2f}, "
                    f"{w:.0f}x{h:.0f}px, ratio={ratio:.2f}, ppc={ppc:.2f}"
                )
        except Exception as e:
            logger.warning(f"RT-DETR inference failed on frame: {e}")
            continue

    return detections


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
