"""
Scale calibration from reference objects.

Supported methods:
  - ArUco marker detection (cv2.aruco, Apache 2.0)
  - A4/credit card detection: uses MediaPipe pose to find hand positions,
    then searches for white rectangular contours near the hands.
    The paper/card is assumed to be held in the person's hand.
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

# MediaPipe pose singleton for hand detection during calibration
_mp_pose = None


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


def _get_hand_regions(frame: np.ndarray) -> list:
    """Detect hand/wrist positions using MediaPipe Pose.

    Returns list of (cx, cy, radius) tuples — circular regions around
    each detected wrist where we expect the held paper to be.
    The search radius is proportional to person height in the frame.
    """
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
        )

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _mp_pose.process(rgb)

    if not result.pose_landmarks:
        return []

    lm = result.pose_landmarks.landmark

    # Estimate person height in pixels for radius sizing
    nose_y = lm[0].y * h
    mid_shoulder_y = (lm[11].y + lm[12].y) / 2 * h
    head_top_y = nose_y - abs(nose_y - mid_shoulder_y) * 0.6
    feet_y = max(lm[27].y, lm[28].y, lm[31].y, lm[32].y) * h
    person_height_px = abs(feet_y - head_top_y)

    # Search radius: ~25% of person height (paper held in hand extends this far)
    radius = max(person_height_px * 0.25, 100)

    regions = []
    # MediaPipe wrist landmarks: 15 = left wrist, 16 = right wrist
    # Also check index finger tips (19, 20) for when wrist is occluded
    for idx in [15, 16, 19, 20]:
        vis = lm[idx].visibility
        if vis > 0.3:
            cx = int(lm[idx].x * w)
            cy = int(lm[idx].y * h)
            regions.append((cx, cy, int(radius)))

    return regions


def _is_near_hand(contour_pts: np.ndarray, hand_regions: list) -> bool:
    """Check if a contour's center is near any detected hand region."""
    if not hand_regions:
        return True  # No pose detected — don't filter (fallback)

    M = cv2.moments(contour_pts)
    if M["m00"] == 0:
        return False
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    for hx, hy, radius in hand_regions:
        dist = np.sqrt((cx - hx) ** 2 + (cy - hy) ** 2)
        if dist < radius:
            return True

    return False


def _calibrate_known_object(
    frames: list,
    object_type: str,
) -> CalibrationResult:
    """Detect A4 paper or credit card held in the person's hand.

    Pipeline:
      1. MediaPipe Pose detects wrist/hand positions in each frame
      2. Search for white rectangular contours NEAR the hands only
      3. Match aspect ratio (A4=1.414, credit card=1.585) with tight tolerance
      4. Use perspective-corrected side lengths for accurate px/cm
    """
    obj = KNOWN_OBJECTS[object_type]
    expected_ratio = max(obj["width_cm"], obj["height_cm"]) / min(
        obj["width_cm"], obj["height_cm"]
    )

    detections = _detect_paper_near_hands(frames, obj, expected_ratio)

    if not detections:
        return CalibrationResult(
            pixels_per_cm=0.0,
            confidence=0.0,
            method_used="contour+pose",
            warning=f"Could not detect {obj['label']} in hand. "
                    f"Hold the paper flat and visible near your hand.",
        )

    median_ppc = float(np.median(detections))

    # Confidence from consistency
    if len(detections) >= 3:
        std = float(np.std(detections))
        cv_coeff = std / median_ppc if median_ppc > 0 else 1.0
        confidence = min(0.8, max(0.3, 0.8 - cv_coeff * 3))
    else:
        confidence = min(0.5, len(detections) * 0.2)

    logger.info(
        f"Calibration: {len(detections)} detections, "
        f"median {median_ppc:.2f} px/cm, confidence {confidence:.2f}"
    )

    return CalibrationResult(
        pixels_per_cm=median_ppc,
        confidence=round(confidence, 2),
        method_used="contour+pose",
        warning=None if confidence >= 0.4 else
            f"Low confidence detecting {obj['label']}. Consider using an ArUco marker.",
    )


def _detect_paper_near_hands(
    frames: list,
    obj: dict,
    expected_ratio: float,
) -> list:
    """Detect white rectangular object near detected hand positions.

    Uses MediaPipe Pose to find wrist locations, then searches for
    paper contours only within a radius around each hand.
    """
    detections = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        frame_area = frame_h * frame_w

        # Find hand positions with MediaPipe
        hand_regions = _get_hand_regions(frame)
        if not hand_regions:
            logger.debug("No hands detected in frame — skipping")
            continue

        # Threshold for white/light regions (paper is white)
        _, bright_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        # Also try adaptive threshold for uneven lighting
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -10
        )

        for mask in [bright_mask, adaptive_mask]:
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

                # Must be near a detected hand
                if not _is_near_hand(approx, hand_regions):
                    continue

                # Filter by area
                area = cv2.contourArea(approx)
                area_fraction = area / frame_area
                if area_fraction < 0.003 or area_fraction > 0.2:
                    continue

                if not cv2.isContourConvex(approx):
                    continue

                # Perspective-corrected side lengths
                pts = approx.reshape(4, 2).astype(np.float64)
                side_lengths = []
                for i in range(4):
                    side_lengths.append(np.linalg.norm(pts[(i+1) % 4] - pts[i]))

                s0, s1, s2, s3 = side_lengths
                pair_a = (s0 + s2) / 2
                pair_b = (s1 + s3) / 2

                # Opposite sides should be roughly equal
                if pair_a > 0 and pair_b > 0:
                    opp_ratio_a = abs(s0 - s2) / max(s0, s2)
                    opp_ratio_b = abs(s1 - s3) / max(s1, s3)
                    if opp_ratio_a > 0.3 or opp_ratio_b > 0.3:
                        continue

                longest_px = max(pair_a, pair_b)
                shortest_px = min(pair_a, pair_b)

                if shortest_px < 20:
                    continue

                aspect = longest_px / shortest_px

                # Tight aspect ratio: A4 = 1.414 ± 0.2
                if abs(aspect - expected_ratio) > 0.2:
                    continue

                # Check brightness inside contour (paper is white/light)
                rect_mask = np.zeros_like(gray)
                cv2.drawContours(rect_mask, [approx], -1, 255, -1)
                mean_brightness = cv2.mean(gray, mask=rect_mask)[0]
                if mean_brightness < 130:
                    continue

                # Compute pixels_per_cm from both dimensions
                longest_cm = max(obj["width_cm"], obj["height_cm"])
                shortest_cm = min(obj["width_cm"], obj["height_cm"])

                ppc_long = longest_px / longest_cm
                ppc_short = shortest_px / shortest_cm
                ppc = (ppc_long + ppc_short) / 2

                logger.info(
                    f"Paper detected near hand: {longest_px:.0f}x{shortest_px:.0f}px, "
                    f"ratio={aspect:.3f}, brightness={mean_brightness:.0f}, "
                    f"ppc={ppc:.2f}"
                )
                detections.append(float(ppc))

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

    return CalibrationResult(
        pixels_per_cm=1.0,
        confidence=0.7,
        method_used="height_cm",
        warning="Height-based calibration is less accurate than marker-based methods.",
    )
