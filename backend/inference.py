"""
Body mesh inference using Anny (Apache 2.0 + CC0) and MediaPipe Pose (Apache 2.0).

Pipeline:
  1. MediaPipe detects 2D/3D pose landmarks from a video frame
  2. Body proportions estimated from landmark distances
  3. Anny generates a parametric body mesh from estimated phenotype parameters
  4. Mesh returned in T-pose for measurement clarity

NOT using SMPL, SMPL-X, or PromptHMR (licensing restrictions).
NOT using the Anny smplx topology variant (non-commercial only).
NOT using Multi-HMR (requires SMPL-X weights).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    vertices: np.ndarray   # (N, 3) mesh vertices in meters
    faces: np.ndarray      # (F, 3) triangular face indices
    phenotypes: dict       # Anny phenotype parameters used
    person_height_px: float  # Person's pixel height in the frame (head to feet)
    foot_length_px: float    # Foot length in pixels (heel to toe) for shoe-size calibration


class InferenceEngine:
    """Generates Anny body mesh from video frame via MediaPipe pose estimation."""

    def __init__(self):
        import anny
        import mediapipe as mp

        # Load Anny body model — native topology only (Apache 2.0 + CC0)
        self.anny_model = anny.create_fullbody_model()
        self.anthropometry = anny.Anthropometry(self.anny_model)
        logger.info(
            f"Anny body model loaded: {self.anny_model.template_vertices.shape[0]} vertices, "
            f"{self.anny_model.faces.shape[0]} faces (native topology)."
        )

        # Triangular faces for Three.js (Anny uses quad faces natively)
        # Filter out eyeball and inner-mouth geometry (driven by facial bones)
        self.tri_faces = self._get_body_faces()
        logger.info(f"Body-only mesh: {self.tri_faces.shape[0]} tri faces (facial geometry removed).")

        # MediaPipe Pose (Apache 2.0)
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        logger.info("MediaPipe Pose loaded.")

    def _get_body_faces(self) -> np.ndarray:
        """Get triangular faces excluding eyeball and tongue geometry.

        Anny's fullbody mesh includes separate eyeball spheres and tongue
        interior driven by specific bones. These look wrong when rendered,
        so we exclude faces weighted to these bones.

        We keep the jaw bone (104) because it controls chin, jaw, and neck
        geometry that is visible and important for the body silhouette.
        Only eyeballs (143, 148) and tongue bones (112-122) are removed.
        """
        # Eyeball bones: eye.L=143, eye.R=148
        # Tongue bones: 112-122
        # NOTE: jaw=104 is intentionally KEPT — it drives visible chin/neck geometry
        facial_bone_ids = {112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 143, 148}

        vbi = self.anny_model.vertex_bone_indices  # (V, 8)
        facial_vert_mask = torch.zeros(vbi.shape[0], dtype=torch.bool)
        for k in range(vbi.shape[1]):
            for bid in facial_bone_ids:
                facial_vert_mask |= (vbi[:, k] == bid)

        facial_vert_set = set(facial_vert_mask.nonzero(as_tuple=True)[0].tolist())

        all_tri = self.anny_model.get_triangular_faces().cpu().numpy().astype(np.int32)
        body_mask = np.array([
            not any(int(v) in facial_vert_set for v in all_tri[i])
            for i in range(all_tri.shape[0])
        ])
        return all_tri[body_mask]

    def run(self, frame: np.ndarray, gender: str = None) -> InferenceResult:
        """Run inference on a single BGR frame.

        Steps:
          1. Detect pose landmarks via MediaPipe
          2. Estimate phenotype parameters from body proportions
          3. Generate Anny mesh in T-pose

        Args:
            frame: BGR image frame
            gender: Optional explicit gender ("male" or "female").
                    If provided, overrides auto-detection from pose.
        """
        # Detect pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose_detector.process(rgb)

        if not pose_result.pose_landmarks:
            raise ValueError("No person detected in frame. Ensure a person is clearly visible.")

        landmarks = pose_result.pose_landmarks.landmark

        # Estimate phenotypes from pose landmarks
        phenotypes = self._estimate_phenotypes(landmarks, frame.shape, gender=gender)

        # Measure person's pixel height (head-top to feet) for reference-object calibration.
        # MediaPipe doesn't have a "top of head" landmark, so we estimate it:
        # head top ≈ nose position - 0.6 × (nose-to-mid-shoulder distance) upward
        person_height_px = self._measure_person_height_px(landmarks, frame.shape)

        # Measure foot length in pixels for shoe-size calibration
        foot_length_px = self._measure_foot_length_px(landmarks, frame.shape)

        # Generate Anny mesh with estimated phenotypes (T-pose for measurements)
        with torch.no_grad():
            result = self.anny_model.forward(
                phenotype_kwargs=phenotypes,
            )

        # rest_vertices are in T-pose, which is what we want for measurements
        vertices = result["rest_vertices"][0].cpu().numpy().astype(np.float32)

        return InferenceResult(
            vertices=vertices,
            faces=self.tri_faces,
            phenotypes={k: round(float(v), 3) for k, v in phenotypes.items()},
            person_height_px=person_height_px,
            foot_length_px=foot_length_px,
        )

    def _measure_person_height_px(self, landmarks, frame_shape: tuple) -> float:
        """Measure person's full height in pixels from head-top to feet.

        MediaPipe doesn't have a "top of head" landmark, so we estimate:
          head_top ≈ nose_y - 0.6 × |nose_y - mid_shoulder_y|
        Then take the lowest of L_ANKLE and R_ANKLE for feet.
        """
        h, w = frame_shape[:2]

        def lm_y(idx):
            return landmarks[idx].y * h

        nose_y = lm_y(0)
        mid_shoulder_y = (lm_y(11) + lm_y(12)) / 2
        head_offset = abs(nose_y - mid_shoulder_y) * 0.6
        head_top_y = nose_y - head_offset

        # Lowest foot point (MediaPipe uses y-down)
        feet_y = max(lm_y(27), lm_y(28), lm_y(31), lm_y(32))

        person_height = abs(feet_y - head_top_y)
        logger.info(f"Person pixel height: {person_height:.0f}px (frame {w}x{h})")
        return float(person_height)

    def _measure_foot_length_px(self, landmarks, frame_shape: tuple) -> float:
        """Measure foot length in pixels from heel to toe using MediaPipe landmarks.

        MediaPipe foot landmarks:
          27 = left ankle,  28 = right ankle
          29 = left heel,   30 = right heel
          31 = left foot index (toe), 32 = right foot index (toe)

        Foot length = distance from heel to toe tip.
        We measure both feet and take the average.
        """
        h, w = frame_shape[:2]

        def lm_px(idx):
            l = landmarks[idx]
            return np.array([l.x * w, l.y * h])

        foot_lengths = []

        # Left foot: heel (29) to toe (31)
        try:
            left_heel = lm_px(29)
            left_toe = lm_px(31)
            left_vis = min(landmarks[29].visibility, landmarks[31].visibility)
            if left_vis > 0.3:
                left_len = float(np.linalg.norm(left_toe - left_heel))
                if left_len > 5:  # minimum plausible pixel length
                    foot_lengths.append(left_len)
        except (IndexError, AttributeError):
            pass

        # Right foot: heel (30) to toe (32)
        try:
            right_heel = lm_px(30)
            right_toe = lm_px(32)
            right_vis = min(landmarks[30].visibility, landmarks[32].visibility)
            if right_vis > 0.3:
                right_len = float(np.linalg.norm(right_toe - right_heel))
                if right_len > 5:
                    foot_lengths.append(right_len)
        except (IndexError, AttributeError):
            pass

        if foot_lengths:
            avg = sum(foot_lengths) / len(foot_lengths)
            logger.info(f"Foot length: {avg:.0f}px (from {len(foot_lengths)} feet)")
            return avg
        else:
            logger.info("Could not measure foot length — landmarks not visible")
            return 0.0

    def _estimate_phenotypes(self, landmarks, frame_shape: tuple, gender: str = None) -> dict:
        """Estimate Anny phenotype parameters from MediaPipe pose landmarks.

        MediaPipe provides 33 landmarks with (x, y, z) in normalized coords.
        We use body proportions to estimate gender, height, weight, etc.
        All Anny phenotype params are in [0, 1] range.

        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: (height, width, channels) of the frame
            gender: Optional explicit gender ("male" or "female").
                    Anny: 1.0 = male, 0.0 = female.
        """
        h, w = frame_shape[:2]

        def lm(idx):
            """Get landmark as pixel coordinates."""
            l = landmarks[idx]
            return np.array([l.x * w, l.y * h, l.z * w])

        def dist(a, b):
            return float(np.linalg.norm(lm(a) - lm(b)))

        # Key MediaPipe landmark indices
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_ANKLE, R_ANKLE = 27, 28
        L_WRIST, R_WRIST = 15, 16
        L_ELBOW, R_ELBOW = 13, 14
        NOSE = 0

        # Compute body proportions
        shoulder_width = dist(L_SHOULDER, R_SHOULDER)
        hip_width = dist(L_HIP, R_HIP)
        torso_length = (dist(L_SHOULDER, L_HIP) + dist(R_SHOULDER, R_HIP)) / 2
        leg_length = (dist(L_HIP, L_ANKLE) + dist(R_HIP, R_ANKLE)) / 2
        arm_length = (
            dist(L_SHOULDER, L_ELBOW) + dist(L_ELBOW, L_WRIST) +
            dist(R_SHOULDER, R_ELBOW) + dist(R_ELBOW, R_WRIST)
        ) / 2

        # Full body height estimate (head top to ankle)
        body_height_px = dist(NOSE, L_ANKLE)

        # ----- Gender -----
        # Use explicit gender selection if provided, otherwise auto-detect.
        # Anny: 1.0 = male, 0.0 = female
        if gender is not None and gender in ("male", "female"):
            gender_val = 1.0 if gender == "male" else 0.0
            logger.info(f"Using explicit gender: {gender} ({gender_val})")
        else:
            # Auto-detect from shoulder-to-hip ratio: male ~1.4–1.6, female ~1.0–1.2
            if hip_width > 0:
                sh_ratio = shoulder_width / hip_width
            else:
                sh_ratio = 1.0
            gender_val = float(np.clip((sh_ratio - 1.05) / 0.55, 0.0, 1.0))
            logger.info(f"Auto-detected gender: {gender_val:.2f}")

        # ----- Weight / build -----
        # Use multiple cues: shoulder/height ratio + hip/height ratio + torso width
        if body_height_px > 0:
            shoulder_ratio = shoulder_width / body_height_px
            hip_ratio = hip_width / body_height_px
            # Average of shoulder and hip width relative to height
            width_ratio = (shoulder_ratio + hip_ratio) / 2
        else:
            width_ratio = 0.2

        # Wider relative to height → heavier build
        # Typical range: thin ~0.15, average ~0.20, heavy ~0.28+
        weight = np.clip((width_ratio - 0.14) / 0.16, 0.05, 0.95)

        # ----- Muscle -----
        # Shoulder-dominated builds → more muscle
        # Also factor in arm thickness relative to body
        if body_height_px > 0 and hip_width > 0:
            shoulder_dominance = shoulder_ratio / hip_ratio if hip_ratio > 0 else 1.0
            muscle = np.clip((shoulder_dominance - 0.9) / 0.6, 0.0, 1.0)
            # Blend with weight — muscular people are also wider
            muscle = np.clip(muscle * 0.6 + weight * 0.4, 0.0, 1.0)
        else:
            muscle = 0.5

        # ----- Height -----
        # Default to 0.5 — actual scaling is done by calibration in main.py
        height = 0.5

        # ----- Proportions -----
        # Leg-to-torso ratio determines body proportions
        # Short legs/long torso → low proportions, long legs → high proportions
        if torso_length > 0:
            leg_torso_ratio = leg_length / torso_length
        else:
            leg_torso_ratio = 1.5
        # Typical range: 1.3 (short legs) to 1.8 (long legs)
        proportions = np.clip((leg_torso_ratio - 1.2) / 0.6, 0.1, 0.9)

        # ----- Age -----
        # Default to 0.5 (young adult). Could be refined with face analysis.
        age = 0.5

        return {
            "gender": float(gender_val),
            "age": float(age),
            "muscle": float(muscle),
            "weight": float(weight),
            "height": float(height),
            "proportions": float(proportions),
        }

    def get_anthropometry(self, vertices: np.ndarray) -> dict:
        """Compute anthropometric measurements from Anny mesh using built-in methods.

        Args:
            vertices: (N, 3) array in meters (as output by Anny).

        Returns:
            Dict with height_m, mass_kg, waist_m, bmi.
        """
        verts_tensor = torch.from_numpy(vertices).unsqueeze(0).double()
        return {
            "height_m": float(self.anthropometry.height(verts_tensor)[0]),
            "mass_kg": float(self.anthropometry.mass(verts_tensor)[0]),
            "waist_m": float(self.anthropometry.waist_circumference(verts_tensor)[0]),
            "bmi": float(self.anthropometry.bmi(verts_tensor)[0]),
        }
