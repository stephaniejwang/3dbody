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
        """Get triangular faces excluding eyeball and inner-mouth geometry.

        Anny's fullbody mesh includes separate eyeball spheres and mouth
        interior driven by facial bones (eye.L, eye.R, jaw, tongue*).
        These look wrong when rendered, so we exclude any face that has
        a vertex weighted to these bones.
        """
        # Facial bone indices: eye.L=143, eye.R=148, jaw=104, tongue=112-122
        facial_bone_ids = {104, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 143, 148}

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

    def run(self, frame: np.ndarray) -> InferenceResult:
        """Run inference on a single BGR frame.

        Steps:
          1. Detect pose landmarks via MediaPipe
          2. Estimate phenotype parameters from body proportions
          3. Generate Anny mesh in T-pose
        """
        # Detect pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose_detector.process(rgb)

        if not pose_result.pose_landmarks:
            raise ValueError("No person detected in frame. Ensure a person is clearly visible.")

        landmarks = pose_result.pose_landmarks.landmark

        # Estimate phenotypes from pose landmarks
        phenotypes = self._estimate_phenotypes(landmarks, frame.shape)

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
        )

    def _estimate_phenotypes(self, landmarks, frame_shape: tuple) -> dict:
        """Estimate Anny phenotype parameters from MediaPipe pose landmarks.

        MediaPipe provides 33 landmarks with (x, y, z) in normalized coords.
        We use body proportions to estimate gender, height, weight, etc.
        All Anny phenotype params are in [0, 1] range.
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

        # Shoulder-to-hip ratio is an indicator of gender/build
        if hip_width > 0:
            sh_ratio = shoulder_width / hip_width
        else:
            sh_ratio = 1.0

        # Gender estimation: higher sh_ratio → more masculine
        # Typical male sh_ratio ~1.4–1.6, female ~1.1–1.3
        gender = np.clip((sh_ratio - 1.1) / 0.5, 0.0, 1.0)

        # Weight estimation from shoulder-width to body-height ratio
        # Wider relative to height → heavier build
        if body_height_px > 0:
            width_ratio = shoulder_width / body_height_px
        else:
            width_ratio = 0.25
        weight = np.clip((width_ratio - 0.18) / 0.14, 0.0, 1.0)

        # Muscle: correlated with weight but slightly offset
        muscle = np.clip(weight * 0.8, 0.0, 1.0)

        # Height: default to 0.5 (will be scaled by calibration in main.py)
        height = 0.5

        # Proportions: based on leg-to-torso ratio
        if torso_length > 0:
            leg_torso_ratio = leg_length / torso_length
        else:
            leg_torso_ratio = 1.5
        proportions = np.clip((leg_torso_ratio - 1.2) / 0.6, 0.0, 1.0)

        # Age: default to young adult (0.5 maps to ~25–35 years in Anny)
        age = 0.5

        return {
            "gender": float(gender),
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
