"""
Measurement extraction from Anny mesh vertices.

Uses Anny's built-in Anthropometry for height, mass, waist, BMI,
plus geometric cross-sectional analysis for additional measurements
(chest, hip, inseam, shoulder width, sleeve length).

Anny outputs vertices in METERS. Z-axis is up (height).
Mesh is in T-pose: arms extend horizontally at shoulder height.

Z-fractions calibrated for Anny's T-pose mesh topology:
  - Arms appear in cross-sections starting at ~Z=0.52 (full_x jumps)
  - Waist narrowest at Z≈0.62 (torso only)
  - Chest at Z≈0.70 (just below where arms attach)
  - Hip at Z≈0.48 (below arm attachment, torso only)

NO SMPL, NO SMPL-X topology assumptions. Works with Anny native mesh topology.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Z-axis proportions calibrated for Anny T-pose mesh.
# In T-pose, arms extend at ~Z=0.52+, so circumference slices must
# either avoid arm Z-levels or filter to torso-only vertices.
LANDMARK_Z_FRACTIONS = {
    "chest": 0.68,      # Nipple/bust line — widest chest circumference
    "waist": 0.62,      # Narrowest torso point in T-pose
    "hip": 0.48,        # Below arm level — safe from arm contamination
    "crotch": 0.45,     # Crotch/inseam point
    "shoulder": 0.82,   # Shoulder level
    "wrist": 0.55,      # T-pose wrist level (arms horizontal)
}


def extract_measurements(
    vertices: np.ndarray,
    faces: np.ndarray,
    anny_anthropometry: Optional[dict] = None,
) -> dict:
    """Extract body measurements from Anny mesh.

    Args:
        vertices: (N, 3) array. If scale_cm is applied, units are cm.
                  If raw from Anny, units are meters.
        faces: (F, 3) array of face indices.
        anny_anthropometry: Optional dict from InferenceEngine.get_anthropometry()
                           with height_m, mass_kg, waist_m, bmi.

    Returns:
        Dict mapping measurement name to {value_cm, value_in}.
    """
    if vertices.shape[0] == 0:
        raise ValueError("Empty vertex array — cannot compute measurements.")

    # Anny uses Z as up-axis
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    height_raw = float(z_max - z_min)

    if height_raw <= 0:
        raise ValueError("Mesh has zero height — cannot compute measurements.")

    # Determine if we're in meters or cm
    if height_raw < 10:
        scale = 100.0
    else:
        scale = 1.0

    verts_cm = vertices * scale
    z_min_cm = z_min * scale
    bbox_height_cm = height_raw * scale

    # Use Anny's anthropometric height (crown-to-sole) when available.
    if anny_anthropometry and "height_m" in anny_anthropometry:
        anny_height_cm = anny_anthropometry["height_m"] * 100
        if anny_height_cm > 0 and abs(anny_height_cm - bbox_height_cm) < bbox_height_cm * 0.15:
            height_cm = anny_height_cm
        else:
            height_cm = bbox_height_cm
    else:
        height_cm = bbox_height_cm

    # Compute torso X-range for filtering arm vertices from circumference slices.
    # In T-pose, the torso is centered around X=0 and much narrower than full span.
    x_center = float(np.median(verts_cm[:, 0]))
    torso_half_width = height_cm * 0.12  # ~20cm for a 170cm person
    # Chest needs wider filter — the ribcage extends wider than the waist
    chest_half_width = height_cm * 0.16  # ~27cm for a 170cm person

    measurements = {}

    # Height
    measurements["height"] = _fmt(height_cm)

    # Chest circumference (torso-only to exclude arms)
    # Scan a range of Z-levels near the chest to find the maximum circumference
    # (the "chest" measurement is defined as the fullest point of the torso)
    chest_z_nominal = z_min_cm + LANDMARK_Z_FRACTIONS["chest"] * height_cm
    best_chest = 0.0
    for z_offset_frac in [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]:
        test_z = chest_z_nominal + z_offset_frac * height_cm
        circ = _compute_circumference(verts_cm, test_z, height_cm,
                                      tolerance=height_cm * 0.015,
                                      torso_filter=(x_center, chest_half_width))
        if circ is not None and circ > best_chest:
            best_chest = circ
    measurements["chest"] = _fmt(best_chest)

    # Waist circumference — prefer Anny's built-in (topology-based)
    if anny_anthropometry and "waist_m" in anny_anthropometry:
        waist_cm = anny_anthropometry["waist_m"] * 100
        measurements["waist"] = _fmt(waist_cm)
    else:
        waist_z = z_min_cm + LANDMARK_Z_FRACTIONS["waist"] * height_cm
        circ = _compute_circumference(verts_cm, waist_z, height_cm,
                                      torso_filter=(x_center, torso_half_width))
        measurements["waist"] = _fmt(circ if circ else 0.0)

    # Hip circumference (Z=0.48 is below arm attachment, safe)
    hip_z = z_min_cm + LANDMARK_Z_FRACTIONS["hip"] * height_cm
    circ = _compute_circumference(verts_cm, hip_z, height_cm)
    measurements["hip"] = _fmt(circ if circ else 0.0)

    # Inseam
    crotch_z = z_min_cm + LANDMARK_Z_FRACTIONS["crotch"] * height_cm
    inseam = crotch_z - z_min_cm
    measurements["inseam"] = _fmt(inseam)

    # Shoulder width (full X span at shoulder height — includes arm start)
    shoulder_z = z_min_cm + LANDMARK_Z_FRACTIONS["shoulder"] * height_cm
    shoulder_width = _compute_width_at_height(verts_cm, shoulder_z, height_cm * 0.02)
    measurements["shoulder_width"] = _fmt(shoulder_width)

    # Sleeve length (T-pose: shoulder to wrist along arm)
    sleeve = _compute_sleeve_length(verts_cm, height_cm, z_min_cm)
    measurements["sleeve_length"] = _fmt(sleeve)

    return measurements


def _fmt(value_cm: float) -> dict:
    return {
        "value_cm": round(value_cm, 1),
        "value_in": round(value_cm / 2.54, 1),
    }


def _compute_circumference(
    vertices: np.ndarray,
    z_level: float,
    height_cm: float,
    tolerance: float = 0.5,
    torso_filter: Optional[tuple] = None,
) -> Optional[float]:
    """Compute circumference at a given Z level using angular perimeter.

    Uses angular ordering around the centroid to trace the actual body
    surface contour, including concavities. This is more accurate than
    ConvexHull, which overestimates by skipping concave curves.

    Args:
        torso_filter: Optional (x_center, half_width) to exclude arm vertices.
    """
    mask = np.abs(vertices[:, 2] - z_level) < tolerance
    section_verts = vertices[mask]

    if len(section_verts) < 3:
        tolerance = height_cm * 0.02
        mask = np.abs(vertices[:, 2] - z_level) < tolerance
        section_verts = vertices[mask]
        if len(section_verts) < 3:
            return None

    # Filter to torso-only if requested (excludes T-pose arm vertices)
    if torso_filter is not None:
        x_center, half_width = torso_filter
        torso_mask = np.abs(section_verts[:, 0] - x_center) < half_width
        torso_verts = section_verts[torso_mask]
        if len(torso_verts) >= 3:
            section_verts = torso_verts

    # Project to XY plane (Anny: X=left/right, Y=front/back, Z=up)
    points_2d = section_verts[:, [0, 1]]

    return _angular_perimeter(points_2d)


def _angular_perimeter(points_2d: np.ndarray) -> Optional[float]:
    """Compute perimeter by tracing points in angular order around centroid.

    This follows the actual body surface contour including concavities,
    unlike ConvexHull which would skip inward curves and overestimate.
    Uses radial binning to get a clean outer boundary per angular sector.
    """
    if len(points_2d) < 3:
        return None

    centroid = points_2d.mean(axis=0)
    dx = points_2d[:, 0] - centroid[0]
    dy = points_2d[:, 1] - centroid[1]
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx ** 2 + dy ** 2)

    # Bin by angle sector and take the outermost point per bin.
    # This smooths out interior vertices while preserving concavities.
    n_bins = min(72, max(24, len(points_2d) // 3))
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    boundary_points = []

    for i in range(n_bins):
        bin_mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if not bin_mask.any():
            continue
        # Take the point farthest from centroid in this angular bin
        bin_radii = radii[bin_mask]
        bin_pts = points_2d[bin_mask]
        best_idx = bin_radii.argmax()
        boundary_points.append(bin_pts[best_idx])

    if len(boundary_points) < 3:
        return None

    boundary = np.array(boundary_points)

    # Sort boundary points by angle for correct perimeter tracing
    bp_dx = boundary[:, 0] - centroid[0]
    bp_dy = boundary[:, 1] - centroid[1]
    bp_angles = np.arctan2(bp_dy, bp_dx)
    order = np.argsort(bp_angles)
    boundary = boundary[order]

    # Compute perimeter
    perimeter = 0.0
    n = len(boundary)
    for i in range(n):
        p1 = boundary[i]
        p2 = boundary[(i + 1) % n]
        perimeter += np.linalg.norm(p2 - p1)

    return float(perimeter)


def _compute_width_at_height(
    vertices: np.ndarray,
    z_level: float,
    tolerance: float = 1.0,
) -> float:
    mask = np.abs(vertices[:, 2] - z_level) < tolerance
    section_verts = vertices[mask]
    if len(section_verts) < 2:
        return 0.0
    return float(section_verts[:, 0].max() - section_verts[:, 0].min())


def _compute_sleeve_length(
    vertices: np.ndarray,
    height_cm: float,
    z_min: float,
) -> float:
    """Estimate sleeve length (shoulder to wrist) in T-pose.

    In T-pose, the arm extends horizontally. We find the outermost
    shoulder point and the outermost wrist point on the same side,
    then compute Euclidean distance.
    """
    shoulder_z = z_min + LANDMARK_Z_FRACTIONS["shoulder"] * height_cm
    wrist_z = z_min + LANDMARK_Z_FRACTIONS["wrist"] * height_cm

    shoulder_tol = 0.02 * height_cm
    shoulder_mask = np.abs(vertices[:, 2] - shoulder_z) < shoulder_tol
    shoulder_verts = vertices[shoulder_mask]

    if len(shoulder_verts) < 2:
        return 0.33 * height_cm

    x_center = float(np.median(vertices[:, 0]))
    right_shoulder_verts = shoulder_verts[shoulder_verts[:, 0] > x_center]
    if len(right_shoulder_verts) == 0:
        return 0.33 * height_cm
    shoulder_point = right_shoulder_verts[right_shoulder_verts[:, 0].argmax()]

    wrist_tol = 0.03 * height_cm
    wrist_mask = (np.abs(vertices[:, 2] - wrist_z) < wrist_tol) & (
        vertices[:, 0] > x_center
    )
    wrist_verts = vertices[wrist_mask]
    if len(wrist_verts) == 0:
        return 0.33 * height_cm
    wrist_point = wrist_verts[wrist_verts[:, 0].argmax()]

    return float(np.linalg.norm(shoulder_point - wrist_point))
