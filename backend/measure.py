"""
Measurement extraction from Anny mesh vertices.

Uses Anny's built-in Anthropometry for height, mass, waist, BMI,
plus geometric cross-sectional analysis for additional measurements
(chest, hip, inseam, shoulder width, sleeve length).

Anny outputs vertices in METERS. Z-axis is up (height).
Mesh is in T-pose: arms extend horizontally at shoulder height.

Accuracy approach:
  - Mesh-edge interpolation: finds exact intersection of mesh edges
    with horizontal Z-planes, giving precise cross-section contours
    instead of relying on nearby vertex positions.
  - Z-scanning: scans multiple Z-levels to find the widest (chest)
    or narrowest (waist) circumference in the region.
  - Torso filtering: excludes T-pose arm vertices from chest/waist
    cross-sections using X-range filtering.

Z-fractions calibrated for Anny's T-pose mesh topology:
  - Arms appear in cross-sections starting at ~Z=0.52 (full_x jumps)
  - Waist narrowest at Z≈0.62 (torso only)
  - Chest at Z≈0.68 (nipple/bust line)
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
    faces_int = faces.astype(np.int32)
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
    # Analysis shows arms enter at Z≥0.53 with X ±54cm. With 12% filter (±20cm),
    # arm roots don't contaminate until Z≥0.73. So 12% is safe for chest (Z=0.64-0.68).
    x_center = float(np.median(verts_cm[:, 0]))
    torso_half_width = height_cm * 0.12  # ~20cm for a 170cm person

    # Pre-compute edge list from faces for mesh-edge interpolation
    edges = _get_unique_edges(faces_int)

    measurements = {}

    # Height
    measurements["height"] = _fmt(height_cm)

    # Chest circumference (torso-only to exclude arms)
    # Scan Z-levels near the chest to find the widest torso circumference.
    # Scan range: Z-frac 0.64 to 0.68 (safe from arm contamination with 12% filter).
    # Analysis shows 12% filter is clean up to Z=0.72, so this range is safe.
    chest_z_nominal = z_min_cm + LANDMARK_Z_FRACTIONS["chest"] * height_cm
    best_chest = 0.0
    scan_step = height_cm * 0.005  # 0.5% of height ≈ 0.85cm for 170cm person
    for offset_i in range(-8, 1):  # scan below nominal only (0.64 to 0.68)
        test_z = chest_z_nominal + offset_i * scan_step
        circ = _compute_circumference_edge(
            verts_cm, edges, test_z,
            torso_filter=(x_center, torso_half_width),
        )
        if circ is not None and circ > best_chest:
            best_chest = circ
    # Fallback to vertex-based if edge method returned nothing
    if best_chest < 1.0:
        for offset_i in range(-8, 1):
            test_z = chest_z_nominal + offset_i * scan_step
            circ = _compute_circumference_vertex(
                verts_cm, test_z, height_cm,
                torso_filter=(x_center, torso_half_width),
            )
            if circ is not None and circ > best_chest:
                best_chest = circ
    measurements["chest"] = _fmt(best_chest)

    # Waist circumference — prefer Anny's built-in (topology-based)
    if anny_anthropometry and "waist_m" in anny_anthropometry:
        waist_cm = anny_anthropometry["waist_m"] * 100
        measurements["waist"] = _fmt(waist_cm)
    else:
        # Scan for the narrowest circumference in the waist region
        waist_z_nominal = z_min_cm + LANDMARK_Z_FRACTIONS["waist"] * height_cm
        best_waist = float("inf")
        for offset_i in range(-4, 5):
            test_z = waist_z_nominal + offset_i * scan_step
            circ = _compute_circumference_edge(
                verts_cm, edges, test_z,
                torso_filter=(x_center, torso_half_width),
            )
            if circ is not None and 10 < circ < best_waist:
                best_waist = circ
        if best_waist == float("inf"):
            # Vertex fallback
            for offset_i in range(-4, 5):
                test_z = waist_z_nominal + offset_i * scan_step
                circ = _compute_circumference_vertex(
                    verts_cm, test_z, height_cm,
                    torso_filter=(x_center, torso_half_width),
                )
                if circ is not None and 10 < circ < best_waist:
                    best_waist = circ
        measurements["waist"] = _fmt(best_waist if best_waist < float("inf") else 0.0)

    # Hip circumference — scan for maximum (Z=0.48 is below arm attachment, safe)
    hip_z_nominal = z_min_cm + LANDMARK_Z_FRACTIONS["hip"] * height_cm
    best_hip = 0.0
    for offset_i in range(-6, 7):
        test_z = hip_z_nominal + offset_i * scan_step
        circ = _compute_circumference_edge(verts_cm, edges, test_z)
        if circ is not None and circ > best_hip:
            best_hip = circ
    if best_hip < 1.0:
        for offset_i in range(-6, 7):
            test_z = hip_z_nominal + offset_i * scan_step
            circ = _compute_circumference_vertex(verts_cm, test_z, height_cm)
            if circ is not None and circ > best_hip:
                best_hip = circ
    measurements["hip"] = _fmt(best_hip)

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


def _get_unique_edges(faces: np.ndarray) -> np.ndarray:
    """Extract unique edges from face array.

    Args:
        faces: (F, 3) array of triangle face indices.

    Returns:
        (E, 2) array of unique edge vertex pairs.
    """
    # Each triangle has 3 edges
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    # Sort each edge so (a,b) and (b,a) are the same
    edges = np.sort(edges, axis=1)
    # Remove duplicates
    edges = np.unique(edges, axis=0)
    return edges


def _compute_circumference_edge(
    vertices: np.ndarray,
    edges: np.ndarray,
    z_level: float,
    torso_filter: Optional[tuple] = None,
) -> Optional[float]:
    """Compute circumference by intersecting mesh edges with a horizontal Z-plane.

    This gives exact cross-section contour points by interpolating along edges
    that straddle the Z-level, rather than just using nearby vertices.
    Much more accurate than vertex-based approach.
    """
    z_vals = vertices[:, 2]

    # Find edges that cross the Z-level
    v0_z = z_vals[edges[:, 0]]
    v1_z = z_vals[edges[:, 1]]

    # Edge crosses if one vertex is above and one is below (or on) the Z-level
    crosses = (v0_z - z_level) * (v1_z - z_level) < 0
    crossing_edges = edges[crosses]

    if len(crossing_edges) < 3:
        return None

    # Interpolate to find exact intersection point on each crossing edge
    v0 = vertices[crossing_edges[:, 0]]  # (N, 3)
    v1 = vertices[crossing_edges[:, 1]]  # (N, 3)
    z0 = v0[:, 2]
    z1 = v1[:, 2]

    # Interpolation parameter: t such that z0 + t*(z1-z0) = z_level
    dz = z1 - z0
    # Avoid division by zero (shouldn't happen since we filtered for crossing)
    safe_dz = np.where(np.abs(dz) > 1e-10, dz, 1e-10)
    t = ((z_level - z0) / safe_dz).reshape(-1, 1)
    t = np.clip(t, 0, 1)

    # Interpolated intersection points
    intersections = v0 + t * (v1 - v0)  # (N, 3)

    # Apply torso filter if requested
    if torso_filter is not None:
        x_center, half_width = torso_filter
        torso_mask = np.abs(intersections[:, 0] - x_center) < half_width
        intersections = intersections[torso_mask]
        if len(intersections) < 3:
            return None

    # Project to XY plane (X=left/right, Y=front/back)
    points_2d = intersections[:, [0, 1]]

    return _angular_perimeter(points_2d)


def _compute_circumference_vertex(
    vertices: np.ndarray,
    z_level: float,
    height_cm: float,
    tolerance: float = 0.5,
    torso_filter: Optional[tuple] = None,
) -> Optional[float]:
    """Fallback: compute circumference using nearby vertices (less accurate).

    Used when edge-based method fails (e.g., mesh doesn't have face data).
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

    Uses radial binning with adaptive bin count to get a clean outer boundary.
    Takes the outermost point per angular sector, then traces the contour.
    """
    if len(points_2d) < 3:
        return None

    centroid = points_2d.mean(axis=0)
    dx = points_2d[:, 0] - centroid[0]
    dy = points_2d[:, 1] - centroid[1]
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx ** 2 + dy ** 2)

    # Use more bins for denser point clouds (from edge interpolation)
    # Minimum 36 bins (10-degree sectors), up to 120 bins (3-degree sectors)
    n_bins = min(120, max(36, len(points_2d) // 2))
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

    # Compute perimeter as sum of distances between adjacent boundary points
    diffs = np.diff(boundary, axis=0, append=boundary[:1])
    # Wrap: last point to first point
    diffs[-1] = boundary[0] - boundary[-1]
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    perimeter = float(segment_lengths.sum())

    return perimeter


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
