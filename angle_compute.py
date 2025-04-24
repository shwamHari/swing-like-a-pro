"""
angle_compute.py

Provides functions to compute biomechanical angles from pose landmarks:
- pairwise angles (shoulder tilt, hip rotation)
- three-point joint angles (elbow, knee)
- spine inclination
"""

import numpy as np

def calculate_angle(p1, p2):
    """
    Compute the angle (in degrees) of the vector from p1 to p2
    relative to the horizontal axis.
    Args:
        p1 (tuple): (x,y) of first point.
        p2 (tuple): (x,y) of second point.
    Returns:
        float: angle in degrees.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def compute_three_point_angle(a, b, c):
    """
    Compute joint angle at point b formed by points a-b-c.
    Args:
        a, b, c (tuple): (x,y) coordinates.
    Returns:
        float: interior angle in degrees.
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def compute_elbow_angle(shoulder, elbow, wrist, shape):
    """
    Compute the elbow flexion angle given landmarks.
    Args:
        shoulder, elbow, wrist: landmark objects with .x, .y normalized.
        shape (tuple): image shape (H, W).
    Returns:
        float: elbow angle in degrees.
    """
    p1 = (shoulder.x * shape[1], shoulder.y * shape[0])
    p2 = (elbow.x    * shape[1], elbow.y    * shape[0])
    p3 = (wrist.x    * shape[1], wrist.y    * shape[0])
    return compute_three_point_angle(p1, p2, p3)

def compute_spine_angle(landmarks, shape):
    """
    Compute the inclination of the spine by averaging shoulders & hips,
    then measuring angle to vertical.
    Args:
        landmarks (list): list of pose landmarks.
        shape (tuple): image shape (H, W).
    Returns:
        float: spine angle in degrees.
    """
    h, w = shape[0], shape[1]
    sl = ((landmarks[11].x + landmarks[12].x) / 2 * w,
          (landmarks[11].y + landmarks[12].y) / 2 * h)
    hl = ((landmarks[23].x + landmarks[24].x) / 2 * w,
          (landmarks[23].y + landmarks[24].y) / 2 * h)
    vertical_ref = (hl[0], hl[1] - 100)  # point directly above hip midpoint
    return compute_three_point_angle(sl, hl, vertical_ref)

def compute_knee_bend(landmarks, shape, side='left'):
    """
    Compute knee bend at left or right knee.
    Args:
        landmarks (list): list of pose landmarks.
        shape (tuple): image shape (H, W).
        side (str): 'left' or 'right'
    Returns:
        float: knee angle in degrees.
    """
    hip_idx, knee_idx, ank_idx = (23, 25, 27) if side == 'left' else (24, 26, 28)
    h, w = shape[0], shape[1]
    p1 = (landmarks[hip_idx].x * w, landmarks[hip_idx].y * h)
    p2 = (landmarks[knee_idx].x * w, landmarks[knee_idx].y * h)
    p3 = (landmarks[ank_idx].x * w, landmarks[ank_idx].y * h)
    return compute_three_point_angle(p1, p2, p3)

def calculate_angles(landmarks, shape):
    """
    Compute all relevant angles for a full-body pose.
    Returns:
        tuple: (
            shoulder_tilt,
            hip_rotation,
            left_elbow_angle,
            right_elbow_angle,
            spine_angle,
            left_knee_bend,
            right_knee_bend
        )
    """
    sl, sr = landmarks[11], landmarks[12]
    hl, hr = landmarks[23], landmarks[24]
    el, wl = landmarks[13], landmarks[15]
    er, wr = landmarks[14], landmarks[16]

    shoulder_tilt = calculate_angle(
        (sl.x * shape[1], sl.y * shape[0]),
        (sr.x * shape[1], sr.y * shape[0])
    )
    hip_rotation = calculate_angle(
        (hl.x * shape[1], hl.y * shape[0]),
        (hr.x * shape[1], hr.y * shape[0])
    )
    left_elbow  = compute_elbow_angle(sl, el, wl, shape)
    right_elbow = compute_elbow_angle(sr, er, wr, shape)
    spine_angle = compute_spine_angle(landmarks, shape)
    left_knee   = compute_knee_bend(landmarks, shape, 'left')
    right_knee  = compute_knee_bend(landmarks, shape, 'right')

    return (
        shoulder_tilt,
        hip_rotation,
        left_elbow,
        right_elbow,
        spine_angle,
        left_knee,
        right_knee
    )
