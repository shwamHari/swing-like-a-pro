"""
constants.py

Holds all static configuration for the swing-analysis pipeline:
- Directory paths
- MediaPipe initialization
- Event definitions
- Pose-drawing parameters
- Layout constants
- Per-event angle checks
"""

import os
import cv2
import mediapipe as mp

# ------------------------------------------------------------------------------
# Directory paths for output frames and final composites.
# ------------------------------------------------------------------------------
USER_EVENT_DIR = 'user_event_frames'
REF_EVENT_DIR  = 'reference_event_frames'
COMBINED_DIR   = 'combined_images'
for d in (USER_EVENT_DIR, REF_EVENT_DIR, COMBINED_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------------------
# MediaPipe Pose setup: single-image, min confidence 0.5
# ------------------------------------------------------------------------------
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

# ------------------------------------------------------------------------------
# Swing event ordering and name lookup
# ------------------------------------------------------------------------------
EVENT_ORDER = [
    'Address',
    'Toe-up',
    'Mid-backswing (arm parallel)',
    'Top',
    'Mid-downswing (arm parallel)',
    'Impact',
    'Mid-follow-through (shaft parallel)',
    'Finish'
]
EVENT_NAMES = {i: name for i, name in enumerate(EVENT_ORDER)}

# ------------------------------------------------------------------------------
# Keypoints & connections for rendering the skeleton
# ------------------------------------------------------------------------------
RELEVANT_KP        = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
CUSTOM_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 31), (28, 32),  # Ankles to feet
    (11, 23), (12, 24)   # Torso
]

# ------------------------------------------------------------------------------
# Layout constants for composite canvases and text rendering
# ------------------------------------------------------------------------------
TITLE_HEIGHT         = 60   # px reserved at top for title bars
GAP                  = 50   # px between images in comparisons
TEXT_AREA_WIDTH      = 400  # px for text columns
FEEDBACK_LINE_HEIGHT = 30   # px per feedback line
FONT                 = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE           = 0.7
FONT_THICKNESS       = 1
FEEDBACK_COL_WIDTH   = 600  # px width of feedback columns
COLUMN_GAP           = 50   # px between simple/detail columns

# ------------------------------------------------------------------------------
# Which angles to evaluate at each event phase
# ------------------------------------------------------------------------------
EVENT_FEEDBACK_ANGLES = {
    'Address':                          ['spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle'],
    'Toe-up':                           ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Mid-backswing (arm parallel)':    ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Top':                              ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle'],
    'Mid-downswing (arm parallel)':    ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Impact':                           ['spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle'],
    'Mid-follow-through (shaft parallel)': ['spine_angle', 'left_knee_bend', 'right_knee_bend'],
    'Finish':                           ['shoulder_tilt', 'spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle']
}
