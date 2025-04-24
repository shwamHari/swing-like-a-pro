"""
feedback.py

Generates comparative feedback between user and reference swings:
1. Per-event biomechanical feedback
2. Summary of key differences and improvement areas
"""

from collections import defaultdict
import numpy as np
import cv2
from constants import (
    EVENT_FEEDBACK_ANGLES, EVENT_ORDER,
    TITLE_HEIGHT, FONT, FONT_SCALE, FONT_THICKNESS
)

def generate_feedback(user_angles, ref_angles, event):
    """
    Analyzes angle differences to generate actionable feedback.

    Args:
        user_angles: Biomechanical angles from user's swing
        ref_angles: Reference angles from ideal swing
        event: Current swing phase being analyzed

    Returns:
        tuple: (
            simple_feedback: Concise improvement suggestions,
            detailed_feedback: Technical descriptions of issues,
            angle_differences: Quantitative angle discrepancies,
            raw_feedbacks: Complete feedback data
        )
    """
    feedbacks = []  # List to store all feedback tuples: (simple_msg, detailed_msg, angle_difference)

    # Define biomechanical metrics being analyzed
    angle_names = ['shoulder_tilt', 'hip_rotation', 'left_elbow_angle', 'right_elbow_angle',
                   'spine_angle', 'left_knee_bend', 'right_knee_bend']

    # Create a dictionary mapping angle names to (user_angle, ref_angle) tuples
    angles_dict = dict(zip(angle_names, zip(user_angles, ref_angles)))

    # Retrieve the relevant angles to check for this event
    angles_to_check = EVENT_FEEDBACK_ANGLES.get(event, [])

    # Set thresholds for flagging a discrepancy
    general_threshold = 10  # Most angles
    top_elbow_threshold = 3  # More sensitive threshold for elbow at Top

    # SHOULDER TILT ANALYSIS
    if 'shoulder_tilt' in angles_to_check:
        user_angle, ref_angle = angles_dict['shoulder_tilt']
        diff = user_angle - ref_angle

        # Invert direction check
        if (ref_angle < 0 and user_angle > 0) or (ref_angle == 0 and user_angle > 0):
            feedbacks.append(("tilt shoulders downward", "Shoulders are tilting upward, should be downward", abs(diff)))
        elif (ref_angle > 0 and user_angle < 0) or (ref_angle == 0 and user_angle < 0):
            feedbacks.append(("tilt shoulders upward", "Shoulders are tilting downward, should be upward", abs(diff)))
        elif abs(diff) > general_threshold:
            # Provide feedback based on severity
            if diff > 0:
                feedbacks.append(("keep shoulders more level", f"Shoulder tilt is {abs(diff):.1f} deg more than reference", abs(diff)))
            else:
                feedbacks.append(("increase shoulder tilt, rotate upper body more", f"Shoulder tilt is {abs(diff):.1f} deg less than reference", abs(diff)))

    # HIP ROTATION ANALYSIS
    if 'hip_rotation' in angles_to_check:
        user_angle, ref_angle = angles_dict['hip_rotation']
        diff = user_angle - ref_angle

        # Directionality correction
        if (ref_angle < 0 and user_angle > 0) or (ref_angle == 0 and user_angle > 0):
            feedbacks.append(("rotate hips away from target", "Hips are rotating toward target, should be away", abs(diff)))
        elif (ref_angle > 0 and user_angle < 0) or (ref_angle == 0 and user_angle < 0):
            feedbacks.append(("rotate hips toward target", "Hips are rotating away from target, should be toward", abs(diff)))
        elif abs(diff) > general_threshold:
            # Severity-based feedback
            if diff > 0:
                feedbacks.append(("reduce hip rotation", f"Hip rotation is {abs(diff):.1f} deg more than reference", abs(diff)))
            else:
                feedbacks.append(("rotate hips more", f"Hip rotation is {abs(diff):.1f} deg less than reference", abs(diff)))

    # SPINE ANGLE ANALYSIS
    if 'spine_angle' in angles_to_check:
        user_angle, ref_angle = angles_dict['spine_angle']
        diff = user_angle - ref_angle
        if abs(diff) > general_threshold:
            if diff > 0:
                feedbacks.append(("straighten back", f"Spine angle is {abs(diff):.1f} deg more than reference", abs(diff)))
            else:
                feedbacks.append(("bend back more", f"Spine angle is {abs(diff):.1f} deg less than reference", abs(diff)))

    # KNEE BEND ANALYSIS
    if 'left_knee_bend' in angles_to_check and 'right_knee_bend' in angles_to_check:
        user_left, ref_left = angles_dict['left_knee_bend']
        user_right, ref_right = angles_dict['right_knee_bend']
        diff_left = user_left - ref_left
        diff_right = user_right - ref_right

        left_action, right_action = None, None
        left_diff_msg, right_diff_msg = None, None

        # Evaluate left knee
        if abs(diff_left) > general_threshold:
            left_action = "bend more" if user_left > ref_left else "bend less"
            left_diff_msg = f"{abs(diff_left):.1f} deg {'more' if user_left > ref_left else 'less'}"

        # Evaluate right knee
        if abs(diff_right) > general_threshold:
            right_action = "bend more" if user_right > ref_right else "bend less"
            right_diff_msg = f"{abs(diff_right):.1f} deg {'more' if user_right > ref_right else 'less'}"

        # Aggregate feedback logically
        if left_action and right_action:
            if left_action == right_action:
                avg_diff = (abs(diff_left) + abs(diff_right)) / 2
                feedbacks.append((f"{left_action} knees", f"Knee bend is {avg_diff:.1f} deg {left_action.split()[1]} than reference", avg_diff))
            else:
                feedbacks.append((f"{left_action} left knee", f"Left knee bend is {left_diff_msg} than reference", abs(diff_left)))
                feedbacks.append((f"{right_action} right knee", f"Right knee bend is {right_diff_msg} than reference", abs(diff_right)))
        elif left_action:
            feedbacks.append((f"{left_action} left knee", f"Left knee bend is {left_diff_msg} than reference", abs(diff_left)))
        elif right_action:
            feedbacks.append((f"{right_action} right knee", f"Right knee bend is {right_diff_msg} than reference", abs(diff_right)))

    # ELBOW ANGLE ANALYSIS (all phases except Top)
    if event != 'Top' and 'left_elbow_angle' in angles_to_check and 'right_elbow_angle' in angles_to_check:
        user_left, ref_left = angles_dict['left_elbow_angle']
        user_right, ref_right = angles_dict['right_elbow_angle']
        diff_left = user_left - ref_left
        diff_right = user_right - ref_right

        left_action, right_action = None, None
        left_diff_msg, right_diff_msg = None, None

        if abs(diff_left) > general_threshold:
            left_action = "bend less" if user_left > ref_left else "bend more"
            left_diff_msg = f"{abs(diff_left):.1f} deg {'more' if user_left > ref_left else 'less'}"

        if abs(diff_right) > general_threshold:
            right_action = "bend less" if user_right > ref_right else "bend more"
            right_diff_msg = f"{abs(diff_right):.1f} deg {'more' if user_right > ref_right else 'less'}"

        if left_action and right_action:
            if left_action == right_action:
                avg_diff = (abs(diff_left) + abs(diff_right)) / 2
                feedbacks.append((f"{left_action} elbows", f"Elbow angle is {avg_diff:.1f} deg {left_action.split()[1]} than reference", avg_diff))
            else:
                feedbacks.append((f"{left_action} left elbow", f"Left elbow angle is {left_diff_msg} than reference", abs(diff_left)))
                feedbacks.append((f"{right_action} right elbow", f"Right elbow angle is {right_diff_msg} than reference", abs(diff_right)))
        elif left_action:
            feedbacks.append((f"{left_action} left elbow", f"Left elbow angle is {left_diff_msg} than reference", abs(diff_left)))
        elif right_action:
            feedbacks.append((f"{right_action} right elbow", f"Right elbow angle is {right_diff_msg} than reference", abs(diff_right)))

    # SPECIAL CASE: 'Top' position should have a straighter left arm
    if event == 'Top' and 'left_elbow_angle' in angles_to_check:
        user_left, ref_left = angles_dict['left_elbow_angle']
        diff_left = user_left - ref_left
        if user_left < ref_left and abs(diff_left) > top_elbow_threshold:
            feedbacks.append(("straighten left arm at top", f"Left elbow angle is {abs(diff_left):.1f} deg less than reference", abs(diff_left)))

    # If no feedback generated, assume good alignment
    if not feedbacks:
        feedbacks.append(("Swing well-aligned", "Your swing is well-aligned with the reference for this event", 0))

    # Separate and return feedback content
    simple_feedback = [fb[0] for fb in feedbacks]
    detailed_feedback = [fb[1] for fb in feedbacks]
    angle_differences = [fb[2] for fb in feedbacks]

    return simple_feedback, detailed_feedback, angle_differences, feedbacks



def generate_feedback_summary(feedback_data):
    """
    Aggregates feedback across events to identify:
    - Common issues in backswing/downswing phases
    - Top 3 largest angle discrepancies

    Args:
        feedback_data: Accumulated feedback from all analyzed events

    Returns:
        np.ndarray: Visual summary image with key findings
    """
    # Define event groupings for backswing and downswing phases
    backswing_events = ['Address', 'Toe-up', 'Mid-backswing (arm parallel)', 'Top']
    downswing_events = ['Mid-downswing (arm parallel)', 'Impact', 'Mid-follow-through (shaft parallel)']
    excluded_events = ['Finish', 'Mid-follow-through (shaft parallel)']

    # Backswing Trends (at least 2/4 events)
    backswing_feedback = defaultdict(list)
    for event in backswing_events:
        if event in feedback_data:
            for simple in feedback_data[event]['simple']:
                if simple != "Swing well-aligned":
                    backswing_feedback[simple].append(event)
    backswing_trends = {fb: events for fb, events in backswing_feedback.items() if len(events) >= 2}

    # Downswing Trends (at least 2/3 events)
    downswing_feedback = defaultdict(list)
    for event in downswing_events:
        if event in feedback_data:
            for simple in feedback_data[event]['simple']:
                if simple != "Swing well-aligned":
                    downswing_feedback[simple].append(event)
    downswing_trends = {fb: events for fb, events in downswing_feedback.items() if len(events) >= 2}

    # Top 3 Angle Differences
    angle_diff_entries = []
    for event in feedback_data:
        if event in excluded_events:
            continue
        for simple, detailed, diff in feedback_data[event]['feedbacks']:
            if simple != "Swing well-aligned" and diff > 0:
                angle_diff_entries.append((diff, event, simple, detailed))
    # Sort by angle difference and take top 3
    top_3_diffs = sorted(angle_diff_entries, key=lambda x: x[0], reverse=True)[:3]

    # Calculate canvas height
    SECTION_GAP = 50
    LINE_SPACING = 30
    backswing_lines = len(backswing_trends) + 2  # Header + items
    downswing_lines = len(downswing_trends) + 2
    top_3_lines = len(top_3_diffs) + 2
    canvas_h = TITLE_HEIGHT + SECTION_GAP * 3 + (backswing_lines + downswing_lines + top_3_lines) * LINE_SPACING
    canvas_w = 1600  # Increased width to accommodate longer text
    summary = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Title
    cv2.putText(summary, "Feedback Summary", (canvas_w // 2 - 100, TITLE_HEIGHT - 20), FONT, 1.2, (255, 255, 255), 2)

    # Backswing Feedback Trends
    y_pos = TITLE_HEIGHT + SECTION_GAP
    cv2.putText(summary, "Backswing Feedback Trends", (50, y_pos), FONT, 1.0, (255, 255, 255), 2)
    y_pos += LINE_SPACING
    if backswing_trends:
        for feedback, events in backswing_trends.items():
            text = f"{feedback}: {', '.join(events)}"
            cv2.putText(summary, text, (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            y_pos += LINE_SPACING
    else:
        cv2.putText(summary, "No common trends identified", (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        y_pos += LINE_SPACING

    # Downswing Feedback Trends
    y_pos += SECTION_GAP
    cv2.putText(summary, "Downswing Feedback Trends", (50, y_pos), FONT, 1.0, (255, 255, 255), 2)
    y_pos += LINE_SPACING
    if downswing_trends:
        for feedback, events in downswing_trends.items():
            text = f"{feedback}: {', '.join(events)}"
            cv2.putText(summary, text, (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            y_pos += LINE_SPACING
    else:
        cv2.putText(summary, "No common trends identified", (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        y_pos += LINE_SPACING

    # Top 3 Angle Differences
    y_pos += SECTION_GAP
    cv2.putText(summary, "Top 3 Angle Differences", (50, y_pos), FONT, 1.0, (255, 255, 255), 2)
    y_pos += LINE_SPACING
    if top_3_diffs:
        for diff, event, simple, detailed in top_3_diffs:
            text = f"In {event}, {simple}, {detailed}"
            cv2.putText(summary, text, (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            y_pos += LINE_SPACING
    else:
        cv2.putText(summary, "No significant differences identified", (50, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    return summary
