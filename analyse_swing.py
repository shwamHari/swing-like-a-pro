import argparse
import cv2
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import ToTensor, Normalize
from model import EventDetector
import torch.nn.functional as F
import mediapipe as mp
from collections import defaultdict

# Create directories for user and reference event frames
user_event_dir = 'user_event_frames'
reference_event_dir = 'reference_event_frames'
combined_dir = 'combined_images'
os.makedirs(user_event_dir, exist_ok=True)
os.makedirs(reference_event_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Event names with order
event_order = [
    'Address', 'Toe-up', 'Mid-backswing (arm parallel)', 'Top',
    'Mid-downswing (arm parallel)', 'Impact', 'Mid-follow-through (shaft parallel)', 'Finish'
]
event_names = {i: name for i, name in enumerate(event_order)}

# Relevant keypoints and connections for pose drawing
relevant_keypoints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
custom_connections = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 31), (28, 32),  # Ankles to feet
    (11, 23), (12, 24)   # Torso (spine approximation)
]

# Layout constants
TITLE_HEIGHT = 60
GAP = 50
TEXT_AREA_WIDTH = 400
FEEDBACK_LINE_HEIGHT = 30
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 1
FEEDBACK_COLUMN_WIDTH = 600  # Increased to provide more space
COLUMN_GAP = 50  # Added gap between Simple Feedback and Detail columns

# Define which angles to check for each event
event_feedback_angles = {
    'Address': ['spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle'],
    'Toe-up': ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Mid-backswing (arm parallel)': ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Top': ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle'],
    'Mid-downswing (arm parallel)': ['shoulder_tilt', 'spine_angle', 'hip_rotation', 'left_knee_bend', 'right_knee_bend'],
    'Impact': ['spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle'],
    'Mid-follow-through (shaft parallel)': ['spine_angle', 'left_knee_bend', 'right_knee_bend'],
    'Finish': ['shoulder_tilt', 'spine_angle', 'left_knee_bend', 'right_knee_bend', 'left_elbow_angle', 'right_elbow_angle']
}

class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret or img is None or img.size == 0:
                continue
            resized = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        if not images:
            raise ValueError(f"No valid frames in {self.path}.")

        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    rows, cols = np.where(thresh > 0)
    if len(rows) == 0 or len(cols) == 0:
        return img
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    return img[min_row:max_row+1, min_col:max_col+1]

def draw_pose(img, landmarks):
    for conn in custom_connections:
        start = landmarks.landmark[conn[0]]
        end = landmarks.landmark[conn[1]]
        pt1 = (int(start.x * img.shape[1]), int(start.y * img.shape[0]))
        pt2 = (int(end.x * img.shape[1]), int(end.y * img.shape[0]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 4)
    for idx in relevant_keypoints:
        kp = landmarks.landmark[idx]
        x = int(kp.x * img.shape[1])
        y = int(kp.y * img.shape[0])
        cv2.circle(img, (x, y), 7, (0, 0, 255), -1)

def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.degrees(np.arctan2(dy, dx))

def compute_three_point_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def compute_elbow_angle(shoulder, elbow, wrist, img_shape):
    p1 = (shoulder.x * img_shape[1], shoulder.y * img_shape[0])
    p2 = (elbow.x * img_shape[1], elbow.y * img_shape[0])
    p3 = (wrist.x * img_shape[1], wrist.y * img_shape[0])
    return compute_three_point_angle(p1, p2, p3)

def compute_spine_angle(landmarks, img_shape):
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    hip_left = landmarks[23]
    hip_right = landmarks[24]
    shoulder_mid = ((shoulder_left.x + shoulder_right.x) / 2 * img_shape[1],
                    (shoulder_left.y + shoulder_right.y) / 2 * img_shape[0])
    hip_mid = ((hip_left.x + hip_right.x) / 2 * img_shape[1],
               (hip_left.y + hip_right.y) / 2 * img_shape[0])
    vertical = (hip_mid[0], hip_mid[1] - 100)
    return compute_three_point_angle(shoulder_mid, hip_mid, vertical)

def compute_knee_bend(landmarks, img_shape, side='left'):
    hip_idx, knee_idx, ankle_idx = (23, 25, 27) if side == 'left' else (24, 26, 28)
    hip = landmarks[hip_idx]
    knee = landmarks[knee_idx]
    ankle = landmarks[ankle_idx]
    p1 = (hip.x * img_shape[1], hip.y * img_shape[0])
    p2 = (knee.x * img_shape[1], knee.y * img_shape[0])
    p3 = (ankle.x * img_shape[1], ankle.y * img_shape[0])
    return compute_three_point_angle(p1, p2, p3)

def calculate_angles(landmarks, img_shape):
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    hip_left = landmarks[23]
    hip_right = landmarks[24]
    elbow_left = landmarks[13]
    wrist_left = landmarks[15]
    elbow_right = landmarks[14]
    wrist_right = landmarks[16]

    shoulder_tilt = calculate_angle(
        (shoulder_left.x * img_shape[1], shoulder_left.y * img_shape[0]),
        (shoulder_right.x * img_shape[1], shoulder_right.y * img_shape[0])
    )
    hip_rotation = calculate_angle(
        (hip_left.x * img_shape[1], hip_left.y * img_shape[0]),
        (hip_right.x * img_shape[1], hip_right.y * img_shape[0])
    )
    left_elbow_angle = compute_elbow_angle(shoulder_left, elbow_left, wrist_left, img_shape)
    right_elbow_angle = compute_elbow_angle(shoulder_right, elbow_right, wrist_right, img_shape)
    spine_angle = compute_spine_angle(landmarks, img_shape)
    knee_bend_left = compute_knee_bend(landmarks, img_shape, side='left')
    knee_bend_right = compute_knee_bend(landmarks, img_shape, side='right')
    return shoulder_tilt, hip_rotation, left_elbow_angle, right_elbow_angle, spine_angle, knee_bend_left, knee_bend_right

def generate_feedback(user_angles, ref_angles, event):
    feedbacks = []  # List of tuples: (simple_feedback, detailed_feedback, angle_difference)
    angle_names = ['shoulder_tilt', 'hip_rotation', 'left_elbow_angle', 'right_elbow_angle', 'spine_angle', 'left_knee_bend', 'right_knee_bend']
    angles_dict = dict(zip(angle_names, zip(user_angles, ref_angles)))

    angles_to_check = event_feedback_angles.get(event, [])
    general_threshold = 10
    top_elbow_threshold = 3

    if 'shoulder_tilt' in angles_to_check:
        user_angle, ref_angle = angles_dict['shoulder_tilt']
        diff = user_angle - ref_angle
        if (ref_angle < 0 and user_angle > 0) or (ref_angle == 0 and user_angle > 0):
            feedbacks.append(("tilt shoulders downward", "Shoulders are tilting upward, should be downward", abs(diff)))
        elif (ref_angle > 0 and user_angle < 0) or (ref_angle == 0 and user_angle < 0):
            feedbacks.append(("tilt shoulders upward", "Shoulders are tilting downward, should be upward", abs(diff)))
        else:
            if abs(diff) > general_threshold:
                if diff > 0:
                    feedbacks.append(("keep shoulders more level", f"Shoulder tilt is {abs(diff):.1f} deg more than reference", abs(diff)))
                else:
                    feedbacks.append(("increase shoulder tilt, rotate upper body more", f"Shoulder tilt is {abs(diff):.1f} deg less than reference", abs(diff)))

    if 'hip_rotation' in angles_to_check:
        user_angle, ref_angle = angles_dict['hip_rotation']
        diff = user_angle - ref_angle
        if (ref_angle < 0 and user_angle > 0) or (ref_angle == 0 and user_angle > 0):
            feedbacks.append(("rotate hips away from target", "Hips are rotating toward target, should be away", abs(diff)))
        elif (ref_angle > 0 and user_angle < 0) or (ref_angle == 0 and user_angle < 0):
            feedbacks.append(("rotate hips toward target", "Hips are rotating away from target, should be toward", abs(diff)))
        else:
            if abs(diff) > general_threshold:
                if diff > 0:
                    feedbacks.append(("reduce hip rotation", f"Hip rotation is {abs(diff):.1f} deg more than reference", abs(diff)))
                else:
                    feedbacks.append(("rotate hips more", f"Hip rotation is {abs(diff):.1f} deg less than reference", abs(diff)))

    if 'spine_angle' in angles_to_check:
        user_angle, ref_angle = angles_dict['spine_angle']
        diff = user_angle - ref_angle
        if abs(diff) > general_threshold:
            if diff > 0:
                feedbacks.append(("straighten back", f"Spine angle is {abs(diff):.1f} deg more than reference", abs(diff)))
            else:
                feedbacks.append(("bend back more", f"Spine angle is {abs(diff):.1f} deg less than reference", abs(diff)))

    if 'left_knee_bend' in angles_to_check and 'right_knee_bend' in angles_to_check:
        user_left, ref_left = angles_dict['left_knee_bend']
        user_right, ref_right = angles_dict['right_knee_bend']
        diff_left = user_left - ref_left
        diff_right = user_right - ref_right

        left_action = None
        right_action = None
        left_diff_msg = None
        right_diff_msg = None

        if abs(diff_left) > general_threshold:
            if user_left > ref_left:
                left_action = "bend more"
                left_diff_msg = f"{abs(diff_left):.1f} deg more"
            else:
                left_action = "bend less"
                left_diff_msg = f"{abs(diff_left):.1f} deg less"

        if abs(diff_right) > general_threshold:
            if user_right > ref_right:
                right_action = "bend more"
                right_diff_msg = f"{abs(diff_right):.1f} deg more"
            else:
                right_action = "bend less"
                right_diff_msg = f"{abs(diff_right):.1f} deg less"

        if left_action and right_action:
            if left_action == right_action:
                avg_diff = (abs(diff_left) + abs(diff_right)) / 2
                feedbacks.append((f"{left_action} knees", f"Knee bend is {avg_diff:.1f} deg {'more' if 'more' in left_diff_msg else 'less'} than reference", avg_diff))
            else:
                feedbacks.append((f"{left_action} left knee", f"Left knee bend is {left_diff_msg} than reference", abs(diff_left)))
                feedbacks.append((f"{right_action} right knee", f"Right knee bend is {right_diff_msg} than reference", abs(diff_right)))
        elif left_action:
            feedbacks.append((f"{left_action} left knee", f"Left knee bend is {left_diff_msg} than reference", abs(diff_left)))
        elif right_action:
            feedbacks.append((f"{right_action} right knee", f"Right knee bend is {right_diff_msg} than reference", abs(diff_right)))

    if event != 'Top' and 'left_elbow_angle' in angles_to_check and 'right_elbow_angle' in angles_to_check:
        user_left, ref_left = angles_dict['left_elbow_angle']
        user_right, ref_right = angles_dict['right_elbow_angle']
        diff_left = user_left - ref_left
        diff_right = user_right - ref_right

        left_action = None
        right_action = None
        left_diff_msg = None
        right_diff_msg = None

        if abs(diff_left) > general_threshold:
            if user_left > ref_left:
                left_action = "bend less"
                left_diff_msg = f"{abs(diff_left):.1f} deg more"
            else:
                left_action = "bend more"
                left_diff_msg = f"{abs(diff_left):.1f} deg less"

        if abs(diff_right) > general_threshold:
            if user_right > ref_right:
                right_action = "bend less"
                right_diff_msg = f"{abs(diff_right):.1f} deg more"
            else:
                right_action = "bend more"
                right_diff_msg = f"{abs(diff_right):.1f} deg less"

        if left_action and right_action:
            if left_action == right_action:
                avg_diff = (abs(diff_left) + abs(diff_right)) / 2
                feedbacks.append((f"{left_action} elbows", f"Elbow angle is {avg_diff:.1f} deg {'more' if 'more' in left_diff_msg else 'less'} than reference", avg_diff))
            else:
                feedbacks.append((f"{left_action} left elbow", f"Left elbow angle is {left_diff_msg} than reference", abs(diff_left)))
                feedbacks.append((f"{right_action} right elbow", f"Right elbow angle is {right_diff_msg} than reference", abs(diff_right)))
        elif left_action:
            feedbacks.append((f"{left_action} left elbow", f"Left elbow angle is {left_diff_msg} than reference", abs(diff_left)))
        elif right_action:
            feedbacks.append((f"{right_action} right elbow", f"Right elbow angle is {right_diff_msg} than reference", abs(diff_right)))

    if event == 'Top' and 'left_elbow_angle' in angles_to_check:
        user_left, ref_left = angles_dict['left_elbow_angle']
        diff_left = user_left - ref_left
        if user_left < ref_left and abs(diff_left) > top_elbow_threshold:
            feedbacks.append(("straighten left arm at top", f"Left elbow angle is {abs(diff_left):.1f} deg less than reference", abs(diff_left)))

    if not feedbacks:
        feedbacks.append(("Swing well-aligned", "Your swing is well-aligned with the reference for this event", 0))

    # Separate the feedbacks into simple, detailed, and differences for compatibility with existing code
    simple_feedback = [fb[0] for fb in feedbacks]
    detailed_feedback = [fb[1] for fb in feedbacks]
    angle_differences = [fb[2] for fb in feedbacks]

    return simple_feedback, detailed_feedback, angle_differences, feedbacks

def resize_and_center(img, target_size=(1280, 720)):
    h, w = img.shape[:2]
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_size[0]) // 2
    y_offset = (target_size[1] - new_size[1]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    return canvas

def process_video(video_path, output_dir):
    ds = SampleVideo(video_path, transform=transforms.Compose([
        ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    device = torch.device("cpu")
    model = EventDetector(pretrain=True, width_mult=1.0, lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False).to(device)
    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()

    for sample in dl:
        images = sample['images'].to(device)
        batch = 0
        probs = None
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch)
            probs = F.softmax(logits.data, dim=1).cpu().numpy() if batch == 0 else np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    confidence = [probs[e, i] for i, e in enumerate(events)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    event_frames = {}
    for i, e in enumerate(events):
        if confidence[i] < confidence_threshold:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret or img is None:
            continue
        event_frames[event_names[i]] = img
        cv2.imwrite(os.path.join(output_dir, f"{event_names[i]}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cap.release()
    return event_frames

def generate_feedback_summary(feedback_data):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to user video', default='test_video.mp4')
    parser.add_argument('-r', '--reference', help='Path to reference video (Tiger Woods)', required=True)
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length
    confidence_threshold = 0

    print("Processing user's video...")
    user_event_frames = process_video(args.path, user_event_dir)

    print("Processing reference video...")
    reference_event_frames = process_video(args.reference, reference_event_dir)

    feedback_data = {}

    for event in event_order:
        if event not in user_event_frames or event not in reference_event_frames:
            continue

        user_img = resize_and_center(user_event_frames[event])
        ref_img = resize_and_center(reference_event_frames[event])

        user_rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        user_results = pose.process(user_rgb)
        ref_results = pose.process(ref_rgb)

        if not user_results.pose_landmarks or not ref_results.pose_landmarks:
            continue

        draw_pose(user_img, user_results.pose_landmarks)
        draw_pose(ref_img, ref_results.pose_landmarks)

        user_img_cropped = crop_black_borders(user_img)
        ref_img_cropped = crop_black_borders(ref_img)

        user_h, user_w = user_img_cropped.shape[:2]
        ref_h, ref_w = ref_img_cropped.shape[:2]

        user_angles = calculate_angles(user_results.pose_landmarks.landmark, user_img.shape)
        ref_angles = calculate_angles(ref_results.pose_landmarks.landmark, ref_img.shape)

        simple_feedback, detailed_feedback, angle_differences, feedbacks = generate_feedback(user_angles, ref_angles, event)
        feedback_data[event] = {
            'simple': simple_feedback,
            'detailed': detailed_feedback,
            'angle_differences': angle_differences,
            'feedbacks': [(simple, detailed, diff) for simple, detailed, diff in feedbacks]
        }

        feedback_height = (len(simple_feedback) + 2) * FEEDBACK_LINE_HEIGHT
        canvas_h = TITLE_HEIGHT + max(user_h, ref_h) + feedback_height + 20
        canvas_w = TEXT_AREA_WIDTH + user_w + GAP + ref_w + TEXT_AREA_WIDTH + COLUMN_GAP
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        title_text = f"Event: {event}"
        (tw, th), _ = cv2.getTextSize(title_text, FONT, 1.2, 2)
        title_x = (canvas_w - tw) // 2
        cv2.putText(canvas, title_text, (title_x, int(TITLE_HEIGHT * 0.7)), FONT, 1.2, (255, 255, 255), 2)

        data_y = TITLE_HEIGHT + 30
        cv2.putText(canvas, "User:", (10, data_y), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Shoulder Tilt: {user_angles[0]:.1f} deg", (10, data_y + 30), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Hip Rotation: {user_angles[1]:.1f} deg", (10, data_y + 60), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"L Elbow Angle: {user_angles[2]:.1f} deg", (10, data_y + 90), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"R Elbow Angle: {user_angles[3]:.1f} deg", (10, data_y + 120), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Spine Angle: {user_angles[4]:.1f} deg", (10, data_y + 150), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"L Knee Bend: {user_angles[5]:.1f} deg", (10, data_y + 180), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"R Knee Bend: {user_angles[6]:.1f} deg", (10, data_y + 210), FONT, 0.8, (255, 255, 255), 2)

        img_y = TITLE_HEIGHT
        canvas[img_y:img_y+user_h, TEXT_AREA_WIDTH:TEXT_AREA_WIDTH+user_w] = user_img_cropped
        ref_x = TEXT_AREA_WIDTH + user_w + GAP
        canvas[img_y:img_y+ref_h, ref_x:ref_x+ref_w] = ref_img_cropped

        ref_data_x = ref_x + ref_w + 10
        cv2.putText(canvas, "Reference:", (ref_data_x, data_y), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Shoulder Tilt: {ref_angles[0]:.1f} deg", (ref_data_x, data_y + 30), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Hip Rotation: {ref_angles[1]:.1f} deg", (ref_data_x, data_y + 60), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"L Elbow Angle: {ref_angles[2]:.1f} deg", (ref_data_x, data_y + 90), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"R Elbow Angle: {ref_angles[3]:.1f} deg", (ref_data_x, data_y + 120), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Spine Angle: {ref_angles[4]:.1f} deg", (ref_data_x, data_y + 150), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"L Knee Bend: {ref_angles[5]:.1f} deg", (ref_data_x, data_y + 180), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"R Knee Bend: {ref_angles[6]:.1f} deg", (ref_data_x, data_y + 210), FONT, 0.8, (255, 255, 255), 2)

        feedback_y = img_y + max(user_h, ref_h) + 20
        feedback_x_start = TEXT_AREA_WIDTH
        simple_header_x = feedback_x_start + (FEEDBACK_COLUMN_WIDTH - cv2.getTextSize("Simple Feedback", FONT, FONT_SCALE, FONT_THICKNESS)[0][0]) // 2
        detailed_header_x = feedback_x_start + FEEDBACK_COLUMN_WIDTH + COLUMN_GAP + (FEEDBACK_COLUMN_WIDTH - cv2.getTextSize("Detail", FONT, FONT_SCALE, FONT_THICKNESS)[0][0]) // 2
        cv2.putText(canvas, "Simple Feedback", (simple_header_x, feedback_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        cv2.putText(canvas, "Detail", (detailed_header_x, feedback_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        feedback_y += FEEDBACK_LINE_HEIGHT
        for i, (simple, detailed) in enumerate(zip(simple_feedback, detailed_feedback)):
            y_pos = feedback_y + i * FEEDBACK_LINE_HEIGHT
            (tw, th), _ = cv2.getTextSize(simple, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x_simple = feedback_x_start + (FEEDBACK_COLUMN_WIDTH - tw) // 2
            cv2.putText(canvas, simple, (text_x_simple, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            (tw, th), _ = cv2.getTextSize(detailed, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x_detailed = feedback_x_start + FEEDBACK_COLUMN_WIDTH + COLUMN_GAP + (FEEDBACK_COLUMN_WIDTH - tw) // 2
            cv2.putText(canvas, detailed, (text_x_detailed, y_pos), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        cv2.imshow(event, canvas)
        cv2.waitKey(0)
        out_path = os.path.join(combined_dir, f"{event.replace(' ', '_')}_comparison.jpg")
        success = cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            print(f"Saved comparison for '{event}' to {out_path}")
        else:
            print(f"Failed to save comparison for '{event}' to {out_path}")

    # Generate and display Feedback Summary
    summary_img = generate_feedback_summary(feedback_data)
    cv2.imshow("Feedback Summary", summary_img)
    cv2.waitKey(0)
    summary_path = os.path.join(combined_dir, "feedback_summary.jpg")
    success = cv2.imwrite(summary_path, summary_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if success:
        print(f"Saved Feedback Summary to {summary_path}")
    else:
        print(f"Failed to save Feedback Summary to {summary_path}")

    cv2.destroyAllWindows()
    pose.close()