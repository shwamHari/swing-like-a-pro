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
    (27, 31), (28, 32)   # Ankles to feet
]

# Layout constants
TITLE_HEIGHT = 60
GAP = 50  # Small gap between images
TEXT_AREA_WIDTH = 350  # Width reserved for user/reference data
FONT = cv2.FONT_HERSHEY_DUPLEX

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
        return img  # Return original if entirely black
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

def compute_elbow_angle(shoulder, elbow, wrist, img_shape):
    p1 = (shoulder.x * img_shape[1], shoulder.y * img_shape[0])
    p2 = (elbow.x * img_shape[1], elbow.y * img_shape[0])
    p3 = (wrist.x * img_shape[1], wrist.y * img_shape[0])
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def calculate_angles(landmarks, img_shape):
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    hip_left = landmarks[23]
    hip_right = landmarks[24]
    elbow_left = landmarks[13]
    wrist_left = landmarks[15]

    shoulder_tilt = calculate_angle(
        (shoulder_left.x * img_shape[1], shoulder_left.y * img_shape[0]),
        (shoulder_right.x * img_shape[1], shoulder_right.y * img_shape[0])
    )
    hip_rotation = calculate_angle(
        (hip_left.x * img_shape[1], hip_left.y * img_shape[0]),
        (hip_right.x * img_shape[1], hip_right.y * img_shape[0])
    )
    elbow_angle = compute_elbow_angle(
        shoulder_left, elbow_left, wrist_left, img_shape
    )
    return shoulder_tilt, hip_rotation, elbow_angle

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to user video', default='test_video.mp4')
    parser.add_argument('-r', '--reference', help='Path to reference video (Tiger Woods)', required=True)
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length
    confidence_threshold = 0.5

    print("Processing user's video...")
    user_event_frames = process_video(args.path, user_event_dir)

    print("Processing reference video...")
    reference_event_frames = process_video(args.reference, reference_event_dir)

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

        # Crop black borders from images
        user_img_cropped = crop_black_borders(user_img)
        ref_img_cropped = crop_black_borders(ref_img)

        user_h, user_w = user_img_cropped.shape[:2]
        ref_h, ref_w = ref_img_cropped.shape[:2]

        user_angles = calculate_angles(user_results.pose_landmarks.landmark, user_img.shape)
        ref_angles = calculate_angles(ref_results.pose_landmarks.landmark, ref_img.shape)

        # Prepare canvas with dynamic width based on cropped images
        canvas_h = TITLE_HEIGHT + max(user_h, ref_h)
        canvas_w = TEXT_AREA_WIDTH + user_w + GAP + ref_w + TEXT_AREA_WIDTH
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Title strip
        title_text = f"Event: {event}"
        (tw, th), _ = cv2.getTextSize(title_text, FONT, 1.2, 2)
        title_x = (canvas_w - tw) // 2
        cv2.putText(canvas, title_text, (title_x, int(TITLE_HEIGHT * 0.7)), FONT, 1.2, (255, 255, 255), 2)

        # User data
        data_y = TITLE_HEIGHT + 30
        cv2.putText(canvas, "User:", (10, data_y), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Shoulder Tilt: {user_angles[0]:.1f}\u00b0", (10, data_y + 30), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Hip Rotation: {user_angles[1]:.1f}\u00b0", (10, data_y + 60), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Elbow Angle: {user_angles[2]:.1f}\u00b0", (10, data_y + 90), FONT, 0.8, (255, 255, 255), 2)

        # Place cropped images
        img_y = TITLE_HEIGHT
        canvas[img_y:img_y+user_h, TEXT_AREA_WIDTH:TEXT_AREA_WIDTH+user_w] = user_img_cropped
        ref_x = TEXT_AREA_WIDTH + user_w + GAP
        canvas[img_y:img_y+ref_h, ref_x:ref_x+ref_w] = ref_img_cropped

        # Reference data
        ref_data_x = ref_x + ref_w + 10
        cv2.putText(canvas, "Reference:", (ref_data_x, data_y), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Shoulder Tilt: {ref_angles[0]:.1f}\u00b0", (ref_data_x, data_y + 30), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Hip Rotation: {ref_angles[1]:.1f}\u00b0", (ref_data_x, data_y + 60), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Elbow Angle: {ref_angles[2]:.1f}\u00b0", (ref_data_x, data_y + 90), FONT, 0.8, (255, 255, 255), 2)

        # Display & save
        cv2.imshow(event, canvas)
        cv2.waitKey(0)
        out_path = os.path.join(combined_dir, f"{event.replace(' ', '_')}_comparison.jpg")
        cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved comparison for '{event}' to {out_path}")

    cv2.destroyAllWindows()
    pose.close()