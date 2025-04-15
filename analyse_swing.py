import argparse
import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import ToTensor, Normalize
from model import EventDetector
import torch.nn.functional as F
import mediapipe as mp
import os

# Create directory for annotated images if it doesn't exist
output_dir = 'annotated_images'
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Event names
event_names = {
    0: 'Address', 1: 'Toe-up', 2: 'Mid-backswing (arm parallel)', 3: 'Top',
    4: 'Mid-downswing (arm parallel)', 5: 'Impact', 6: 'Mid-follow-through (shaft parallel)', 7: 'Finish'
}

# Relevant keypoints for golf swing
relevant_keypoints = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Nose, shoulders, elbows, wrists, hips
custom_connections = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (23, 24)             # Hips
]

# Alignment point options: 'hips', 'torso', 'nose', 'shoulders'
alignment_point = 'shoulders'  # Default: align using hips midpoint

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
                continue  # Skip invalid frames
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        if not images:
            raise ValueError(f"No valid frames could be read from {self.path}. Check if the file is corrupted or supported.")

        labels = np.zeros(len(images))  # For transform compatibility
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([
        ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    device = torch.device("cpu")  # Force CPU for M1 Mac compatibility
    model = EventDetector(pretrain=True, width_mult=1.0, lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False).to(device)
    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    print("Loaded model weights")

    # Load Tiger Woods' poses
    with open('tiger_poses.pkl', 'rb') as f:
        tiger_poses = pickle.load(f)

    print('Testing...')
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
    print('Predicted event frames: {}'.format(events))

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.path}")

    confidence = [probs[e, i] for i, e in enumerate(events)]
    print('Confidence: {}'.format([np.round(c, 3) for c in confidence]))

    # Desired output image size
    output_size = (1280, 720)  # Width, Height

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret or img is None:
            print(f"Warning: Could not read frame {e} for event {event_names[i]}")
            continue

        # Resize frame to output size while preserving aspect ratio
        img_height, img_width = img.shape[:2]
        ratio = min(output_size[0] / img_width, output_size[1] / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        # Create a blank canvas of output_size and center the resized image
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        x_offset = (output_size[0] - new_size[0]) // 2
        y_offset = (output_size[1] - new_size[1]) // 2
        canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = img_resized
        img = canvas

        # User's pose estimation
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            # Draw user's pose (red) for relevant keypoints
            for idx in relevant_keypoints:
                kp = results.pose_landmarks.landmark[idx]
                x = int(kp.x * img.shape[1])
                y = int(kp.y * img.shape[0])
                cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            for conn in custom_connections:
                start_kp = results.pose_landmarks.landmark[conn[0]]
                end_kp = results.pose_landmarks.landmark[conn[1]]
                start_point = (int(start_kp.x * img.shape[1]), int(start_kp.y * img.shape[0]))
                end_point = (int(end_kp.x * img.shape[1]), int(end_kp.y * img.shape[0]))
                cv2.line(img, start_point, end_point, (0, 0, 255), 4)

            # Compute user's alignment center and shoulder width
            if alignment_point == 'hips':
                user_left = results.pose_landmarks.landmark[23]
                user_right = results.pose_landmarks.landmark[24]
                user_center_x = (user_left.x + user_right.x) / 2 * img.shape[1]
                user_center_y = (user_left.y + user_right.y) / 2 * img.shape[0]
            elif alignment_point == 'torso':
                user_shoulder_left = results.pose_landmarks.landmark[11]
                user_shoulder_right = results.pose_landmarks.landmark[12]
                user_hip_left = results.pose_landmarks.landmark[23]
                user_hip_right = results.pose_landmarks.landmark[24]
                user_center_x = (user_shoulder_left.x + user_shoulder_right.x + user_hip_left.x + user_hip_right.x) / 4 * img.shape[1]
                user_center_y = (user_shoulder_left.y + user_shoulder_right.y + user_hip_left.y + user_hip_right.y) / 4 * img.shape[0]
            elif alignment_point == 'nose':
                user_nose = results.pose_landmarks.landmark[0]
                user_center_x = user_nose.x * img.shape[1]
                user_center_y = user_nose.y * img.shape[0]
            elif alignment_point == 'shoulders':
                user_shoulder_left = results.pose_landmarks.landmark[11]
                user_shoulder_right = results.pose_landmarks.landmark[12]
                user_center_x = (user_shoulder_left.x + user_shoulder_right.x) / 2 * img.shape[1]
                user_center_y = (user_shoulder_left.y + user_shoulder_right.y) / 2 * img.shape[0]
            else:
                raise ValueError(f"Unknown alignment_point: {alignment_point}")

            # Compute user's shoulder width for scaling
            user_shoulder_left = results.pose_landmarks.landmark[11]
            user_shoulder_right = results.pose_landmarks.landmark[12]
            user_shoulder_width = np.sqrt(
                (user_shoulder_left.x * img.shape[1] - user_shoulder_right.x * img.shape[1])**2 +
                (user_shoulder_left.y * img.shape[0] - user_shoulder_right.y * img.shape[0])**2
            )

            # Overlay Tiger's pose (green) with alignment and scaling
            tiger_keypoints = tiger_poses.get(event_names[i], [])
            if len(tiger_keypoints) >= max(relevant_keypoints) + 1:
                # Compute Tiger's alignment center and shoulder width
                if alignment_point == 'hips':
                    tiger_left = tiger_keypoints[23]
                    tiger_right = tiger_keypoints[24]
                    tiger_center_x = (tiger_left['x'] + tiger_right['x']) / 2 * img.shape[1]
                    tiger_center_y = (tiger_left['y'] + tiger_right['y']) / 2 * img.shape[0]
                elif alignment_point == 'torso':
                    tiger_shoulder_left = tiger_keypoints[11]
                    tiger_shoulder_right = tiger_keypoints[12]
                    tiger_hip_left = tiger_keypoints[23]
                    tiger_hip_right = tiger_keypoints[24]
                    tiger_center_x = (tiger_shoulder_left['x'] + tiger_shoulder_right['x'] + tiger_hip_left['x'] + tiger_hip_right['x']) / 4 * img.shape[1]
                    tiger_center_y = (tiger_shoulder_left['y'] + tiger_shoulder_right['y'] + tiger_hip_left['y'] + tiger_hip_right['y']) / 4 * img.shape[0]
                elif alignment_point == 'nose':
                    tiger_nose = tiger_keypoints[0]
                    tiger_center_x = tiger_nose['x'] * img.shape[1]
                    tiger_center_y = tiger_nose['y'] * img.shape[0]
                elif alignment_point == 'shoulders':
                    tiger_shoulder_left = tiger_keypoints[11]
                    tiger_shoulder_right = tiger_keypoints[12]
                    tiger_center_x = (tiger_shoulder_left['x'] + tiger_shoulder_right['x']) / 2 * img.shape[1]
                    tiger_center_y = (tiger_shoulder_left['y'] + tiger_shoulder_right['y']) / 2 * img.shape[0]

                tiger_shoulder_left = tiger_keypoints[11]
                tiger_shoulder_right = tiger_keypoints[12]
                tiger_shoulder_width = np.sqrt(
                    ((tiger_shoulder_left['x'] - tiger_shoulder_right['x']) * img.shape[1])**2 +
                    ((tiger_shoulder_left['y'] - tiger_shoulder_right['y']) * img.shape[0])**2
                )

                # Calculate offset and scale
                offset_x = user_center_x - tiger_center_x
                offset_y = user_center_y - tiger_center_y
                scale_factor = user_shoulder_width / tiger_shoulder_width if tiger_shoulder_width > 0 else 1.0

                # Draw Tiger's keypoints with offset and scale
                for idx in relevant_keypoints:
                    kp = tiger_keypoints[idx]
                    scaled_x = (kp['x'] * img.shape[1] - tiger_center_x) * scale_factor + user_center_x
                    scaled_y = (kp['y'] * img.shape[0] - tiger_center_y) * scale_factor + user_center_y
                    x = int(scaled_x)
                    y = int(scaled_y)
                    cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
                for conn in custom_connections:
                    start_kp = tiger_keypoints[conn[0]]
                    end_kp = tiger_keypoints[conn[1]]
                    start_x = (start_kp['x'] * img.shape[1] - tiger_center_x) * scale_factor + user_center_x
                    start_y = (start_kp['y'] * img.shape[0] - tiger_center_y) * scale_factor + user_center_y
                    end_x = (end_kp['x'] * img.shape[1] - tiger_center_x) * scale_factor + user_center_x
                    end_y = (end_kp['y'] * img.shape[0] - tiger_center_y) * scale_factor + user_center_y
                    start_point = (int(start_x), int(start_y))
                    end_point = (int(end_x), int(end_y))
                    cv2.line(img, start_point, end_point, (0, 255, 0), 4)

                # Highlight differences (smaller blue circles) for relevant keypoints
                threshold = 0.05  # Adjust as needed
                for idx in relevant_keypoints:
                    ukp = results.pose_landmarks.landmark[idx]
                    tkp = tiger_keypoints[idx]
                    tiger_x = ((tkp['x'] * img.shape[1] - tiger_center_x) * scale_factor + user_center_x) / img.shape[1]
                    tiger_y = ((tkp['y'] * img.shape[0] - tiger_center_y) * scale_factor + user_center_y) / img.shape[0]
                    distance = np.sqrt((ukp.x - tiger_x)**2 + (ukp.y - tiger_y)**2)
                    if distance > threshold:
                        x = int(ukp.x * img.shape[1])
                        y = int(ukp.y * img.shape[0])
                        cv2.circle(img, (x, y), 6, (255, 0, 0), 2)

        # Add confidence text with scaled font
        cv2.putText(img, '{:.3f}'.format(confidence[i]), (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)

        # Display resized image
        display_img = cv2.resize(img, output_size)  # Ensure display matches output size
        cv2.imshow(event_names[i], display_img)
        cv2.waitKey(0)

        # Save annotated image
        event_name = event_names.get(i, f"event_{i}")
        filename = f"{i}_{event_name.replace(' ', '_')}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Saved annotated frame for '{event_name}' to {output_path}")


    cap.release()
    cv2.destroyAllWindows()
    pose.close()