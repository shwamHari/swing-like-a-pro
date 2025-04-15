import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import ToTensor, Normalize
from model import EventDetector
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Event names from test_video.py
event_names = {
    0: 'Address', 1: 'Toe-up', 2: 'Mid-backswing (arm parallel)', 3: 'Top',
    4: 'Mid-downswing (arm parallel)', 5: 'Impact', 6: 'Mid-follow-through (shaft parallel)', 7: 'Finish'
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
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Video path
tiger_video_path = 'tiger_swing.mp4'

# Preprocess video
ds = SampleVideo(tiger_video_path, transform=transforms.Compose([
    ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

# Load SwingNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EventDetector(pretrain=True, width_mult=1., lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False).to(device)
save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
model.load_state_dict(save_dict['model_state_dict'])
model.eval()

# Detect events
for sample in dl:
    images = sample['images'].to(device)
    batch = 0
    seq_length = 64
    while batch * seq_length < images.shape[1]:
        if (batch + 1) * seq_length > images.shape[1]:
            image_batch = images[:, batch * seq_length:, :, :, :]
        else:
            image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
        logits = model(image_batch)
        probs = torch.softmax(logits.data, dim=1).cpu().numpy() if batch == 0 else np.append(probs, torch.softmax(logits.data, dim=1).cpu().numpy(), 0)
        batch += 1

events = np.argmax(probs, axis=0)[:-1]  # Exclude no-event class

# Extract poses
cap = cv2.VideoCapture(tiger_video_path)
tiger_poses = {}
for i, e in enumerate(events):
    cap.set(cv2.CAP_PROP_POS_FRAMES, e)
    ret, img = cap.read()
    if not ret:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        keypoints = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.pose_landmarks.landmark]
        tiger_poses[event_names[i]] = keypoints

cap.release()
pose.close()

# Save poses
with open('tiger_poses.pkl', 'wb') as f:
    pickle.dump(tiger_poses, f)

print("Tiger Woods' poses saved to tiger_poses.pkl")