"""
analyse_swing.py

Command-line entry point:
1. Parses arguments (user video, reference video, sequence length).
2. Runs the SwingNet model to detect key frames.
3. Computes angles, generates per-event feedback, and builds comparison canvases.
4. Produces an overall feedback summary.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from constants import (
    USER_EVENT_DIR, REF_EVENT_DIR, COMBINED_DIR,
    pose, EVENT_ORDER, EVENT_NAMES,
    RELEVANT_KP, CUSTOM_CONNECTIONS,
    TITLE_HEIGHT, GAP, TEXT_AREA_WIDTH,
    FONT, FONT_SCALE, FONT_THICKNESS,
    FEEDBACK_COL_WIDTH, COLUMN_GAP,
    FEEDBACK_LINE_HEIGHT
)
from angle_compute import calculate_angles
from feedback import generate_feedback, generate_feedback_summary
from utils import ToTensor, Normalize
from model import EventDetector

# ------------------------------------------------------------------------------
# Video dataset wrapper: loads a video, resizes to square, pads,
# converts to RGB and returns all frames as a single sample.
# ------------------------------------------------------------------------------
class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        """
        Args:
            path (str): path to video file
            input_size (int): target height/width for network input
            transform (callable): transforms to apply to frames
        """
        self.path       = path
        self.input_size = input_size
        self.transform  = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        # Compute resize ratio and padding
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ratio = self.input_size / max(h, w)
        nh, nw = int(h * ratio), int(w * ratio)
        dh, dw = self.input_size - nh, self.input_size - nw
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                continue
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=[0.406*255, 0.456*255, 0.485*255]
            )
            frames.append(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError(f"No valid frames in {self.path}")

        sample = {'images': np.asarray(frames),
                  'labels': np.zeros(len(frames))}
        if self.transform:
            sample = self.transform(sample)
        return sample

# ------------------------------------------------------------------------------
# Utility functions for cropping black borders, drawing skeletons,
# resizing & centering images on a blank canvas.
# ------------------------------------------------------------------------------
def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(thr > 0)
    if not xs.size or not ys.size:
        return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def draw_pose(img, landmarks):
    """
    Draws skeleton lines & keypoints onto img in-place.
    Args:
        img (np.ndarray): BGR image.
        landmarks: MediaPipe pose landmarks.
    """
    h, w = img.shape[:2]
    # Draw bones
    for a, b in CUSTOM_CONNECTIONS:
        p1 = landmarks.landmark[a]
        p2 = landmarks.landmark[b]
        pt1 = (int(p1.x * w), int(p1.y * h))
        pt2 = (int(p2.x * w), int(p2.y * h))
        cv2.line(img, pt1, pt2, (0, 255, 0), 4)
    # Draw joints
    for idx in RELEVANT_KP:
        kp = landmarks.landmark[idx]
        cv2.circle(
            img,
            (int(kp.x * w), int(kp.y * h)),
            7, (0, 0, 255), -1
        )

def resize_and_center(img, target_size=(1280, 720)):
    """
    Resize img to fit within target_size, then center on blank canvas.
    Args:
        img (np.ndarray): BGR image.
        target_size (tuple): (width, height) of canvas.
    Returns:
        np.ndarray: canvas with centered image.
    """
    h, w = img.shape[:2]
    ratio = min(target_size[0]/w, target_size[1]/h)
    nw, nh = int(w * ratio), int(h * ratio)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    xo = (target_size[0] - nw)//2
    yo = (target_size[1] - nh)//2
    canvas[yo:yo+nh, xo:xo+nw] = small
    return canvas

def process_video(video_path, output_dir, seq_length, confidence_threshold):
    """
    Runs SwingNet event detection on the video.
    Saves the highest-confidence frame per event to output_dir.
    Returns:
        dict: {event_name: frame_image}
    """
    # Prepare dataset & loader
    ds = SampleVideo(
        video_path,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
        ])
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # Load the pre-trained SwingNet model
    device = torch.device("cpu")
    model  = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    ).to(device)
    ckpt = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Accumulate probabilities across all frames
    probs = None
    for sample in dl:
        imgs = sample['images'].to(device)
        batch = 0
        while batch * seq_length < imgs.shape[1]:
            end    = min((batch+1) * seq_length, imgs.shape[1])
            logits = model(imgs[:, batch*seq_length:end])
            chunk  = F.softmax(logits.data, dim=1).cpu().numpy()
            probs  = chunk if probs is None else np.append(probs, chunk, axis=0)
            batch += 1

    # Determine best frame per event
    events = np.argmax(probs, axis=0)[:-1]
    confs  = [probs[e, i] for i, e in enumerate(events)]

    cap    = cv2.VideoCapture(video_path)
    frames = {}
    for idx, ev in enumerate(events):
        if confs[idx] < confidence_threshold:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev)
        ret, img = cap.read()
        if not ret:
            continue
        name = EVENT_NAMES[idx]
        frames[name] = img
        cv2.imwrite(
            os.path.join(output_dir, f"{name}.jpg"),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
    cap.release()
    return frames

# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare user vs reference golf swing videos"
    )
    parser.add_argument(
        '-p', '--path',
        help="Path to user video",
        default='test_video.mp4'
    )
    parser.add_argument(
        '-r', '--reference',
        help="Path to reference video (Tiger Woods)",
        required=True
    )
    parser.add_argument(
        '-s', '--seq-length',
        type=int,
        default=64,
        help="Number of frames per model forward pass"
    )
    args = parser.parse_args()

    seq_len = args.seq_length
    conf_th = 0

    print("Processing user video...")
    user_frames = process_video(
        args.path, USER_EVENT_DIR, seq_len, conf_th
    )
    print("Processing reference video...")
    ref_frames  = process_video(
        args.reference, REF_EVENT_DIR, seq_len, conf_th
    )

    feedback_data = {}
    for event in EVENT_ORDER:
        # Skip if missing
        if event not in user_frames or event not in ref_frames:
            continue

        # Prepare images & run pose estimation
        u_img = resize_and_center(user_frames[event])
        r_img = resize_and_center(ref_frames[event])
        u_rgb = cv2.cvtColor(u_img, cv2.COLOR_BGR2RGB)
        r_rgb = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        u_res = pose.process(u_rgb)
        r_res = pose.process(r_rgb)
        if not u_res.pose_landmarks or not r_res.pose_landmarks:
            continue

        # Overlay skeletons
        draw_pose(u_img, u_res.pose_landmarks)
        draw_pose(r_img, r_res.pose_landmarks)

        # Crop borders, compute angles, generate feedback
        u_crop = crop_black_borders(u_img)
        r_crop = crop_black_borders(r_img)
        u_ang  = calculate_angles(u_res.pose_landmarks.landmark, u_img.shape)
        r_ang  = calculate_angles(r_res.pose_landmarks.landmark, r_img.shape)
        simple, detail, diffs, rawfb = generate_feedback(u_ang, r_ang, event)

        feedback_data[event] = {
            'simple':           simple,
            'detailed':         detail,
            'angle_differences': diffs,
            'feedbacks':        rawfb
        }

        # --- Build & show comparison canvas (identical to original) ---
        h_u, w_u = u_crop.shape[:2]
        h_r, w_r = r_crop.shape[:2]
        fb_h     = (len(simple) + 2) * FEEDBACK_LINE_HEIGHT
        canvas_h = TITLE_HEIGHT + max(h_u, h_r) + fb_h + 20
        canvas_w = (TEXT_AREA_WIDTH + w_u + GAP +
                    w_r + TEXT_AREA_WIDTH + COLUMN_GAP)
        canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Title
        title = f"Event: {event}"
        (tw, th), _ = cv2.getTextSize(title, FONT, 1.2, 2)
        cv2.putText(
            canvas, title,
            ((canvas_w - tw)//2, int(TITLE_HEIGHT*0.7)),
            FONT, 1.2, (255,255,255), 2
        )

        # User angle text
        dy = TITLE_HEIGHT + 30
        cv2.putText(canvas, "User:", (10, dy), FONT, 1.0, (255,255,255), 2)
        labels = [
            "Shoulder Tilt","Hip Rotation","L Elbow","R Elbow",
            "Spine","L Knee","R Knee"
        ]
        for i, val in enumerate(u_ang):
            cv2.putText(
                canvas,
                f"{labels[i]}: {val:.1f} deg",
                (10, dy + 30*(i+1)),
                FONT, 0.8, (255,255,255), 2
            )

        # Place user & reference images
        img_y = TITLE_HEIGHT
        canvas[img_y:img_y+h_u,
               TEXT_AREA_WIDTH:TEXT_AREA_WIDTH+w_u] = u_crop
        ref_x = TEXT_AREA_WIDTH + w_u + GAP
        canvas[img_y:img_y+h_r,
               ref_x:ref_x+w_r] = r_crop

        # Reference angle text
        dx = ref_x + w_r + 10
        cv2.putText(canvas, "Reference:", (dx, dy),
                    FONT, 1.0, (255,255,255), 2)
        for i, val in enumerate(r_ang):
            cv2.putText(
                canvas,
                f"{labels[i]}: {val:.1f} deg",
                (dx, dy + 30*(i+1)),
                FONT, 0.8, (255,255,255), 2
            )

        # Feedback table headers
        fy = img_y + max(h_u, h_r) + 20
        sx = TEXT_AREA_WIDTH
        cv2.putText(
            canvas, "Simple Feedback",
            (sx + (FEEDBACK_COL_WIDTH - cv2.getTextSize("Simple Feedback", FONT, FONT_SCALE, FONT_THICKNESS)[0][0])//2,
             fy),
            FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS
        )
        cv2.putText(
            canvas, "Detail",
            (sx + FEEDBACK_COL_WIDTH + COLUMN_GAP +
             (FEEDBACK_COL_WIDTH - cv2.getTextSize("Detail", FONT, FONT_SCALE, FONT_THICKNESS)[0][0])//2,
             fy),
            FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS
        )

        # Entries
        for i, (s, d) in enumerate(zip(simple, detail)):
            y_line = fy + FEEDBACK_LINE_HEIGHT*(i+1)
            tx1 = sx + (FEEDBACK_COL_WIDTH - cv2.getTextSize(s, FONT, FONT_SCALE, FONT_THICKNESS)[0][0])//2
            cv2.putText(canvas, s, (tx1, y_line), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
            tx2 = sx + FEEDBACK_COL_WIDTH + COLUMN_GAP + (
                (FEEDBACK_COL_WIDTH - cv2.getTextSize(d, FONT, FONT_SCALE, FONT_THICKNESS)[0][0])//2
            )
            cv2.putText(canvas, d, (tx2, y_line), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)

        # Show & save
        cv2.imshow(event, canvas)
        cv2.waitKey(0)
        out_file = os.path.join(
            COMBINED_DIR, f"{event.replace(' ', '_')}_comparison.jpg"
        )
        cv2.imwrite(out_file, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate & display summary
    summary = generate_feedback_summary(feedback_data)
    cv2.imshow("Feedback Summary", summary)
    cv2.waitKey(0)
    cv2.imwrite(
        os.path.join(COMBINED_DIR, "feedback_summary.jpg"),
        summary, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )

    cv2.destroyAllWindows()
    pose.close()
