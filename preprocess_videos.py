import os
import cv2
import numpy as np

def preprocess_local_video(input_path, output_path, dim=160, start_frame=0, end_frame=None, rotate=True):
    """
    Processes a local portrait video:
    - Rotates it to fix orientation (if needed)
    - Center-crops to square
    - Resizes to dim x dim
    - Saves processed video

    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        dim (int): Target size (square)
        start_frame (int): Frame to start processing
        end_frame (int or None): Frame to stop processing (None = till end)
        rotate (bool): Whether to rotate the video 90 degrees clockwise
    """
    if os.path.isfile(output_path):
        print(f"{output_path} already exists. Overwriting...")
        os.remove(output_path)

    print(f"Processing {input_path}...")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = end_frame if end_frame is not None else total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (dim, dim))

    frame_idx = 0
    success, frame = cap.read()

    while success:
        if start_frame <= frame_idx < end_frame:
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            h, w = frame.shape[:2]
            min_dim = min(h, w)
            x_start = (w - min_dim) // 2
            y_start = (h - min_dim) // 2
            square_frame = frame[y_start:y_start + min_dim, x_start:x_start + min_dim]

            resized = cv2.resize(square_frame, (dim, dim))
            out.write(resized)

        elif frame_idx >= end_frame:
            break

        success, frame = cap.read()
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished processing. Saved to {output_path}")


# Example usage
if __name__ == '__main__':
    input_video = 'test_video.mp4'                  # 👈 Replace with your input file path
    output_video = 'processed_test_video.mp4'       # 👈 Replace with your output path
    preprocess_local_video(input_video, output_video, dim=400, rotate=True)
