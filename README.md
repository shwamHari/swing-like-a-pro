# SwingLikeAPro ⛳🏌️💻

_A golf swing analysis system that compares your biomechanics with professional golfers using AI and computer vision._

## 🚀 Overview

SwingLikeAPro helps golfers improve their technique by:
- Detecting 8 key swing phases using LSTM neural networks
- Analyzing 7 critical biomechanical angles via pose estimation
- Providing actionable feedback comparing swings to professional references
- Demonstrating proven results: **4.5m average distance gain** and **1.3m reduced shot curvature**


## 🔍 Key Features

### 🎯 Swing Phase Detection
Identifies 8 key swing events:
1. Address
2. Toe-up
3. Mid-backswing
4. Top
5. Mid-downswing
6. Impact
7. Mid-follow-through
8. Finish

### 📐 Biomechanical Analysis
Measures seven critical angles:
```python
['shoulder_tilt', 'hip_rotation', 'left_elbow', 'right_elbow',
 'spine_angle', 'left_knee', 'right_knee']
```

### 📊 Smart Feedback System
- Rule-based engine with 10° sensitivity threshold
- Generates personalized improvement suggestions
- Provides professional comparison overlays

## ⚙️ Technical Architecture

```
Input Video (.mp4)
   │
   ├──▶ Preprocessing (Resizing, Orientation Correction)
   │        └─ Ensures correct aspect ratio and upright positioning for both user and reference videos
   │
   ├──▶ LSTM Neural Network (Swing Phase Detection)
   │
   ├──▶ MediaPipe Pose Estimation (Joint Detection)
   │
   ├──▶ Angle Calculation (Biomechanics Analysis)
   │
   ├──▶ Feedback Engine (Suggestions & Thresholds)
   │
   └──▶ Visualization (Side-by-side Comparison Frames & Summary Report)
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/SwingLikeAPro.git
   cd SwingLikeAPro
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models**:
   - Create a directory:
     ```bash
     mkdir -p models/
     ```
   - Download and place models:
     - **MobileNetV2 weights**:
       - Download from: https://github.com/tonylins/pytorch-mobilenet-v2
       - Place in `models/` directory.
     - **SwingNet model**:
       - Download from: https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view
       - Place in `models/` directory.

---

## 🏌️ How to Use

### 🛠️ Step 1: Preprocess Your Videos

Before running the analysis, it's important to preprocess the videos to ensure correct orientation and size. This avoids issues such as sideways videos or incorrect aspect ratios.

Use the provided `preprocess_video.py` script on **both your swing video** and the **reference video**.

Edit the `__main__` section as needed
```bash
if __name__ == '__main__':
    input_video = 'video_to_be_processed.mp4'     # Replace with your input file path
    output_video = 'output_video_name.mp4'        # Replace with your desired output path
    preprocess_local_video(input_video, output_video, dim=400, rotate=True)
```

**Configuration Options:**
- `rotate=True` or `False` depending on video orientation. If videos are sideways when running `analyse_swing.py` then change this variable
- `dim=400` for output video resolution (adjustable)

### 🎥 Step 2: Analyze Your Swing

Run the analysis with:

```bash
python analyse_swing.py -p your_swing.mp4  --reference reference_swing.mp4
```

### ✅ Example

```bash
python analyse_swing.py -p test_video.mp4 --reference tiger_swing.mp4
```

### 📁 Results

Generated output will be as images stored in:

```
├── user_event_frames/                     # Your key swing frames
├── reference_event_frames/                # Professional reference frames
├── combined_images/                       # Side-by-side analysis frames
└── combined_images/feedback_summary.jpg   # Visual feedback summary
```
