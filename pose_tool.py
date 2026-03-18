"""
Phase 5.1: MediaPipe Pose squat analysis tool.

Extracts 33 body landmarks per frame and computes squat-specific measurements:
- Knee angle (depth detection)
- Back angle deviation (form)
- Stance width ratio (stance classification)
- Knee-over-toe displacement
- Hold detection (pause at bottom)
- Visibility scoring

Usage:
  from pose_tool import analyze_squat
  result = analyze_squat("path/to/video.mp4", fps=2.0)

  # Or from CLI:
  python pose_tool.py path/to/video.mp4
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import os
os.environ["GLOG_minloglevel"] = "3"  # suppress mediapipe C++ warnings

import cv2
import numpy as np

try:
    import mediapipe as mp
    _vision = mp.tasks.vision
    _BaseOptions = mp.tasks.BaseOptions
    _MODEL_PATH = str(Path(__file__).parent / "pose_landmarker_lite.task")
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT = 31
RIGHT_FOOT = 32


def _angle_3pt(a, b, c) -> float:
    """Compute angle at point b given 3 points (in degrees)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _angle_from_vertical(a, b) -> float:
    """Angle between line a->b and vertical (degrees). 0 = perfectly vertical."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]  # positive = downward in image coords
    # Vertical is (0, 1) in image coords (downward)
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def _landmark_xy(landmarks, idx):
    """Extract (x, y) from landmark by index."""
    lm = landmarks[idx]
    return (lm.x, lm.y)


def _landmark_visibility(landmarks, idx) -> float:
    return landmarks[idx].visibility


def analyze_squat(video_path: str, fps: float = 2.0) -> dict:
    """
    Analyze a squat video using MediaPipe Pose.

    Args:
        video_path: Path to video file
        fps: Frame sampling rate

    Returns:
        dict with squat measurements and suggested labels
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    # Per-frame measurements
    knee_angles = []
    back_deviations = []
    stance_ratios = []
    knee_over_toe = []
    visibilities = []
    frame_count = 0

    options = _vision.PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with _vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp_ms = int(frame_idx * 1000 / video_fps)
            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not results.pose_landmarks:
                visibilities.append(0.0)
                continue

            lm = results.pose_landmarks[0]  # first (only) pose
            frame_count += 1

            # Visibility (average of key landmarks)
            key_indices = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
                          LEFT_SHOULDER, RIGHT_SHOULDER]
            avg_vis = np.mean([_landmark_visibility(lm, i) for i in key_indices])
            visibilities.append(avg_vis)

            if avg_vis < 0.3:
                continue  # Skip low-visibility frames

            # --- Knee angle (average of left and right) ---
            l_hip = _landmark_xy(lm, LEFT_HIP)
            r_hip = _landmark_xy(lm, RIGHT_HIP)
            l_knee = _landmark_xy(lm, LEFT_KNEE)
            r_knee = _landmark_xy(lm, RIGHT_KNEE)
            l_ankle = _landmark_xy(lm, LEFT_ANKLE)
            r_ankle = _landmark_xy(lm, RIGHT_ANKLE)
            l_shoulder = _landmark_xy(lm, LEFT_SHOULDER)
            r_shoulder = _landmark_xy(lm, RIGHT_SHOULDER)

            l_knee_angle = _angle_3pt(l_hip, l_knee, l_ankle)
            r_knee_angle = _angle_3pt(r_hip, r_knee, r_ankle)
            avg_knee = (l_knee_angle + r_knee_angle) / 2
            knee_angles.append(avg_knee)

            # --- Back angle (shoulder-hip line vs vertical) ---
            mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            back_dev = _angle_from_vertical(mid_hip, mid_shoulder)
            back_deviations.append(back_dev)

            # --- Stance width ratio (ankle distance / hip width) ---
            ankle_dist = math.sqrt((l_ankle[0] - r_ankle[0])**2 + (l_ankle[1] - r_ankle[1])**2)
            hip_dist = math.sqrt((l_hip[0] - r_hip[0])**2 + (l_hip[1] - r_hip[1])**2)
            if hip_dist > 0.01:
                stance_ratios.append(ankle_dist / hip_dist)

            # --- Knee over toe (horizontal displacement, normalized) ---
            # Positive = knee is forward of ankle
            l_kot = l_knee[0] - l_ankle[0]
            r_kot = r_knee[0] - r_ankle[0]
            # Use the more forward knee
            knee_over_toe.append(max(abs(l_kot), abs(r_kot)))

    cap.release()

    # Compute summary statistics
    if not knee_angles:
        return {
            "error": "no_pose_detected",
            "avg_landmark_visibility": float(np.mean(visibilities)) if visibilities else 0.0,
            "num_frames": frame_count,
            "suggested_labels": {
                "stance": None, "depth": None, "form_issues": None,
                "variant": None, "visible": False,
            },
        }

    knee_min = float(np.min(knee_angles))
    knee_max = float(np.max(knee_angles))
    knee_standing = float(np.max(knee_angles))
    back_max = float(np.max(back_deviations)) if back_deviations else 0.0
    stance_median = float(np.median(stance_ratios)) if stance_ratios else 1.0
    kot_max = float(np.max(knee_over_toe)) if knee_over_toe else 0.0
    avg_vis = float(np.mean(visibilities)) if visibilities else 0.0

    # Hold detection: check if knee angle stays within 2° for >1 second at bottom
    hold_duration = 0.0
    if len(knee_angles) > 2:
        # Find the bottom portion (within 5° of minimum)
        bottom_threshold = knee_min + 5
        consecutive_bottom = 0
        max_consecutive = 0
        for ka in knee_angles:
            if ka <= bottom_threshold:
                consecutive_bottom += 1
                max_consecutive = max(max_consecutive, consecutive_bottom)
            else:
                consecutive_bottom = 0
        # Convert frames to seconds
        hold_duration = max_consecutive / fps if fps > 0 else 0.0

    # --- Suggest labels based on measurements ---
    # Depth
    if knee_min > 110:
        depth = "shallow"
    elif knee_min > 85:
        depth = "90 degrees"
    else:
        depth = "over 90 degrees"

    # Stance
    if stance_median < 0.8:
        stance = "narrow"
    elif stance_median > 2.0:
        stance = "plie"
    elif stance_median > 1.5:
        stance = "wide"
    else:
        stance = "shoulder-width"

    # Form issues
    form_issues = []
    if back_max > 20:
        form_issues.append("back not straight")
    if kot_max > 0.08:  # Normalized threshold
        form_issues.append("knees over toes")

    # Variant
    variant = "hold" if hold_duration > 1.0 else None

    result = {
        "knee_angle_min": round(knee_min, 1),
        "knee_angle_standing": round(knee_standing, 1),
        "back_deviation_max_deg": round(back_max, 1),
        "stance_ratio": round(stance_median, 2),
        "knee_over_toe_normalized": round(kot_max, 3),
        "hold_duration_sec": round(hold_duration, 1),
        "avg_landmark_visibility": round(avg_vis, 3),
        "num_frames": frame_count,
        "num_pose_frames": len(knee_angles),
        "suggested_labels": {
            "stance": stance,
            "depth": depth,
            "form_issues": form_issues,
            "variant": variant,
            "visible": True,
        },
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python pose_tool.py <video_path> [fps]")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    result = analyze_squat(video_path, fps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
