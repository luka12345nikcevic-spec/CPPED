import cv2
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from collections import deque
from skimage.metrics import structural_similarity as ssim

# ==========================
# CONFIG (CHANGE THESE)
# ==========================
INPUT_VIDEO_FOLDER = "/content/videos/"
OUTPUT_FRAME_FOLDER = "/content/frames"
FRAMES_PER_VIDEO = 50
YOLO_MODEL = "yolov8m.pt"
CONF_THRESHOLD = 0.35           # minimum confidence for a person detection
PERSON_CLASS_ID = 0            # COCO class id for "person"
DEVICE = 0                     # 0 = first GPU, "cpu" = CPU

# ==========================
# SETUP
# ==========================
os.makedirs(OUTPUT_FRAME_FOLDER, exist_ok=True)
model = YOLO(YOLO_MODEL)


# SSIM duplicate filtering config
SSIM_THRESHOLD = 0.90
COMPARE_LAST_K = 3
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480


def compute_ssim(frame1, frame2):
    score, _ = ssim(frame1, frame2, full=True)
    return score


def is_duplicate(frame_gray, prev_frames, threshold):
    for prev in prev_frames:
        if compute_ssim(prev, frame_gray) > threshold:
            return True
    return False

def has_person(frame):
    """Run YOLO on a frame and return True if at least one person is detected."""
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        device=DEVICE,
        verbose=False,
    )
    # results[0].boxes holds detections; if any survived the class+conf filter, a person is present
    return len(results[0].boxes) > 0


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Skipping {video_path.name}: no frames")
        cap.release()
        return

    # Same logic as before: evenly spaced frame indices across the whole video
    indices = np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int,
    )

    saved_count = 0
    prev_frames = deque(maxlen=COMPARE_LAST_K)   # <-- here
    skipped_no_person = 0
    skipped_duplicate = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Keep only frames containing at least one person
        if not has_person(frame):
            skipped_no_person += 1
            continue

        frame_small = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if is_duplicate(frame_gray, prev_frames, SSIM_THRESHOLD):
            skipped_duplicate += 1
            continue

        filename = f"{video_path.stem}_frame_{saved_count:03d}.jpg"
        save_path = os.path.join(OUTPUT_FRAME_FOLDER, filename)
        cv2.imwrite(save_path, frame)
        prev_frames.append(frame_gray)
        saved_count += 1

    cap.release()
    print(
        f"{video_path.name}: saved {saved_count} frames "
        f"(skipped {skipped_no_person} with no person)"
        f"(skipped {skipped_duplicate} with duplicate)"

    )


def main():
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_folder = Path(INPUT_VIDEO_FOLDER)

    for video_file in sorted(video_folder.iterdir()):
        if video_file.suffix.lower() in video_extensions:
            extract_frames(video_file)


if __name__ == "__main__":
    main()