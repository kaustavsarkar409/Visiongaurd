from __future__ import annotations

import argparse
from collections import deque

import cv2

from action_model import ViolenceActionRecognizer
from detection_config import ACTION_CLIP_LENGTH


def run_smoke(video_path: str | None) -> None:
    recognizer = ViolenceActionRecognizer()
    print("Action model enabled:", recognizer.enabled)
    if recognizer.last_error:
        print("Action model load error:", recognizer.last_error)
    if not recognizer.enabled:
        return

    source = 0 if video_path is None else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open source:", source)
        return

    buffer = deque(maxlen=ACTION_CLIP_LENGTH)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        buffer.append(frame)
        if len(buffer) == ACTION_CLIP_LENGTH and frame_idx % 8 == 0:
            pred = recognizer.predict(list(buffer))
            print(
                f"frame={frame_idx} label={pred.label} "
                f"conf={pred.confidence:.3f} violence={pred.is_violence}"
            )
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test action model inference.")
    parser.add_argument("--video", type=str, default=None, help="Optional video path.")
    args = parser.parse_args()
    run_smoke(args.video)
