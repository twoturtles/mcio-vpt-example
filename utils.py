import cv2
import numpy as np


def save_frames_to_mp4(
    frames: list[np.ndarray],
    filepath,
    to_bgr=True,
    fps: int = 20,
    codec="avc1",
):
    f = cv2.VideoWriter(
        filepath,
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    try:
        for frame in frames:
            if to_bgr:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_RGB2BGR)
            f.write(frame)
    finally:
        f.release()
