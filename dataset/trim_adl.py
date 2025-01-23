import os
import sys
import glob
import subprocess
import tempfile
import shutil

import cv2


def get_duration_opencv(video_path):
    """
    Get video duration in seconds using OpenCV.
    Returns 0.0 on error or if duration can't be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frame_count / fps)


def center_trim_to_1_9(video_path, target_duration=1.9):
    """
    Trim the video to 1.9s centered in the middle of the clip.
    Overwrites the original file using ffmpeg re-encode (H.264 + AAC).
    If the video is shorter than 1.9s, we skip it with a warning.
    """
    dur = get_duration_opencv(video_path)
    if dur <= 0:
        print(f"WARNING: Could not read duration (or zero length): {video_path}")
        return

    if dur < target_duration:
        print(f"WARNING: Video {video_path} is only {dur:.2f}s, skipping because it's shorter than 1.9s.")
        return

    # Calculate start and end (centered)
    half = target_duration / 2
    start = (dur / 2) - half
    end = (dur / 2) + half

    # Safety clamp if floating issues
    if start < 0:
        start = 0
    if end > dur:
        end = dur

    # Temp file for re-encoded subclip
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        tmp_out
    ]
    print(f"[TRIM] {os.path.basename(video_path)}: {dur:.2f}s -> 1.9s (centered)")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Overwrite original
    shutil.move(tmp_out, video_path)
    print(f"[OK] Trimmed and overwrote {video_path}")


def main():
    # If script is called with an argument, interpret as the path to the "Fall" folder.
    # Otherwise, default to "./Fall".
    if len(sys.argv) > 1:
        fall_folder = sys.argv[1]
    else:
        fall_folder = "/Users/ivanursul/Downloads/Dataset/ADL"

    # Gather .mp4 files RECURSIVELY
    pattern = os.path.join(fall_folder, "**", "*.mp4")
    videos = glob.glob(pattern, recursive=True)

    if not videos:
        print(f"No .mp4 files found in {fall_folder}.")
        return

    for vid in videos:
        center_trim_to_1_9(vid, target_duration=1.9)

    print("Done processing all videos in Fall folder (recursive).")


if __name__ == "__main__":
    main()
