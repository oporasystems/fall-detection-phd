import os
import sys
import glob
import subprocess
import tempfile
import shutil

def remove_audio_from_video(video_path):
    """
    Remove the audio track from the given MP4 file in-place.
    Uses ffmpeg with -an (no audio) and copies the video track.
    """
    # Create a temporary output file
    import tempfile
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    # ffmpeg command
    # -an removes audio track, -c:v copy just copies the existing video without re-encoding.
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "copy",
        "-an",
        tmp_out
    ]
    print(f"[Removing audio] {os.path.basename(video_path)}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Overwrite original file with the audio-removed version
    shutil.move(tmp_out, video_path)
    print(f"[OK] Audio removed from {video_path}")

def main():
    """
    Recursively remove audio from all .mp4 videos under the specified folder (or './videos' by default).
    Usage: python remove_audio_recursive.py /path/to/folder
    """
    if len(sys.argv) > 1:
        root_folder = sys.argv[1]
    else:
        root_folder = "/Users/ivanursul/Downloads/Dataset/"  # or your default directory

    # Recursively find .mp4 files
    pattern = os.path.join(root_folder, "**", "*.mp4")
    videos = glob.glob(pattern, recursive=True)

    if not videos:
        print(f"No .mp4 files found in {root_folder}. Exiting.")
        return

    for vid in videos:
        remove_audio_from_video(vid)

    print("Done! All videos have had their audio removed.")

if __name__ == "__main__":
    main()
