import sys
import os
import glob
import tempfile
import shutil
import subprocess
import uuid

import cv2
from PySide6.QtCore import (
    Qt, QUrl, Signal, QRect, QPoint
)
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QFileDialog
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


def find_mp4_files(folder):
    """Recursively gather all .mp4 files under `folder`."""
    return glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True)


def get_video_duration(file_path):
    """
    Return the duration of the video in seconds (float),
    using OpenCV's CAP_PROP_FRAME_COUNT / CAP_PROP_FPS.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frame_count / fps)


class FixedWindowRangeSlider(QWidget):
    """
    A custom two-handle slider that locks the difference between the handles
    to a fixed window size (the subclip length).

    Emits:
      - `valuesChanged(float, float)`: whenever the handles move
      - `editingFinished()`: once the user releases the mouse button after dragging
    """

    valuesChanged = Signal(float, float)
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)

        self._minValue = 0.0
        self._maxValue = 100.0

        # We keep (upperValue - lowerValue) = _fixedWindowSize
        self._lowerValue = 0.0
        self._upperValue = 100.0
        self._fixedWindowSize = 100.0

        self._draggingLower = False
        self._draggingUpper = False
        self._handleRadius = 12

        # Colors
        self._trackColor = QColor("#999999")
        self._selectedColor = QColor("#64B0FF")
        self._handleColor = QColor("#1E90FF")

    def setRange(self, minVal, maxVal):
        """Set the overall allowed range for the slider."""
        self._minValue = minVal
        self._maxValue = maxVal

        # If the window is bigger than the entire range, clamp it
        totalRange = self._maxValue - self._minValue
        if self._fixedWindowSize > totalRange:
            self._fixedWindowSize = totalRange

        # Initialize to [0, fixedWindowSize] in that range
        self._lowerValue = self._minValue
        self._upperValue = self._minValue + self._fixedWindowSize
        if self._upperValue > self._maxValue:
            self._upperValue = self._maxValue
            self._lowerValue = self._upperValue - self._fixedWindowSize
        self.update()

    def setFixedWindowSize(self, size):
        self._fixedWindowSize = size

    def values(self):
        """Return the (lowerValue, upperValue)."""
        return (self._lowerValue, self._upperValue)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        track_height = 6
        track_y = h // 2 - track_height // 2

        xLower = self.valueToX(self._lowerValue)
        xUpper = self.valueToX(self._upperValue)

        # Draw track
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._trackColor)
        painter.drawRect(0, track_y, w, track_height)

        # Draw selected range
        painter.setBrush(self._selectedColor)
        leftX = min(xLower, xUpper)
        width = abs(xUpper - xLower)
        painter.drawRect(leftX, track_y, width, track_height)

        # Draw handles
        painter.setBrush(self._handleColor)
        painter.setPen(QPen(Qt.black, 1))

        painter.drawEllipse(QPoint(xLower, h // 2), self._handleRadius, self._handleRadius)
        painter.drawEllipse(QPoint(xUpper, h // 2), self._handleRadius, self._handleRadius)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            xPos = event.position().x()
            distLower = abs(xPos - self.valueToX(self._lowerValue))
            distUpper = abs(xPos - self.valueToX(self._upperValue))

            if distLower <= self._handleRadius + 2 and distLower <= distUpper:
                self._draggingLower = True
            elif distUpper <= self._handleRadius + 2:
                self._draggingUpper = True

    def mouseMoveEvent(self, event):
        if self._draggingLower or self._draggingUpper:
            x = event.position().x()
            val = self.xToValue(x)

            if self._draggingLower:
                # Move lower handle -> recalc upper
                newLower = val
                newUpper = newLower + self._fixedWindowSize
                if newUpper > self._maxValue:
                    newUpper = self._maxValue
                    newLower = newUpper - self._fixedWindowSize
                if newLower < self._minValue:
                    newLower = self._minValue
                    newUpper = newLower + self._fixedWindowSize
                self._lowerValue = newLower
                self._upperValue = newUpper
            else:
                # Move upper handle -> recalc lower
                newUpper = val
                newLower = newUpper - self._fixedWindowSize
                if newLower < self._minValue:
                    newLower = self._minValue
                    newUpper = newLower + self._fixedWindowSize
                if newUpper > self._maxValue:
                    newUpper = self._maxValue
                    newLower = newUpper - self._fixedWindowSize
                self._lowerValue = newLower
                self._upperValue = newUpper

            self.update()
            self.valuesChanged.emit(self._lowerValue, self._upperValue)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._draggingLower or self._draggingUpper:
            # The user just finished a handle drag
            self.editingFinished.emit()
        self._draggingLower = False
        self._draggingUpper = False

    def valueToX(self, v):
        w = self.width()
        if self._maxValue == self._minValue:
            return 0
        ratio = (v - self._minValue) / (self._maxValue - self._minValue)
        return ratio * w

    def xToValue(self, x):
        w = self.width()
        if w == 0:
            return self._minValue
        ratio = x / w
        return self._minValue + ratio * (self._maxValue - self._minValue)


class FixedDurationTrimmer(QMainWindow):
    """
    A video trimmer UI where every video subclip is forced
    to have the same duration as the shortest video in the dataset.

    * We remember the average left/right fraction from saved subclips
      and use it to position the subclip for the next video.
    * We create a TEMPORARY subclip with ffmpeg to PREVIEW exactly the final
      trimmed segment (no partial frames beyond).
    * On 'Save Trim', we overwrite the original file with the same subclip range.
    """

    def __init__(self, video_paths, shortest_duration, shortest_path):
        super().__init__()
        self.video_paths = video_paths
        self.index = 0

        # This is the fixed subclip length we want for all videos
        self.fixed_duration = shortest_duration
        self.shortest_path = shortest_path

        self.setWindowTitle("Fixed-Duration Video Trimmer + Actual Subclip Preview")

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)

        # Info label
        self.info_label = QLabel()
        self.main_layout.addWidget(self.info_label)

        # Video widget
        self.video_widget = QVideoWidget()
        self.main_layout.addWidget(self.video_widget)

        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Range slider
        self.range_slider = FixedWindowRangeSlider()
        self.main_layout.addWidget(self.range_slider)
        self.range_slider.editingFinished.connect(self.on_subclip_changed)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play (Re-Encode Preview)")
        self.btn_save = QPushButton("Save Trim")
        self.btn_reject = QPushButton("Reject")
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_reject)
        self.main_layout.addLayout(btn_layout)

        # Connect signals
        self.btn_play.clicked.connect(self.preview_subclip)
        self.btn_save.clicked.connect(self.save_trim)
        self.btn_reject.clicked.connect(self.reject_video)

        # We'll store a path to the last temp file used for preview, so we can delete it
        self._last_temp_preview_path = None

        # Track average left/right fractions
        self.sum_left_fraction = 0.0
        self.sum_right_fraction = 0.0
        self.count_saved = 0

        # Load first video
        if self.video_paths:
            self.load_video(self.video_paths[self.index])
        else:
            QMessageBox.information(self, "No Videos", "No .mp4 files found.")
            self.close()

        self.update_info_label()

    def update_info_label(self):
        total = len(self.video_paths)
        current = self.index + 1
        processed_text = f"Processed: {current - 1} / {total}"
        shortest_text = f"All subclips = {round(self.fixed_duration, 2)}s | Shortest video was @ {self.shortest_path}"

        if self.index < total:
            current_file_path = self.video_paths[self.index]
            current_text = f"Current ({current}/{total}): {current_file_path}"
        else:
            current_text = "No more videos."

        self.info_label.setText(f"{processed_text}\n{shortest_text}\n{current_text}")

    # -----------------------------------------------------------------------
    # Load the next video
    # -----------------------------------------------------------------------
    def load_video(self, path):
        """
        Load the next video. We do NOT set the player directly to 'path',
        because we'll always preview from a subclip. But we do read the
        length, set the RangeSlider, etc.
        """
        # Clear any old preview
        self._remove_temp_preview()

        video_length = get_video_duration(path)
        self.range_slider.setRange(0, video_length)
        self.range_slider.setFixedWindowSize(self.fixed_duration)

        if video_length <= 0:
            print(f"Warning: {path} has zero or negative length.")
        elif video_length < self.fixed_duration:
            print(f"Warning: {path} is shorter than the fixed duration. The slider clamps to entire video.")

        # If we have an average alignment, apply it
        if self.count_saved > 0 and video_length > 0:
            self._apply_average_alignment(video_length)

        self.update_info_label()

        # Immediately generate a preview subclip
        self.preview_subclip()

    def _apply_average_alignment(self, video_length):
        avgLeft = self.sum_left_fraction / self.count_saved
        avgRight = self.sum_right_fraction / self.count_saved
        sumAVG = avgLeft + avgRight
        target_gap = 1.0 - (self.fixed_duration / video_length)

        if sumAVG > 0 and target_gap > 0:
            newLeftFrac = (avgLeft / sumAVG) * target_gap
            newRightFrac = (avgRight / sumAVG) * target_gap

            start = newLeftFrac * video_length
            end = video_length - (newRightFrac * video_length)

            if end - start < self.fixed_duration:
                diff = self.fixed_duration - (end - start)
                start = max(0, start - diff / 2)
                end = start + self.fixed_duration
                if end > video_length:
                    end = video_length
                    start = end - self.fixed_duration
            elif end - start > self.fixed_duration:
                extra = (end - start) - self.fixed_duration
                end -= extra / 2
                if end < start:
                    end = start

            if start < 0:
                start = 0
            if start + self.fixed_duration > video_length:
                start = video_length - self.fixed_duration

            self.range_slider._lowerValue = start
            self.range_slider._upperValue = start + self.fixed_duration
            self.range_slider.update()

    # -----------------------------------------------------------------------
    # TEMP SUBCLIP PREVIEW
    # -----------------------------------------------------------------------
    def on_subclip_changed(self):
        """When the user finishes dragging, auto-generate the preview subclip."""
        self.preview_subclip()

    def preview_subclip(self):
        """Re-encode a subclip [lower, upper] to a TEMP file, set player to that file, and play it."""
        if self.index >= len(self.video_paths):
            return

        in_path = self.video_paths[self.index]
        vid_len = get_video_duration(in_path)

        lowerVal, upperVal = self.range_slider.values()
        if upperVal <= lowerVal or upperVal > vid_len:
            print("Invalid subclip range for preview. Skipping.")
            return

        # Remove old preview if any
        self._remove_temp_preview()

        # Re-encode subclip to a new temp file
        out_temp = self._make_temp_subclip(in_path, lowerVal, upperVal)
        if out_temp:
            # Set QMediaPlayer to this temp file
            self._last_temp_preview_path = out_temp
            self.player.setSource(QUrl.fromLocalFile(out_temp))
            self.player.setPosition(0)
            self.player.play()

    def _make_temp_subclip(self, in_path, start, end):
        """ffmpeg re-encode subclip [start, end] => temp path. Returns the path or None on error."""
        try:
            tmp_file = os.path.join(tempfile.gettempdir(), f"subclip_preview_{uuid.uuid4().hex}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", in_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264",
                "-c:a", "aac",
                tmp_file
            ]
            print("Preview subclip:", " ".join(cmd))
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not os.path.exists(tmp_file):
                print("Preview subclip creation failed.")
                return None
            return tmp_file
        except Exception as e:
            print("Error creating subclip:", e)
            return None

    def _remove_temp_preview(self):
        """Delete the old preview file from disk, if any, to avoid junk accumulation."""
        if self._last_temp_preview_path and os.path.exists(self._last_temp_preview_path):
            try:
                os.remove(self._last_temp_preview_path)
                print(f"Removed old preview: {self._last_temp_preview_path}")
            except OSError as e:
                print(f"Error removing old preview: {e}")
        self._last_temp_preview_path = None

    # -----------------------------------------------------------------------
    # SAVE & REJECT
    # -----------------------------------------------------------------------
    def save_trim(self):
        """Trim the current video to the selected subclip, overwriting the original."""
        if self.index >= len(self.video_paths):
            return

        in_path = self.video_paths[self.index]
        vid_len = get_video_duration(in_path)
        lowerVal, upperVal = self.range_slider.values()

        if upperVal <= lowerVal:
            QMessageBox.warning(self, "Invalid Range", "End time must be greater than start time.")
            return
        if upperVal > vid_len:
            QMessageBox.warning(self, "Invalid Range", f"End time {upperVal:.2f} > video length {vid_len:.2f}.")
            return

        actualDur = upperVal - lowerVal
        if abs(actualDur - self.fixed_duration) > 0.01:
            QMessageBox.warning(self, "Wrong Duration",
                                f"Selected subclip length {actualDur:.2f} != required {self.fixed_duration:.2f}.")
            return

        # Update average alignment
        leftFraction = lowerVal / vid_len
        rightFraction = (vid_len - upperVal) / vid_len
        self.sum_left_fraction += leftFraction
        self.sum_right_fraction += rightFraction
        self.count_saved += 1

        # Re-encode subclip to a temp path
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-ss", str(lowerVal),
            "-to", str(upperVal),
            "-c:v", "libx264",
            "-c:a", "aac",
            tmp_path
        ]
        print("Saving subclip:", " ".join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Overwrite original
        shutil.move(tmp_path, in_path)
        print(f"Trimmed video saved (length={actualDur:.2f}s): {in_path}")

        # Next video
        self._remove_temp_preview()
        self.index += 1
        if self.index < len(self.video_paths):
            self.load_video(self.video_paths[self.index])
        else:
            QMessageBox.information(self, "Done", "All videos processed!")
            self.close()

    def reject_video(self):
        """Delete the current file and move on. (No alignment is recorded if we didn't save.)"""
        if self.index >= len(self.video_paths):
            return
        in_path = self.video_paths[self.index]
        try:
            os.remove(in_path)
            print(f"Deleted {in_path}")
        except OSError as e:
            print(f"Error deleting {in_path}: {e}")

        self._remove_temp_preview()
        self.index += 1
        if self.index < len(self.video_paths):
            self.load_video(self.video_paths[self.index])
        else:
            QMessageBox.information(self, "Done", "All videos processed (some rejected)!")
            self.close()


def main():
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = QFileDialog.getExistingDirectory(None, "Select Folder Containing MP4 Videos")
        if not folder:
            print("No folder selected. Exiting.")
            sys.exit(0)

    video_paths = find_mp4_files(folder)
    if not video_paths:
        print("No .mp4 files found in", folder)
        sys.exit(0)

    durations = [get_video_duration(vp) for vp in video_paths]
    min_duration = min(durations)
    i_min = durations.index(min_duration)
    shortest_path = video_paths[i_min]

    window = FixedDurationTrimmer(video_paths, min_duration, shortest_path)
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
