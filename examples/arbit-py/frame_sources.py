from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from slam import CameraMatrix, from_intrinsics


@dataclass(slots=True)
class FrameData:
    """Container describing a single frame emitted by a frame source."""

    index: int
    gray: np.ndarray
    color: np.ndarray
    path: Optional[Path] = None
    timestamp: Optional[float] = None


class FrameSource(Protocol):
    """Protocol implemented by iterable frame sources."""

    def __iter__(self) -> Iterator[FrameData]:
        ...


class VideoFrameSource:
    """Frame source that streams frames from a video file."""

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)

    def __iter__(self) -> Iterator[FrameData]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {self.video_path}")

        index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else None
                yield FrameData(
                    index=index,
                    gray=gray,
                    color=frame,
                    path=self.video_path,
                    timestamp=timestamp,
                )
                index += 1
        finally:
            cap.release()


class ImageSequenceFrameSource:
    """Frame source that iterates over a directory containing image files."""

    def __init__(
        self,
        frame_paths: Sequence[Path],
        timestamps: Optional[Sequence[float]] = None,
    ):
        if not frame_paths:
            raise ValueError("No image frames were found for the provided directory")
        self.frame_paths = [Path(p) for p in frame_paths]
        self.timestamps = list(timestamps) if timestamps is not None else None

    def __iter__(self) -> Iterator[FrameData]:
        for index, frame_path in enumerate(self.frame_paths):
            color = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if color is None:
                raise ValueError(f"Unable to read image frame: {frame_path}")
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            timestamp = (
                self.timestamps[index]
                if self.timestamps is not None and index < len(self.timestamps)
                else None
            )
            yield FrameData(
                index=index,
                gray=gray,
                color=color,
                path=frame_path,
                timestamp=timestamp,
            )


def _list_image_paths(
    directory: Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[Path]:
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted(paths)


def _read_timestamps(file_path: Path) -> list[float]:
    if not file_path.exists():
        return []
    timestamps: list[float] = []
    with file_path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                timestamps.append(float(stripped))
            except ValueError:
                continue
    return timestamps


def _normalize_camera_key(camera_subdir: str) -> str:
    camera_subdir = camera_subdir.strip()
    camera_to_projection = {
        "image_0": "P0",
        "image_1": "P1",
        "image_2": "P2",
        "image_3": "P3",
    }
    if camera_subdir in camera_to_projection:
        return camera_to_projection[camera_subdir]
    if camera_subdir.upper().startswith("P"):
        return camera_subdir.upper()
    raise ValueError(
        f"Unknown camera identifier '{camera_subdir}'. "
        "Expected one of image_[0-3] or P[0-3]."
    )


def _read_kitti_camera_matrix(calib_path: Path, projection_key: str) -> CameraMatrix:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    projection_matrix: Optional[np.ndarray] = None
    with calib_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            if key.strip() != projection_key:
                continue
            parts = [float(val) for val in values.strip().split()]
            if len(parts) != 12:
                raise ValueError(
                    f"Expected 12 values for {projection_key} in {calib_path}, "
                    f"found {len(parts)}"
                )
            projection_matrix = np.array(parts, dtype=float).reshape(3, 4)
            break
    if projection_matrix is None:
        raise ValueError(
            f"Projection matrix '{projection_key}' not found in {calib_path}"
        )
    intrinsics = projection_matrix[:, :3]
    return from_intrinsics(intrinsics)


def _infer_kitti_sequence_root(path: Path) -> Optional[Path]:
    if (path / "calib.txt").exists():
        return path
    if path.name.startswith("image_") and (path.parent / "calib.txt").exists():
        return path.parent
    return None


FrameSourceType = Literal["auto", "video", "image_dir", "kitti"]


def create_frame_source(
    input_path: Path,
    source_type: FrameSourceType = "auto",
    camera_subdir: str = "image_2",
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> tuple[FrameSource, Optional[CameraMatrix], str]:
    """Create an appropriate frame source and optionally infer intrinsics."""

    resolved_path = Path(input_path)
    inferred_type: FrameSourceType
    if source_type == "auto":
        if resolved_path.is_file():
            inferred_type = "video"
        else:
            seq_root = _infer_kitti_sequence_root(resolved_path)
            inferred_type = "kitti" if seq_root else "image_dir"
    else:
        inferred_type = source_type

    if inferred_type == "video":
      source = VideoFrameSource(resolved_path)
      setattr(source, "label", resolved_path.name)
      return source, None, resolved_path.name

    if inferred_type == "image_dir":
        target_dir = resolved_path
        if not target_dir.is_dir():
            raise ValueError(f"Image directory not found: {resolved_path}")
        frame_paths = _list_image_paths(target_dir, image_extensions)
        frame_source = ImageSequenceFrameSource(frame_paths)
        setattr(frame_source, "label", target_dir.name)
        return frame_source, None, target_dir.name

    if inferred_type == "kitti":
        sequence_root = _infer_kitti_sequence_root(resolved_path)
        if sequence_root is None:
            raise ValueError(
                f"Unable to locate KITTI-style sequence root for {resolved_path}"
            )
        projection_key = _normalize_camera_key(camera_subdir)
        if resolved_path.name.startswith("image_"):
            camera_dir = resolved_path
        else:
            camera_dir = sequence_root / camera_subdir
        if not camera_dir.is_dir():
            raise ValueError(f"Camera directory not found: {camera_dir}")
        frame_paths = _list_image_paths(camera_dir, image_extensions)
        timestamps = _read_timestamps(sequence_root / "times.txt")
        frame_source = ImageSequenceFrameSource(frame_paths, timestamps=timestamps)
        setattr(frame_source, "label", f"{sequence_root.name}:{camera_dir.name}")
        camera_matrix = _read_kitti_camera_matrix(
            sequence_root / "calib.txt", projection_key
        )
        label = f"{sequence_root.name}:{camera_dir.name}"
        return frame_source, camera_matrix, label

    raise ValueError(f"Unsupported frame source type: {inferred_type}")


__all__ = [
    "FrameData",
    "FrameSource",
    "FrameSourceType",
    "VideoFrameSource",
    "ImageSequenceFrameSource",
    "create_frame_source",
]

