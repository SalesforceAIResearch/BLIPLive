"""Reference frame decoding and timestamp conversion."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def kept_frame_indices(num_frames: int, stride: int) -> list[int]:
    """Use the training sampler's end-aligned stride."""
    if stride < 1:
        raise ValueError("stride must be at least one")
    return [index for index in range(num_frames) if (num_frames - 1 - index) % stride == 0]


def source_frame_times(
    source_indices: list[int],
    *,
    origin: int,
    source_fps: float,
) -> list[float]:
    """Convert absolute source-frame indices to seconds relative to ``origin``."""
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    return [(source_index - origin) / source_fps for source_index in source_indices]


def uniform_subsample(items: list, limit: int | None) -> list:
    if not limit or len(items) <= limit:
        return items
    indices = sorted({round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)})
    return [items[index] for index in indices]


def video_metadata(path: str | Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if count <= 0 or fps <= 0:
        raise RuntimeError(f"bad video metadata for {path}: frames={count}, fps={fps}")
    return count, fps


def decode_indices(
    path: str | Path,
    start: int,
    stop: int,
    *,
    stride: int = 1,
    end_aligned: bool = True,
    frame_cap: int | None = None,
    max_frames: int | None = None,
    return_source_indices: bool = False,
) -> list[Image.Image] | tuple[list[Image.Image], list[int]]:
    """Decode the half-open source-frame interval and apply end-aligned sampling."""
    start = max(0, int(start))
    stop = max(start, int(stop))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Image.Image] = []
    source_indices: list[int] = []
    keep_offsets = set(
        kept_frame_indices(stop - start, stride) if end_aligned else range(0, stop - start, stride)
    )
    for offset, source_index in enumerate(range(start, stop)):
        ok, frame = capture.read()
        if not ok:
            break
        if offset in keep_offsets:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            source_indices.append(source_index)
    capture.release()
    if frame_cap:
        frames = frames[: int(frame_cap)]
        source_indices = source_indices[: int(frame_cap)]
    paired = uniform_subsample(list(zip(frames, source_indices, strict=True)), max_frames)
    frames = [frame for frame, _ in paired]
    source_indices = [source_index for _, source_index in paired]
    if return_source_indices:
        return frames, source_indices
    return frames


def decode_seconds(
    path: str | Path,
    *,
    start: float,
    end: float | None,
    source_fps: float,
    target_fps: int = 1,
    include_end: bool = False,
    frame_cap: int | None = None,
    max_frames: int | None = None,
    return_source_indices: bool = False,
):

    count, measured_fps = video_metadata(path)
    use_measured_fps = source_fps <= 0
    source_fps = measured_fps if use_measured_fps else source_fps
    first = max(0, int(start * source_fps))
    stop = count if end is None else min(count, int(end * source_fps) + (1 if include_end else 0))
    stop = max(first + 1, stop)
    stride = max(1, round(source_fps / target_fps))
    return decode_indices(
        path,
        first,
        stop,
        stride=stride,
        end_aligned=not use_measured_fps,
        frame_cap=frame_cap,
        max_frames=max_frames,
        return_source_indices=return_source_indices,
    )


def load_video(path, *, source_fps=2):
    _, measured_fps = video_metadata(path)
    images, indices = decode_seconds(
        path, start=0, end=None, source_fps=source_fps, return_source_indices=True
    )
    if not images:
        raise RuntimeError(f"No frames decoded: {path}")
    return images, source_frame_times(
        indices, origin=0, source_fps=source_fps if source_fps > 0 else measured_fps
    )
