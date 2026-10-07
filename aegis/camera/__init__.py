from __future__ import annotations

from .camera_source import CameraSource
from .frame import Frame


def open_source(spec, camera_id: str = "CAM-001") -> CameraSource:
    """'0' / 0 -> webcam, 'rtsp://...' -> RTSP, 'mock' -> synthetic, anything else -> video file."""
    s = str(spec)
    if s.isdigit():
        from .webcam_source import WebcamSource
        return WebcamSource(int(s), camera_id)
    if s.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        from .rtsp_source import RTSPSource
        return RTSPSource(s, camera_id)
    if s.lower() == "mock":
        from .mock_source import MockSource
        return MockSource(camera_id)
    from .video_file_source import VideoFileSource
    return VideoFileSource(s, camera_id)


__all__ = ["CameraSource", "Frame", "open_source"]
