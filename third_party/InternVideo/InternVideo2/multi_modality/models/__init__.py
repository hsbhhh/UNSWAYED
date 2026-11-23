# Minimal imports for video-only feature extraction.
# Avoid importing audio/text/stage2 modules that require torchaudio/deepspeed/etc.

from .internvideo2_clip import InternVideo2_CLIP

__all__ = ["InternVideo2_CLIP"]
