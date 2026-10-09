"""Provider clients.

Each one answers the same three questions: are we configured, can you do the job,
and what do I fall back to when you cannot. Stages never import `httpx` directly.
"""

from .images import ImageProvider
from .llm import ChatLLM, LLMError
from .tts import SpeechProvider
from .youtube import YouTubeUploader

__all__ = ["ChatLLM", "LLMError", "SpeechProvider", "ImageProvider", "YouTubeUploader"]
