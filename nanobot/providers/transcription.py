"""Voice transcription providers: Groq (cloud, fast) and local Whisper (CPU fallback)."""

import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class LocalWhisperProvider:
    """Transcribe audio using the locally-installed openai-whisper package (CPU, free).

    Uses the 'tiny' model by default — fast enough for voice notes (~1-3s on CPU
    once the model is loaded). Falls back gracefully if whisper is not installed.
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None  # lazy-loaded on first call

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    async def transcribe(self, file_path: str | Path) -> str:
        import asyncio
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found for whisper: {}", file_path)
            return ""
        try:
            loop = asyncio.get_event_loop()
            # Run blocking whisper in a thread so we don't freeze the event loop
            result = await loop.run_in_executor(None, self._transcribe_sync, str(path))
            return result
        except ImportError:
            logger.warning("openai-whisper not installed — voice transcription unavailable")
            return ""
        except Exception as e:
            logger.error("Local whisper error: {}", e)
            return ""

    def _transcribe_sync(self, path: str) -> str:
        model = self._load_model()
        result = model.transcribe(path, fp16=False)  # fp16=False required for CPU
        return (result.get("text") or "").strip()


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.
    
    Groq offers extremely fast transcription with a generous free tier.
    """
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""
        
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        
        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }
                    
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")
                    
        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""
