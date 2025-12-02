import os
import io
import gc
import json
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock, Thread
from typing import Optional, Union

import torch
import uvicorn
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import omegaconf
import typing
import collections
import pyannote.audio

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])
torch.serialization.add_safe_globals([typing.Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
torch.serialization.add_safe_globals([pyannote.audio.core.model.Introspection])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Resolution])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Whisper ASR Service",
    description="Speech-to-text transcription service using Whisper model",
    version="2.0.0"
)

# Configuration
ASR_ENGINE = os.getenv("ASR_ENGINE", "whisperx")  # "transformers" or "whisperx"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3") # antony66/whisper-large-v3-russian
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "auto")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "0"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./data")  # Directory for model cache
DEBUG_LOG_DIR = os.getenv("DEBUG_LOG_DIR", None)  # Directory for debug logging (None = disabled), # samples
SAMPLE_RATE = 16000

# Set cache directories if specified
if MODEL_CACHE_DIR:
    os.environ["HF_HOME"] = MODEL_CACHE_DIR
    os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")

# Create debug log directory if enabled
if DEBUG_LOG_DIR:
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
    logger.info(f"Debug logging enabled, saving to: {DEBUG_LOG_DIR}")


def save_debug_log(audio_data: np.ndarray, result: Union["TranscriptionResponse", str], filename: str):
    """Save audio and transcription result for debugging."""
    if not DEBUG_LOG_DIR:
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(filename or "audio")[0]
        prefix = f"{timestamp}_{base_name}"

        # Save audio as WAV
        audio_path = os.path.join(DEBUG_LOG_DIR, f"{prefix}.wav")
        sf.write(audio_path, audio_data, SAMPLE_RATE)

        # Save transcription result as JSON
        result_path = os.path.join(DEBUG_LOG_DIR, f"{prefix}.json")
        if isinstance(result, str):
            result_data = {"text": result}
        else:
            result_data = result.model_dump(exclude_none=True)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Debug log saved: {prefix}")
    except Exception as e:
        logger.warning(f"Failed to save debug log: {e}")

# Pydantic models
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[list[WordTimestamp]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: Optional[list[Segment]] = None


def get_device():
    """Determine the best available device for inference."""
    if DEVICE != "auto":
        return torch.device(DEVICE)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        setattr(torch.distributed, "is_initialized", lambda: False)  # monkey patching for MPS
    return torch.device(device)


def load_audio_from_file(audio_content: bytes) -> np.ndarray:
    """Load audio from bytes and convert to 16kHz mono numpy array."""
    audio_buffer = io.BytesIO(audio_content)

    try:
        # Try soundfile first (faster, supports wav/flac/ogg)
        audio_data, sample_rate = sf.read(audio_buffer)
    except Exception:
        # Fallback to librosa (supports more formats)
        audio_buffer.seek(0)
        audio_data, sample_rate = librosa.load(audio_buffer, sr=None, mono=True)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if needed (Whisper requirement)
    if sample_rate != SAMPLE_RATE:
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    # Ensure float32
    return audio_data.astype(np.float32)


class ASRModel(ABC):
    """Abstract base class for ASR models."""

    model = None
    model_lock = Lock()
    last_activity_time = time.time()

    @abstractmethod
    def load_model(self):
        """Loads the model."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        """Perform transcription on the given audio."""
        pass

    def monitor_idleness(self):
        """Monitors idleness and releases model if idle too long."""
        if MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            if time.time() - self.last_activity_time > MODEL_IDLE_TIMEOUT:
                with self.model_lock:
                    self.release_model()
                    break

    def release_model(self):
        """Unloads the model from memory."""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        logger.info("Model unloaded due to timeout")


class TransformersASR(ASRModel):
    """ASR using Hugging Face Transformers pipeline."""

    def __init__(self):
        self.torch_dtype = torch.bfloat16
        self.pipeline = None

    def load_model(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

        device = get_device()
        logger.info(f"Loading Transformers Whisper model on device: {device}")

        cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None

        whisper = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir,
        )

        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, cache_dir=cache_dir)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=device,
        )

        self.model = self.pipeline
        logger.info("Transformers Whisper model loaded successfully")

        if MODEL_IDLE_TIMEOUT > 0:
            Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.pipeline is None:
                self.load_model()

        generate_kwargs = {"max_new_tokens": 256}
        if language:
            generate_kwargs["language"] = language
        if task == "translate":
            generate_kwargs["task"] = "translate"

        return_ts = word_timestamps and output == "json"

        result = self.pipeline(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if return_ts else True,
        )

        logger.info(f"Transcription completed: {len(result.get('text', ''))} characters")

        return self._format_result(result, language, output, return_ts)

    def _format_result(self, result: dict, language: Optional[str], output: str, return_timestamps: bool):
        """Format the transcription result."""
        if output == "text":
            return result.get("text", "")

        response = TranscriptionResponse(
            text=result.get("text", "").strip(),
            language=language,
        )

        if "chunks" in result and result["chunks"]:
            segments = []
            all_words = []

            for idx, chunk in enumerate(result["chunks"]):
                timestamp = chunk.get("timestamp", (0, 0))
                start_time = timestamp[0] if timestamp[0] is not None else 0
                end_time = timestamp[1] if timestamp[1] is not None else start_time

                segment = Segment(
                    id=idx,
                    start=start_time,
                    end=end_time,
                    text=chunk.get("text", "").strip(),
                )

                if return_timestamps:
                    word = WordTimestamp(
                        word=chunk.get("text", "").strip(),
                        start=start_time,
                        end=end_time,
                    )
                    all_words.append(word)

                segments.append(segment)

            if return_timestamps and all_words:
                if segments:
                    first_start = segments[0].start
                    last_end = segments[-1].end
                    response.segments = [
                        Segment(
                            id=0,
                            start=first_start,
                            end=last_end,
                            text=response.text,
                            words=all_words,
                        )
                    ]
            else:
                response.segments = segments

        return response


class WhisperXASR(ASRModel):
    """ASR using WhisperX with alignment."""

    def __init__(self):
        self.model = {
            'whisperx': None,
            'align_model': {}
        }

    def load_model(self):
        import whisperx

        device = get_device()
        device_str = str(device) if device != torch.device('mps') else 'cpu'

        logger.info(f"Loading WhisperX model on device: {device_str}")

        asr_options = {"without_timestamps": False}
        download_root = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None
        self.model['whisperx'] = whisperx.load_model(
            WHISPERX_MODEL,
            device=device_str,
            compute_type=COMPUTE_TYPE,
            asr_options=asr_options,
            download_root=download_root
        )

        logger.info("WhisperX model loaded successfully")

        if MODEL_IDLE_TIMEOUT > 0:
            Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        import whisperx

        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model['whisperx'] is None:
                self.load_model()

        device = get_device()
        device_str = str(device) if device != torch.device('mps') else 'cpu'

        # Transcribe
        transcribe_options = {"task": task}
        if language:
            transcribe_options["language"] = language

        with self.model_lock:
            result = self.model['whisperx'].transcribe(audio, **transcribe_options)
            detected_language = result.get("language", language)

        # Alignment
        try:
            if detected_language in self.model['align_model']:
                model_x, metadata = self.model['align_model'][detected_language]
            else:
                self.model['align_model'][detected_language] = whisperx.load_align_model(
                    language_code=detected_language, device=device_str
                )
                model_x, metadata = self.model['align_model'][detected_language]

            result = whisperx.align(
                result["segments"], model_x, metadata, audio, device_str, return_char_alignments=False
            )
        except Exception as e:
            logger.warning(f"Alignment failed: {e}")

        result["language"] = detected_language

        logger.info(f"WhisperX transcription completed: {len(result.get('segments', []))} segments")

        return self._format_result(result, output)

    def _format_result(self, result: dict, output: str) -> Union[TranscriptionResponse, str]:
        """Format WhisperX result."""
        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)

        if output == "text":
            return full_text

        if output == "srt":
            return self._to_srt(segments)

        if output == "vtt":
            return self._to_vtt(segments)

        if output == "tsv":
            return self._to_tsv(segments)

        # JSON output
        response_segments = []
        for idx, seg in enumerate(segments):
            words = None
            if "words" in seg:
                words = [
                    WordTimestamp(
                        word=w.get("word", ""),
                        start=w.get("start", 0),
                        end=w.get("end", 0),
                        probability=w.get("score"),
                    )
                    for w in seg["words"]
                ]

            response_segments.append(Segment(
                id=idx,
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
                words=words,
            ))

        return TranscriptionResponse(
            text=full_text,
            language=result.get("language"),
            segments=response_segments,
        )

    def _to_srt(self, segments: list) -> str:
        lines = []
        for idx, seg in enumerate(segments):
            start = self._format_timestamp_srt(seg.get("start", 0))
            end = self._format_timestamp_srt(seg.get("end", 0))
            lines.append(str(idx + 1))
            lines.append(f"{start} --> {end}")
            lines.append(seg.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    def _to_vtt(self, segments: list) -> str:
        lines = ["WEBVTT", ""]
        for seg in segments:
            start = self._format_timestamp_vtt(seg.get("start", 0))
            end = self._format_timestamp_vtt(seg.get("end", 0))
            lines.append(f"{start} --> {end}")
            lines.append(seg.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    def _to_tsv(self, segments: list) -> str:
        lines = ["start\tend\ttext"]
        for seg in segments:
            start = int(seg.get("start", 0) * 1000)
            end = int(seg.get("end", 0) * 1000)
            text = seg.get("text", "").strip().replace("\t", " ")
            lines.append(f"{start}\t{end}\t{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# Global ASR model instance
asr_model: Optional[ASRModel] = None


def create_asr_model() -> ASRModel:
    """Factory function to create the appropriate ASR model."""
    if ASR_ENGINE == "whisperx":
        return WhisperXASR()
    elif ASR_ENGINE == "transformers":
        return TransformersASR()
    else:
        raise ValueError(f"Unsupported ASR engine: {ASR_ENGINE}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global asr_model
    logger.info(f"Initializing ASR with engine: {ASR_ENGINE}")
    asr_model = create_asr_model()
    asr_model.load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": asr_model is not None,
        "engine": ASR_ENGINE,
    }


@app.post("/asr")
async def transcribe(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    output: str = Query("json", description="Output format: text, json, vtt, srt, tsv"),
    task: str = Query("transcribe", description="Task: transcribe or translate"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'russian', 'en')"),
    word_timestamps: bool = Query(True, description="Include word-level timestamps"),
    encode: bool = Query(False, description="Whether audio needs encoding (ignored, handled automatically)"),
):
    """
    Transcribe audio file using Whisper model.

    Supports two engines:
    - transformers: Hugging Face Transformers pipeline (default)
    - whisperx: WhisperX with word-level alignment

    Set ASR_ENGINE environment variable to choose the engine.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read audio file
        audio_content = await audio_file.read()
        logger.info(f"Received audio file: {audio_file.filename}, size: {len(audio_content)} bytes")

        # Convert audio to numpy array
        audio_data = load_audio_from_file(audio_content)
        logger.info(f"Audio converted: {len(audio_data)} samples at {SAMPLE_RATE}Hz")

        # Run transcription
        result = asr_model.transcribe(
            audio=audio_data,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
        )

        # Save debug log if enabled
        save_debug_log(audio_data, result, audio_file.filename)

        # Format response based on output type
        if output == "text":
            if isinstance(result, str):
                return PlainTextResponse(content=result)
            return PlainTextResponse(content=result.text)

        elif output == "json":
            if isinstance(result, TranscriptionResponse):
                return JSONResponse(
                    content=result.model_dump(exclude_none=True),
                    headers={"Asr-Engine": ASR_ENGINE},
                )
            return JSONResponse(content={"text": str(result)}, headers={"Asr-Engine": ASR_ENGINE})

        elif output == "vtt":
            if isinstance(result, str):
                return PlainTextResponse(content=result, media_type="text/vtt")
            return PlainTextResponse(content=format_vtt(result), media_type="text/vtt")

        elif output == "srt":
            if isinstance(result, str):
                return PlainTextResponse(content=result, media_type="text/plain")
            return PlainTextResponse(content=format_srt(result), media_type="text/plain")

        elif output == "tsv":
            if isinstance(result, str):
                return PlainTextResponse(content=result, media_type="text/tab-separated-values")
            return PlainTextResponse(content=format_tsv(result), media_type="text/tab-separated-values")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output}")

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def format_vtt(response: TranscriptionResponse) -> str:
    """Format TranscriptionResponse to VTT."""
    lines = ["WEBVTT", ""]
    if response.segments:
        for seg in response.segments:
            start = format_timestamp_vtt(seg.start)
            end = format_timestamp_vtt(seg.end)
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
    return "\n".join(lines)


def format_srt(response: TranscriptionResponse) -> str:
    """Format TranscriptionResponse to SRT."""
    lines = []
    if response.segments:
        for idx, seg in enumerate(response.segments):
            start = format_timestamp_srt(seg.start)
            end = format_timestamp_srt(seg.end)
            lines.append(str(idx + 1))
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
    return "\n".join(lines)


def format_tsv(response: TranscriptionResponse) -> str:
    """Format TranscriptionResponse to TSV."""
    lines = ["start\tend\ttext"]
    if response.segments:
        for seg in response.segments:
            start = int(seg.start * 1000)
            end = int(seg.end * 1000)
            text = seg.text.replace("\t", " ")
            lines.append(f"{start}\t{end}\t{text}")
    return "\n".join(lines)


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9007"))

    logger.info(f"Starting Whisper ASR service on {host}:{port}")
    logger.info(f"ASR Engine: {ASR_ENGINE}")
    uvicorn.run(app, host=host, port=port)
