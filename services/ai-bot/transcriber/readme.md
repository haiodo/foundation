# Whisper ASR Transcription Service

Speech-to-text transcription service supporting multiple ASR engines.

## Supported Engines

- **transformers** (default): Hugging Face Transformers pipeline with Whisper models
- **whisperx**: WhisperX with word-level alignment

## Installation

```bash
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Configure the service using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_ENGINE` | `transformers` | ASR engine to use: `transformers` or `whisperx` |
| `WHISPER_MODEL` | `antony66/whisper-large-v3-russian` | Model name for transformers engine |
| `WHISPERX_MODEL` | `large-v3` | Model name for WhisperX engine |
| `DEVICE` | `auto` | Device to use: `auto`, `cuda`, `cpu`, `mps` |
| `COMPUTE_TYPE` | `float16` | Compute type for WhisperX: `float16`, `int8`, etc. |
| `MODEL_CACHE_DIR` | `./data` | Directory for caching downloaded models |
| `DEBUG_LOG_DIR` | (disabled) | Directory for saving audio chunks and results for debugging |
| `MODEL_IDLE_TIMEOUT` | `0` | Seconds before unloading idle model (0 = never) |
| `HOST` | `0.0.0.0` | Host to bind the server |
| `PORT` | `9007` | Port to bind the server |

## Running

```bash
# Using transformers (default)
python main.py

# Using WhisperX
ASR_ENGINE=whisperx python main.py

# With custom model cache directory
MODEL_CACHE_DIR=/path/to/cache ASR_ENGINE=whisperx python main.py

# With debug logging enabled (saves audio and results)
DEBUG_LOG_DIR=./debug_logs python main.py
```

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "engine": "whisperx"
}
```

### `POST /asr`

Transcribe an audio file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | file | required | Audio file to transcribe |
| `output` | string | `json` | Output format: `text`, `json`, `vtt`, `srt`, `tsv` |
| `task` | string | `transcribe` | Task: `transcribe` or `translate` |
| `language` | string | null | Language code (e.g., `russian`, `en`) |
| `word_timestamps` | bool | `true` | Include word-level timestamps |

**Example:**

```bash
# Basic transcription
curl -X POST "http://localhost:9007/asr" \
  -F "audio_file=@audio.wav" \
  -F "output=json"
```

## WhisperX Features

When using the `whisperx` engine, you get:

- **Word-level alignment**: Accurate word timestamps using phoneme-based alignment
- **Multi-language support**: Automatic language detection and alignment

## Debug Logging

When `DEBUG_LOG_DIR` is set, the service saves each transcription request for debugging:

- `{timestamp}_{filename}.wav` - The audio file (16kHz mono)
- `{timestamp}_{filename}.json` - The transcription result

This is useful for debugging transcription issues or building test datasets.

## Development

Update dependencies:

```bash
pip install pip-tools
pip-compile --upgrade requirements.in -o requirements.txt
```
