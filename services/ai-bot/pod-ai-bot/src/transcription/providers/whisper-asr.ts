// Copyright © 2025 Andrey Sobolev (haiodo@gmail.com)

import { MeasureContext } from '@hcengineering/core'
import { TranscriptionOptions, TranscriptionProvider, TranscriptionResult, TranscriptionWord } from '../types'

/**
 * Response from Whisper ASR Webservice /asr endpoint with JSON output
 */
interface WhisperAsrResponse {
  text: string
  language?: string
  segments?: Array<{
    id: number
    start: number
    end: number
    text: string
    tokens?: number[]
    temperature?: number
    avg_logprob?: number
    compression_ratio?: number
    no_speech_prob?: number
    words?: Array<{
      word: string
      start: number
      end: number
      probability?: number
    }> | null
  }>
}

/**
 * Whisper ASR Webservice provider
 *
 * Uses the whisper-asr-webservice API (https://github.com/ahmetoner/whisper-asr-webservice)
 * Supports multiple engines: openai_whisper, faster_whisper, whisperx
 *
 * API endpoint: POST /asr
 * - audio_file: multipart form data with audio file
 * - output: text | json | vtt | srt | tsv
 * - task: transcribe | translate
 * - language: optional language code
 * - word_timestamps: enable word-level timestamps (faster_whisper only)
 */
export class WhisperAsrProvider implements TranscriptionProvider {
  readonly name = 'whisper-asr'

  constructor (
    private readonly ctx: MeasureContext,
    private readonly baseUrl: string
  ) {}

  async transcribe (audioData: Buffer, options?: TranscriptionOptions): Promise<TranscriptionResult> {
    const startTime = Date.now()

    try {
      // Build URL with query parameters
      const params = new URLSearchParams({
        output: 'json',
        task: 'transcribe',
        word_timestamps: options?.wordTimestamps !== false ? 'true' : 'false',
        encode: 'false' // Let whisper-asr handle audio conversion via FFmpeg
      })

      if (options?.language !== undefined && options.language !== '') {
        params.set('language', options.language)
      }

      const url = `${this.baseUrl}/asr?${params.toString()}`

      // Create multipart form data manually
      // Node.js doesn't have native FormData with Blob support, so we build it manually
      const boundary = `----WebKitFormBoundary${Date.now().toString(16)}`

      const formParts: Buffer[] = []

      // Add audio file part
      formParts.push(
        Buffer.from(
          `--${boundary}\r\n` +
          'Content-Disposition: form-data; name="audio_file"; filename="audio.wav"\r\n' +
          'Content-Type: audio/wav\r\n\r\n'
        )
      )
      formParts.push(audioData)
      formParts.push(Buffer.from('\r\n'))

      // Add closing boundary
      formParts.push(Buffer.from(`--${boundary}--\r\n`))

      const body = Buffer.concat(formParts)

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': body.length.toString()
        },
        body
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Whisper ASR request failed: ${response.status} ${response.statusText} - ${errorText}`)
      }

      const data = await response.json() as WhisperAsrResponse

      const result = this.parseResponse(data)

      const elapsed = Date.now() - startTime
      this.ctx.info('Whisper ASR transcription completed', {
        provider: this.name,
        language: result.language,
        textLength: result.text.length,
        wordCount: result.words?.length ?? 0,
        elapsedMs: elapsed
      })

      return result
    } catch (err: any) {
      const elapsed = Date.now() - startTime
      this.ctx.error('Whisper ASR transcription failed', {
        provider: this.name,
        error: err.message,
        elapsedMs: elapsed
      })
      throw err
    }
  }

  /**
   * Parse Whisper ASR response to standard format
   */
  private parseResponse (data: WhisperAsrResponse): TranscriptionResult {
    const result: TranscriptionResult = {
      text: data.text?.trim() ?? '',
      language: data.language
    }

    // Parse segments
    if (data.segments !== undefined && data.segments.length > 0) {
      result.segments = data.segments.map(seg => ({
        start: seg.start,
        end: seg.end,
        text: seg.text.trim()
      }))

      // Extract words from segments if available
      const words: TranscriptionWord[] = []
      for (const segment of data.segments) {
        if (segment.words != null) {
          for (const word of segment.words) {
            words.push({
              word: word.word.trim(),
              start: word.start,
              end: word.end,
              confidence: word.probability
            })
          }
        }
      }

      if (words.length > 0) {
        result.words = words
      }

      // Calculate average confidence from segment probabilities
      const probabilities = data.segments
        .filter(seg => seg.avg_logprob !== undefined)
        .map(seg => Math.exp(seg.avg_logprob ?? 1))

      if (probabilities.length > 0) {
        result.confidence = probabilities.reduce((a, b) => a + b, 0) / probabilities.length
      }
    }

    return result
  }
}

/**
 * Create Whisper ASR provider instance
 */
export function createWhisperAsrProvider (
  ctx: MeasureContext,
  baseUrl: string
): WhisperAsrProvider {
  return new WhisperAsrProvider(ctx, baseUrl)
}
