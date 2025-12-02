//
// Copyright © 2025 Andrey Sobolev (haiodo@gmail.com)
//

// A module to dump recordings into files with proper timestamps and user identification
// Uses smart VAD (Voice Activity Detection) to detect phrase boundaries
// Sends chunks immediately after detecting end of speech (silence)

import { AudioStream, RemoteParticipant, RemoteTrack, RemoteTrackPublication, Room } from '@livekit/rtc-node'
import { randomUUID } from 'crypto'

import { Stt } from '../type.js'
import config from '../config.js'
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, unlinkSync, writeSync } from 'fs'
import { join } from 'path'
import { gzipSync } from 'zlib'
import { spawn } from 'child_process'
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg'

/**
 * Sanitize a string for use in file paths - replace spaces and special characters
 */
function sanitizePath (name: string): string {
  return name.replace(/\s+/g, '_').replace(/[<>:"/\\|?*]/g, '_')
}

interface StreamTiming {
  streamStartTime: number // Absolute timestamp when stream started (ms)
  lastFrameEndTime: number // End time of the last frame (ms)
}

// Write buffer configuration
const WRITE_BUFFER_SIZE = 4096 // 4KB buffer - reduces syscalls

// Chunk timing configuration
const MAX_CHUNK_DURATION_MS = 30000 // Maximum 30 seconds per chunk (safety limit)
const MIN_CHUNK_DURATION_MS = 500 // Minimum 500ms to avoid tiny chunks
const SILENCE_THRESHOLD_MS = 1000 // 1 second of silence triggers chunk end
const SPEECH_START_THRESHOLD_MS = 100 // 100ms of speech to start a new chunk

interface ChunkState {
  fd: number | null // File descriptor for current chunk
  chunkStartTime: number // Start timestamp of current chunk (ms) - absolute
  chunkEndTime: number // End timestamp of current chunk (ms) - absolute
  chunkDataLength: number // Current data length in bytes (excluding header)
  chunkFilePath: string | null // Current chunk file path
  // VAD metrics
  totalSamples: number // Total samples in chunk
  activeSamples: number // Samples above VAD threshold
  peakAmplitude: number // Peak amplitude in chunk
  sumSquares: number // Sum of squares for RMS calculation
  // Write buffer
  writeBuffer: Buffer // Buffer for accumulating data before writing
  writeBufferOffset: number // Current position in write buffer
  // VAD state for smart chunking
  isSpeaking: boolean // Whether user is currently speaking
  speechStartTime: number // When current speech started (ms)
  silenceStartTime: number // When silence started (ms)
  lastSpeechEndTime: number // When last speech ended (ms)
  consecutiveSpeechMs: number // Consecutive milliseconds of speech
  consecutiveSilenceMs: number // Consecutive milliseconds of silence
  chunkIndex: number // Chunk index for this participant stream
}

// Full session state per participant
interface SessionState {
  fd: number | null // File descriptor for full session WAV
  filePath: string | null // Full session file path
  dataLength: number // Total data length in bytes
  writeBuffer: Buffer // Write buffer
  writeBufferOffset: number // Current position in write buffer
  startTimeFromMeeting: number // Start time in seconds from meeting start
  // Debug tracking
  frameCount: number // Number of frames written
  lastFrameHash: number // Simple hash of last frame to detect duplicates
}

interface ChunkMetadata {
  startTimeSec: number // Start time in seconds from meeting start
  endTimeSec: number // End time in seconds from meeting start
  durationSec: number // Duration in seconds
  participant: string // Participant identity (Ref<Person>) for sending transcription
  participantName: string // Participant display name for logging/files
  sampleRate: number
  channels: number
  bitsPerSample: number
  // VAD info (for debugging/logging only, not used for filtering)
  endReason: 'silence' | 'max_duration' | 'stream_end' // Why chunk was ended
  speechRatio: number // Ratio of speech to total duration (0-1) - informational only
  peakAmplitude: number // Peak amplitude (0-1 normalized) - informational only
  rmsAmplitude: number // RMS amplitude (0-1 normalized) - informational only
}

const VAD_THRESHOLD = 0.015 // RMS threshold for voice activity (normalized 0-1)
const VAD_FRAME_THRESHOLD = 0.01 // Per-frame threshold for counting active samples
const SPEECH_RATIO_THRESHOLD = 0.1 // At least 10% of samples should be active for speech

/**
 * Creates a WAV file header for PCM audio data
 */
function createWavHeader (dataLength: number, sampleRate: number, channels: number, bitsPerSample: number): Buffer {
  const byteRate = sampleRate * channels * (bitsPerSample / 8)
  const blockAlign = channels * (bitsPerSample / 8)
  const header = Buffer.alloc(44)

  // RIFF header
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + dataLength, 4) // File size - 8
  header.write('WAVE', 8)

  // fmt chunk
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16) // fmt chunk size
  header.writeUInt16LE(1, 20) // Audio format (1 = PCM)
  header.writeUInt16LE(channels, 22) // Number of channels
  header.writeUInt32LE(sampleRate, 24) // Sample rate
  header.writeUInt32LE(byteRate, 28) // Byte rate
  header.writeUInt16LE(blockAlign, 32) // Block align
  header.writeUInt16LE(bitsPerSample, 34) // Bits per sample

  // data chunk
  header.write('data', 36)
  header.writeUInt32LE(dataLength, 40) // Data size

  return header
}

/**
 * Updates WAV header with correct data length
 */
function updateWavHeader (fd: number, dataLength: number): void {
  const fileSizeBuffer = Buffer.alloc(4)
  const dataSizeBuffer = Buffer.alloc(4)

  // Update RIFF chunk size at offset 4
  fileSizeBuffer.writeUInt32LE(36 + dataLength, 0)
  writeSync(fd, fileSizeBuffer, 0, 4, 4)

  // Update data chunk size at offset 40
  dataSizeBuffer.writeUInt32LE(dataLength, 0)
  writeSync(fd, dataSizeBuffer, 0, 4, 40)
}

/**
 * Calculate RMS and detect voice activity for a buffer of 16-bit PCM samples
 */
function analyzeAudioBuffer (buf: Buffer): {
  rms: number
  peak: number
  activeSamples: number
  totalSamples: number
  sumSquares: number
} {
  const samples = buf.length / 2 // 16-bit samples
  let sumSquares = 0
  let peak = 0
  let activeSamples = 0

  for (let i = 0; i < buf.length; i += 2) {
    const sample = buf.readInt16LE(i)
    const normalized = Math.abs(sample) / 32768 // Normalize to 0-1
    sumSquares += normalized * normalized
    if (normalized > peak) {
      peak = normalized
    }
    if (normalized > VAD_FRAME_THRESHOLD) {
      activeSamples++
    }
  }

  const rms = Math.sqrt(sumSquares / samples)
  return { rms, peak, activeSamples, totalSamples: samples, sumSquares }
}

/**
 * Convert WAV file to MP3 using ffmpeg
 */
async function convertWavToMp3 (wavPath: string, mp3Path: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const ffmpegPath = ffmpegInstaller.path
    const args = [
      '-i',
      wavPath,
      '-codec:a',
      'libmp3lame',
      '-qscale:a',
      '2', // High quality VBR
      '-y', // Overwrite output
      mp3Path
    ]

    const proc = spawn(ffmpegPath, args)

    let stderr = ''
    proc.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`ffmpeg exited with code ${code}: ${stderr}`))
      }
    })

    proc.on('error', (err) => {
      reject(new Error(`Failed to start ffmpeg: ${err.message}`))
    })
  })
}

export class STT implements Stt {
  private isInProgress = false
  private language: string = 'en'

  private readonly trackBySid = new Map<string, RemoteTrack>()
  private readonly streamBySid = new Map<string, AudioStream>()
  private readonly participantBySid = new Map<string, RemoteParticipant>()
  private readonly stoppedSids = new Set<string>() // Track streams that should be terminated

  private readonly sessionBySid = new Map<string, any>()
  private readonly timingBySid = new Map<string, StreamTiming>()
  private readonly chunkStateBySid = new Map<string, ChunkState>()
  private readonly sessionStateBySid = new Map<string, SessionState>() // Full session recording

  private transcriptionCount = 0
  private sessionNumber = 0 // Incremented on each start() for session file naming
  private readonly meetingStartTime: number = Date.now() // Time when the meeting started (object creation)

  private readonly rootDir: string
  private readonly meetingId: string = randomUUID()
  private readonly sampleRate = 16000
  private readonly channels = 1
  private readonly bitsPerSample = 16

  constructor (
    readonly room: Room,
    readonly workspace: string,
    readonly token: string
  ) {
    this.rootDir = join('dumps', sanitizePath(this.workspace), this.meetingId)
  }

  updateLanguage (language: string): void {
    this.language = language
  }

  start (): void {
    if (this.isInProgress) return
    this.isInProgress = true
    this.sessionNumber++

    console.info('Start transcription', {
      workspace: this.workspace,
      room: this.room.name,
      rootDir: this.rootDir,
      sessionNumber: this.sessionNumber,
      meetingStartTime: this.meetingStartTime
    })
    if (!existsSync(this.rootDir)) {
      mkdirSync(this.rootDir, { recursive: true })
    }

    for (const sid of this.trackBySid.keys()) {
      this.processTrack(sid)
    }
  }

  stop (): void {
    if (!this.isInProgress) return
    console.info('Stopping transcription', { workspace: this.workspace, room: this.room.name })
    this.isInProgress = false
    for (const sid of this.trackBySid.keys()) {
      this.stopWs(sid)
    }
  }

  subscribe (track: RemoteTrack, publication: RemoteTrackPublication, participant: RemoteParticipant): void {
    console.info('subscribe', { kind: track.kind, sid: publication.sid, name: participant.name, identity: participant.identity })
    const sid = publication.sid
    if (sid === undefined) return
    if (this.trackBySid.has(sid)) return
    this.trackBySid.set(sid, track)
    this.participantBySid.set(sid, participant)
    if (this.isInProgress) {
      this.processTrack(sid)
    }
  }

  unsubscribe (
    track: RemoteTrack | undefined,
    publication: RemoteTrackPublication,
    participant: RemoteParticipant
  ): void {
    const sid = publication.sid
    if (sid === undefined) return

    console.info('unsubscribe', { kind: track?.kind ?? '', sid: publication.sid, name: participant.name, identity: participant.identity })
    this.trackBySid.delete(sid)
    this.participantBySid.delete(sid)
    this.stopWs(sid)
  }

  stopWs (sid: string): void {
    try {
      // Mark this sid as stopped so the streamToFiles loop will exit
      this.stoppedSids.add(sid)

      const stream = this.streamBySid.get(sid)
      if (stream !== undefined) {
        ;(stream as any).close?.()
      }

      // Finalize current chunk if exists
      this.finalizeChunk(sid)

      // Finalize full session recording
      void this.finalizeSession(sid)

      this.streamBySid.delete(sid)
      this.sessionBySid.delete(sid)
      this.timingBySid.delete(sid)
      this.chunkStateBySid.delete(sid)
      this.sessionStateBySid.delete(sid)
    } catch (e) {
      console.error(e)
    }
  }

  /**
   * Flush write buffer to file
   */
  private flushWriteBuffer (chunkState: ChunkState): void {
    if (chunkState.fd !== null && chunkState.writeBufferOffset > 0) {
      writeSync(chunkState.fd, chunkState.writeBuffer, 0, chunkState.writeBufferOffset)
      chunkState.writeBufferOffset = 0
    }
  }

  /**
   * Flush session write buffer to file
   */
  private flushSessionWriteBuffer (sessionState: SessionState): void {
    if (sessionState.fd !== null && sessionState.writeBufferOffset > 0) {
      writeSync(sessionState.fd, sessionState.writeBuffer, 0, sessionState.writeBufferOffset)
      sessionState.writeBufferOffset = 0
    }
  }

  /**
   * Write data to chunk with buffering
   */
  private writeToChunk (chunkState: ChunkState, data: Buffer): void {
    let dataOffset = 0

    while (dataOffset < data.length) {
      const spaceInBuffer = WRITE_BUFFER_SIZE - chunkState.writeBufferOffset
      const bytesToCopy = Math.min(spaceInBuffer, data.length - dataOffset)

      data.copy(chunkState.writeBuffer, chunkState.writeBufferOffset, dataOffset, dataOffset + bytesToCopy)
      chunkState.writeBufferOffset += bytesToCopy
      dataOffset += bytesToCopy

      // Flush buffer when full
      if (chunkState.writeBufferOffset >= WRITE_BUFFER_SIZE) {
        this.flushWriteBuffer(chunkState)
      }
    }

    chunkState.chunkDataLength += data.length
  }

  /**
   * Detect if current frame contains speech based on analysis
   */
  private isFrameSpeech (analysis: { rms: number, activeSamples: number, totalSamples: number }): boolean {
    const speechRatio = analysis.totalSamples > 0 ? analysis.activeSamples / analysis.totalSamples : 0
    return analysis.rms > VAD_THRESHOLD || speechRatio > SPEECH_RATIO_THRESHOLD
  }

  /**
   * Write data to full session with buffering
   */
  private writeToSession (sessionState: SessionState, data: Buffer): void {
    let dataOffset = 0

    while (dataOffset < data.length) {
      const spaceInBuffer = WRITE_BUFFER_SIZE - sessionState.writeBufferOffset
      const bytesToCopy = Math.min(spaceInBuffer, data.length - dataOffset)

      data.copy(sessionState.writeBuffer, sessionState.writeBufferOffset, dataOffset, dataOffset + bytesToCopy)
      sessionState.writeBufferOffset += bytesToCopy
      dataOffset += bytesToCopy

      // Flush buffer when full
      if (sessionState.writeBufferOffset >= WRITE_BUFFER_SIZE) {
        this.flushSessionWriteBuffer(sessionState)
      }
    }

    sessionState.dataLength += data.length
  }

  private finalizeChunk (sid: string, endReason: 'silence' | 'max_duration' | 'stream_end' = 'stream_end'): void {
    const chunkState = this.chunkStateBySid.get(sid)
    const timing = this.timingBySid.get(sid)
    const participant = this.participantBySid.get(sid)

    if (chunkState?.fd !== null && chunkState?.fd !== undefined && timing !== undefined) {
      // Skip if chunk is too short (less than minimum duration)
      const chunkDurationMs = chunkState.chunkEndTime - chunkState.chunkStartTime
      if (chunkDurationMs < MIN_CHUNK_DURATION_MS && endReason !== 'stream_end') {
        console.info('Skipping too short chunk', { sid, durationMs: chunkDurationMs, endReason })
        return
      }

      try {
        // Flush any remaining data in write buffer
        this.flushWriteBuffer(chunkState)

        // Update WAV header with correct data length
        updateWavHeader(chunkState.fd, chunkState.chunkDataLength)
        closeSync(chunkState.fd)

        // Calculate metadata using meeting start time (not stream start)
        const startTimeSec = (chunkState.chunkStartTime - this.meetingStartTime) / 1000
        const endTimeSec = (chunkState.chunkEndTime - this.meetingStartTime) / 1000
        const durationSec = endTimeSec - startTimeSec

        const rmsAmplitude =
          chunkState.totalSamples > 0 ? Math.sqrt(chunkState.sumSquares / chunkState.totalSamples) : 0

        const speechRatio = chunkState.totalSamples > 0 ? chunkState.activeSamples / chunkState.totalSamples : 0

        // Note: hasSpeech removed - we always send chunks to transcription
        // The transcription model will determine if there's actual speech

        const metadata: ChunkMetadata = {
          startTimeSec,
          endTimeSec,
          durationSec,
          participant: participant?.identity ?? sid,
          participantName: participant?.name ?? participant?.identity ?? sid,
          sampleRate: this.sampleRate,
          channels: this.channels,
          bitsPerSample: this.bitsPerSample,
          endReason,
          // Informational VAD metrics (for debugging/logging only)
          speechRatio,
          peakAmplitude: chunkState.peakAmplitude,
          rmsAmplitude
        }

        // Read WAV file, compress with gzip, and send to platform
        if (chunkState.chunkFilePath !== null) {
          try {
            const wavData = readFileSync(chunkState.chunkFilePath)
            const gzippedData = gzipSync(wavData, { level: 6 })

            // Send to platform
            void this.sendChunkToPlatform(gzippedData, sid, metadata).catch((e) => {
              console.error('Error sending chunk to platform', { error: e, sid })
            })

            // Clean up local WAV file (skip in debug mode)
            if (!config.Debug) {
              unlinkSync(chunkState.chunkFilePath)
            }

            console.info('Finalized chunk', {
              sid,
              chunkIndex: chunkState.chunkIndex,
              filePath: chunkState.chunkFilePath,
              dataLength: chunkState.chunkDataLength,
              durationMs: chunkDurationMs,
              participant: metadata.participant,
              participantName: metadata.participantName,
              endReason,
              speechRatio: speechRatio.toFixed(2),
              rmsAmplitude: rmsAmplitude.toFixed(4),
              originalSize: wavData.length,
              compressedSize: gzippedData.length,
              compressionRatio: ((1 - gzippedData.length / wavData.length) * 100).toFixed(1) + '%'
            })
          } catch (e) {
            console.error('Error compressing and sending chunk', { error: e, sid })
          }
        }
      } catch (e) {
        console.error('Error finalizing chunk', { error: e, sid })
      }

      // Reset chunk state for next chunk
      chunkState.fd = null
      chunkState.chunkFilePath = null
      chunkState.chunkDataLength = 0
      chunkState.totalSamples = 0
      chunkState.activeSamples = 0
      chunkState.peakAmplitude = 0
      chunkState.sumSquares = 0
      chunkState.writeBufferOffset = 0
      // Reset VAD state
      chunkState.isSpeaking = false
      chunkState.consecutiveSpeechMs = 0
      chunkState.consecutiveSilenceMs = 0
      chunkState.chunkIndex++
    }
  }

  /**
   * Finalize full session recording - convert to MP3 and send to platform
   */
  private async finalizeSession (sid: string): Promise<void> {
    const sessionState = this.sessionStateBySid.get(sid)
    const participant = this.participantBySid.get(sid)
    const timing = this.timingBySid.get(sid)

    if (sessionState?.fd === null || sessionState?.fd === undefined) {
      return
    }

    try {
      // Flush remaining buffer
      this.flushSessionWriteBuffer(sessionState)

      // Update WAV header
      updateWavHeader(sessionState.fd, sessionState.dataLength)
      closeSync(sessionState.fd)

      if (sessionState.filePath === null || sessionState.dataLength === 0) {
        console.info('No session data to finalize', { sid })
        return
      }

      const durationSec = timing !== undefined ? (Date.now() - timing.streamStartTime) / 1000 : 0

      // Calculate end time from meeting start
      const startTimeSec = sessionState.startTimeFromMeeting
      const endTimeSec = startTimeSec + durationSec
      const participantIdentity = participant?.identity ?? sid
      const participantName = sanitizePath(participant?.name ?? participant?.identity ?? sid)

      console.info('Finalizing session recording', {
        sid,
        participant: participantIdentity,
        participantName,
        dataLength: sessionState.dataLength,
        startTimeSec: startTimeSec.toFixed(1),
        endTimeSec: endTimeSec.toFixed(1),
        durationSec: durationSec.toFixed(1)
      })

      // Convert WAV to MP3
      const mp3Path = sessionState.filePath.replace('.wav', '.mp3')

      try {
        await convertWavToMp3(sessionState.filePath, mp3Path)
        console.info('Converted session to MP3', { sid, mp3Path })

        // Read MP3 and send to platform
        const mp3Data = readFileSync(mp3Path)

        await this.sendSessionToPlatform(mp3Data, participantIdentity, participantName, startTimeSec, endTimeSec, this.sessionNumber)

        // Clean up files (skip in debug mode)
        if (!config.Debug) {
          unlinkSync(sessionState.filePath)
          unlinkSync(mp3Path)
        }

        console.info('Session recording sent to platform', {
          sid,
          participant: participantName,
          mp3Size: mp3Data.length,
          durationSec: durationSec.toFixed(1)
        })
      } catch (e) {
        console.error('Error converting/sending session', { error: e, sid })
        // Clean up WAV file on error (skip in debug mode)
        if (!config.Debug) {
          try {
            unlinkSync(sessionState.filePath)
          } catch {}
        }
      }
    } catch (e) {
      console.error('Error finalizing session', { error: e, sid })
    }
  }

  private startNewChunk (sid: string, streamDir: string, startTimeAbs: number): void {
    const chunkState = this.chunkStateBySid.get(sid)
    if (chunkState === undefined) return

    // Calculate relative time in seconds from meeting start
    const startTimeSec = (startTimeAbs - this.meetingStartTime) / 1000

    // Create new chunk file
    const filename = `chunk_${chunkState.chunkIndex}_${startTimeSec.toFixed(1)}.wav`
    const filePath = join(streamDir, filename)

    try {
      // Open file and write placeholder header
      const fd = openSync(filePath, 'w')
      const placeholderHeader = createWavHeader(0, this.sampleRate, this.channels, this.bitsPerSample)
      writeSync(fd, placeholderHeader)

      chunkState.fd = fd
      chunkState.chunkStartTime = startTimeAbs
      chunkState.chunkEndTime = startTimeAbs
      chunkState.chunkDataLength = 0
      chunkState.chunkFilePath = filePath
      chunkState.totalSamples = 0
      chunkState.activeSamples = 0
      chunkState.peakAmplitude = 0
      chunkState.sumSquares = 0
      chunkState.writeBufferOffset = 0
      // Reset VAD state for new chunk
      chunkState.speechStartTime = startTimeAbs
      chunkState.silenceStartTime = 0
      chunkState.consecutiveSpeechMs = 0
      chunkState.consecutiveSilenceMs = 0

      console.info('Started new chunk', {
        sid,
        chunkIndex: chunkState.chunkIndex,
        filePath,
        startTimeSec: startTimeSec.toFixed(1)
      })
    } catch (e) {
      console.error('Error starting new chunk', { error: e, filePath })
    }
  }

  /**
   * Start full session recording for a participant
   */
  private startSession (sid: string, streamDir: string): void {
    const participant = this.participantBySid.get(sid)
    const participantName = sanitizePath(participant?.name ?? participant?.identity ?? sid)
    const filename = `${participantName}_session_${this.sessionNumber}.wav`
    const filePath = join(streamDir, filename)

    // Calculate start time from meeting start
    const now = Date.now()
    const startTimeFromMeeting = (now - this.meetingStartTime) / 1000

    try {
      const fd = openSync(filePath, 'w')
      const placeholderHeader = createWavHeader(0, this.sampleRate, this.channels, this.bitsPerSample)
      writeSync(fd, placeholderHeader)

      const sessionState: SessionState = {
        fd,
        filePath,
        dataLength: 0,
        writeBuffer: Buffer.alloc(WRITE_BUFFER_SIZE),
        writeBufferOffset: 0,
        startTimeFromMeeting,
        frameCount: 0,
        lastFrameHash: 0
      }

      this.sessionStateBySid.set(sid, sessionState)

      console.info('Started session recording', {
        sid,
        filePath,
        participant: participant?.name ?? participant?.identity,
        startTimeFromMeeting: startTimeFromMeeting.toFixed(1)
      })
    } catch (e) {
      console.error('Error starting session', { error: e, filePath })
    }
  }

  processTrack (sid: string): void {
    const track = this.trackBySid.get(sid)
    if (track === undefined) return

    // Don't start if already streaming for this sid
    if (this.streamBySid.has(sid)) {
      console.info('Stream already active for track, skipping', { sid })
      return
    }

    // Clear stopped flag when starting new stream
    this.stoppedSids.delete(sid)

    const stream = new AudioStream(track, 16000)

    const participant = this.participantBySid.get(sid)
    console.info('Starting transcription for track', { room: this.room.name, sid, participant: participant?.name ?? participant?.identity })

    this.streamBySid.set(sid, stream)

    void this.streamToFiles(sid, stream).catch((err) => {
      console.error('Failed to stream', { participant: participant?.name ?? participant?.identity ?? sid, error: err })
    })
  }

  async streamToFiles (sid: string, stream: AudioStream): Promise<void> {
    const participant = this.participantBySid.get(sid)
    // Create directory for this stream using participant identity
    const streamDir = join(this.rootDir, sanitizePath(participant?.name ?? participant?.identity ?? sid))
    if (!existsSync(streamDir)) {
      mkdirSync(streamDir, { recursive: true })
    }

    // Initialize timing for this stream
    const streamStartTime = Date.now()
    const timing: StreamTiming = {
      streamStartTime,
      lastFrameEndTime: streamStartTime
    }
    this.timingBySid.set(sid, timing)

    // Initialize chunk state with write buffer and VAD state
    const chunkState: ChunkState = {
      fd: null,
      chunkStartTime: streamStartTime,
      chunkEndTime: streamStartTime,
      chunkDataLength: 0,
      chunkFilePath: null,
      totalSamples: 0,
      activeSamples: 0,
      peakAmplitude: 0,
      sumSquares: 0,
      writeBuffer: Buffer.alloc(WRITE_BUFFER_SIZE),
      writeBufferOffset: 0,
      // VAD state for smart chunking
      isSpeaking: false,
      speechStartTime: 0,
      silenceStartTime: 0,
      lastSpeechEndTime: 0,
      consecutiveSpeechMs: 0,
      consecutiveSilenceMs: 0,
      chunkIndex: 0
    }
    this.chunkStateBySid.set(sid, chunkState)

    // Start full session recording
    this.startSession(sid, streamDir)

    console.info('Stream started', {
      sid,
      streamStartTime,
      meetingStartTime: this.meetingStartTime,
      streamDir,
      writeBufferSize: WRITE_BUFFER_SIZE,
      maxChunkDurationMs: MAX_CHUNK_DURATION_MS,
      silenceThresholdMs: SILENCE_THRESHOLD_MS
    })

    for await (const frame of stream) {
      // Exit loop if this stream was stopped
      if (this.stoppedSids.has(sid)) {
        console.info('Stream stopped, exiting loop', { sid })
        break
      }

      if (!this.isInProgress) continue

      const frameStartTime = timing.lastFrameEndTime
      // Calculate frame duration based on samples: duration_ms = (samples / sampleRate) * 1000
      const frameDurationMs = (frame.samplesPerChannel / this.sampleRate) * 1000
      const frameEndTime = frameStartTime + frameDurationMs

      // Update timing
      timing.lastFrameEndTime = frameEndTime

      // IMPORTANT: Create a proper copy of the buffer, not a view
      // frame.data.buffer may be reused by LiveKit for the next frame
      const buf = Buffer.from(new Uint8Array(frame.data.buffer))

      // Analyze audio for VAD
      const analysis = analyzeAudioBuffer(buf)
      const frameHasSpeech = this.isFrameSpeech(analysis)

      // Update VAD state
      if (frameHasSpeech) {
        // Speech detected
        chunkState.consecutiveSpeechMs += frameDurationMs
        chunkState.consecutiveSilenceMs = 0
        chunkState.lastSpeechEndTime = frameEndTime

        if (!chunkState.isSpeaking && chunkState.consecutiveSpeechMs >= SPEECH_START_THRESHOLD_MS) {
          // Speech started - transition from silence to speaking
          chunkState.isSpeaking = true
          chunkState.speechStartTime = frameStartTime - chunkState.consecutiveSpeechMs

          // Start a new chunk if we don't have one
          if (chunkState.fd === null) {
            this.startNewChunk(sid, streamDir, chunkState.speechStartTime)
          }
        }
      } else {
        // Silence detected
        chunkState.consecutiveSilenceMs += frameDurationMs
        chunkState.consecutiveSpeechMs = 0

        if (chunkState.isSpeaking && chunkState.consecutiveSilenceMs >= SILENCE_THRESHOLD_MS) {
          // End of phrase detected - 1 second of silence after speech
          chunkState.isSpeaking = false

          if (chunkState.fd !== null) {
            // Finalize chunk at the point where speech ended (not including trailing silence)
            chunkState.chunkEndTime = chunkState.lastSpeechEndTime
            this.finalizeChunk(sid, 'silence')

            console.info('Phrase ended (silence detected)', {
              sid,
              silenceDurationMs: chunkState.consecutiveSilenceMs,
              chunkIndex: chunkState.chunkIndex
            })
          }
        }
      }

      // Safety check: if chunk is too long, finalize it regardless of speech state
      if (chunkState.fd !== null) {
        const currentDuration = frameEndTime - chunkState.chunkStartTime
        if (currentDuration >= MAX_CHUNK_DURATION_MS) {
          chunkState.chunkEndTime = frameEndTime
          this.finalizeChunk(sid, 'max_duration')

          console.info('Chunk finalized (max duration reached)', {
            sid,
            durationMs: currentDuration,
            chunkIndex: chunkState.chunkIndex
          })

          // If still speaking, start a new chunk immediately
          if (chunkState.isSpeaking) {
            this.startNewChunk(sid, streamDir, frameEndTime)
          }
        }
      }

      // Append audio data to current chunk using buffered write
      if (chunkState.fd !== null) {
        try {
          this.writeToChunk(chunkState, buf)
          chunkState.chunkEndTime = frameEndTime

          // Update VAD metrics
          chunkState.totalSamples += analysis.totalSamples
          chunkState.activeSamples += analysis.activeSamples
          chunkState.sumSquares += analysis.sumSquares
          if (analysis.peak > chunkState.peakAmplitude) {
            chunkState.peakAmplitude = analysis.peak
          }
        } catch (e) {
          console.error('Error writing audio data to chunk', { error: e, sid })
        }
      }

      // Write to full session recording
      const sessionState = this.sessionStateBySid.get(sid)
      if (sessionState?.fd !== null && sessionState !== undefined) {
        try {
          // Calculate simple hash to detect duplicate frames
          let frameHash = 0
          for (let i = 0; i < Math.min(buf.length, 100); i++) {
            frameHash = ((frameHash << 5) - frameHash + buf[i]) | 0
          }
          frameHash = (frameHash + buf.length) | 0

          // Check for potential duplicate
          if (sessionState.lastFrameHash === frameHash && sessionState.frameCount > 0) {
            console.warn('Potential duplicate frame detected in session', {
              sid,
              frameCount: sessionState.frameCount,
              frameHash,
              bufLength: buf.length,
              frameStartTime,
              frameEndTime
            })
          }

          sessionState.lastFrameHash = frameHash
          sessionState.frameCount++

          // Log every 100 frames for debugging
          if (sessionState.frameCount % 100 === 0) {
            console.info('Session write progress', {
              sid,
              frameCount: sessionState.frameCount,
              dataLength: sessionState.dataLength,
              bufLength: buf.length,
              frameStartTime,
              frameEndTime
            })
          }

          this.writeToSession(sessionState, buf)
        } catch (e) {
          console.error('Error writing to session', { error: e, sid })
        }
      }
    }

    // Check if we were stopped externally (via stopWs) or ended naturally
    const wasStopped = this.stoppedSids.has(sid)

    // Only finalize chunk if stream ended naturally (not stopped via stopWs)
    // When stopped via stopWs, finalization is already done there
    if (!wasStopped && chunkState.fd !== null) {
      this.finalizeChunk(sid, 'stream_end')
    }

    // Clean up
    this.stoppedSids.delete(sid)
    this.streamBySid.delete(sid)

    const streamEndTime = Date.now()
    console.info('Stream ended', {
      sid,
      wasStopped,
      streamStartTime: timing.streamStartTime,
      streamEndTime,
      totalDurationMs: streamEndTime - timing.streamStartTime,
      totalChunks: chunkState.chunkIndex + 1
    })
  }

  async sendChunkToPlatform (gzipData: Buffer, sid: string, metadata: ChunkMetadata): Promise<void> {
    try {
      const response = await fetch(`${config.PlatformUrl}/love/send_raw`, {
        method: 'POST',
        keepalive: true,
        headers: {
          'Content-Type': 'application/octet-stream',
          Authorization: 'Bearer ' + this.token,
          'X-Room-Name': this.room.name ?? '',
          'X-Participant': metadata.participant,
          'X-Participant-Name': metadata.participantName,
          'X-Start-Time': metadata.startTimeSec.toString(),
          'X-End-Time': metadata.endTimeSec.toString(),
          'X-Duration': metadata.durationSec.toString(),
          'X-Sample-Rate': metadata.sampleRate.toString(),
          'X-Channels': metadata.channels.toString(),
          'X-Bits-Per-Sample': metadata.bitsPerSample.toString(),
          'X-End-Reason': metadata.endReason
        },
        body: new Uint8Array(gzipData)
      })

      if (!response.ok) {
        console.error('Failed to send chunk to platform', {
          status: response.status,
          statusText: response.statusText,
          sid
        })
      }
    } catch (e) {
      console.error('Error sending chunk to platform', { error: e, sid })
    }
  }

  /**
   * Send full session MP3 to platform for attachment
   */
  async sendSessionToPlatform (
    mp3Data: Buffer,
    participant: string,
    participantName: string,
    startTimeSec: number,
    endTimeSec: number,
    sessionNumber: number
  ): Promise<void> {
    try {
      const response = await fetch(`${config.PlatformUrl}/love/send_session`, {
        method: 'POST',
        keepalive: true,
        headers: {
          'Content-Type': 'audio/mpeg',
          'Content-Length': mp3Data.length.toString(),
          Authorization: 'Bearer ' + this.token,
          'X-Room-Name': this.room.name ?? '',
          'X-Participant': participant,
          'X-Participant-Name': participantName,
          'X-Start-Time': startTimeSec.toString(),
          'X-End-Time': endTimeSec.toString(),
          'X-Session-Number': sessionNumber.toString()
        },
        body: new Uint8Array(mp3Data)
      })

      if (!response.ok) {
        console.error('Failed to send session to platform', {
          status: response.status,
          statusText: response.statusText,
          participant
        })
      }
    } catch (e) {
      console.error('Error sending session to platform', { error: e, participant })
    }
  }

  async sendToPlatform (transcript: string, sid: string): Promise<void> {
    const request = {
      transcript,
      participant: this.participantBySid.get(sid)?.identity,
      roomName: this.room.name
    }

    this.transcriptionCount++

    if (this.transcriptionCount === 1 || this.transcriptionCount % 50 === 0) {
      console.log('Sending transcript', this.room.name, this.transcriptionCount)
    }

    try {
      await fetch(`${config.PlatformUrl}/love/transcript`, {
        method: 'POST',
        keepalive: true,
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer ' + this.token
        },
        body: JSON.stringify(request)
      })
    } catch (e) {
      console.error('Error sending to platform', e)
    }
  }

  close (): void {
    // Finalize all open chunks and sessions
    for (const sid of this.chunkStateBySid.keys()) {
      this.finalizeChunk(sid)
    }
    for (const sid of this.sessionStateBySid.keys()) {
      void this.finalizeSession(sid)
    }
    this.trackBySid.clear()
    this.participantBySid.clear()
    this.chunkStateBySid.clear()
    this.sessionStateBySid.clear()
  }
}
