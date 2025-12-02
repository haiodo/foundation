// Copyright © 2025 Andrey Sobolev (haiodo@gmail.com)

import { gunzip as _gunzip } from 'zlib'

import { MeasureContext, withContext, WorkspaceUuid, type WorkspaceIds } from '@hcengineering/core'
import { StorageAdapter } from '@hcengineering/server-core'

import { TranscriptionQueueTask, TranscriptionProvider, TranscriptionConfig } from './types'
import { analyzeAudio } from './vad'
import { normalizeAudio } from './normalize'
import { createTranscriptionProvider } from './index'
import { promisify } from 'util'
import path from 'path'
import { writeFile } from 'fs/promises'

const gunzip = promisify(_gunzip)

/**
 * Callback type for sending transcription results
 */
export type TranscriptSendCallback = (
  ctx: MeasureContext,
  workspace: WorkspaceUuid,
  roomName: string,
  participant: string,
  transcript: string,
  startTimeSec: number,
  endTimeSec: number
) => Promise<void>

/**
 * Callback type for getting workspace storage info
 */
export type GetWorkspaceStorageCallback = (workspace: WorkspaceUuid) => Promise<{
  wsIds: WorkspaceIds
} | undefined>

/**
 * Transcription queue consumer
 *
 * Processes audio chunks from the queue:
 * 1. Loads gzipped WAV from storage
 * 2. Verifies speech presence with VAD
 * 3. Transcribes using configured provider
 * 4. Handles overlap correction
 * 5. Sends transcript to platform
 * 6. Cleans up storage
 */
export class TranscriptionConsumer {
  private readonly provider: TranscriptionProvider

  constructor (
    private readonly ctx: MeasureContext,
    private readonly config: TranscriptionConfig,
    private readonly storageAdapter: StorageAdapter,
    private readonly getWorkspaceStorage: GetWorkspaceStorageCallback,
    private readonly sendTranscript: TranscriptSendCallback,
    private readonly debugDir?: string
  ) {
    this.provider = createTranscriptionProvider(ctx, config)
  }

  /**
   * Process a transcription task from the queue
   */
  @withContext('processTranscription')
  async processTask (ctx: MeasureContext, workspace: WorkspaceUuid, task: TranscriptionQueueTask): Promise<void> {
    if (this.provider === undefined) {
      this.ctx.error('Transcription provider not available, skipping task', { blobId: task.blobId })
      return
    }

    const startTime = Date.now()

    try {
      // Get workspace storage info
      const wsInfo = await this.getWorkspaceStorage(workspace)
      if (wsInfo === undefined) {
        this.ctx.error('Failed to get workspace storage info', { workspace, blobId: task.blobId })
        return
      }

      // Load gzipped WAV from storage
      const gzipData = await this.loadFromStorage(ctx, wsInfo.wsIds, task.blobId)
      if (gzipData === undefined) {
        this.ctx.error('Failed to load audio from storage', { workspace, blobId: task.blobId })
        return
      }

      // Decompress gzip to WAV
      let wavData: Buffer
      try {
        wavData = await gunzip(gzipData)
      } catch (err: any) {
        this.ctx.error('Failed to decompress audio', { workspace, blobId: task.blobId, error: err.message })
        return
      }

      // Verify speech presence with VAD (don't trust task.hasSpeech)
      const vadResult = analyzeAudio(
        ctx,
        wavData,
        this.config.vadRmsThreshold,
        this.config.vadSpeechRatioThreshold
      )

      if (!vadResult.hasSpeech) {
        this.ctx.info('No speech detected by VAD, skipping transcription', {
          workspace,
          blobId: task.blobId,
          participant: task.participant,
          originalHasSpeech: task.hasSpeech,
          vadRms: vadResult.rmsAmplitude.toFixed(4),
          vadSpeechRatio: vadResult.speechRatio.toFixed(4)
        })
      }

      // Normalize audio before transcription
      const normalizedWavData = ctx.withSync('normalizeAudio', {}, () => normalizeAudio(wavData))

      // Transcribe audio
      const result = await ctx.with('transcribe', {}, () => this.provider.transcribe(normalizedWavData, {
        wordTimestamps: true, // Needed for overlap correction
        sampleRate: task.sampleRate,
        channels: task.channels
      }))

      ctx.info('Received transcription result', { result: result.text, language: result.language })

      if (result.text.trim() === '') {
        this.ctx.info('Empty transcription result', {
          workspace,
          blobId: task.blobId,
          participant: task.participant
        })
      }

      const finalText = result.text

      if (this.debugDir !== '' && this.debugDir != null) {
        // We need to store chunk and transcription to testing file.
        const blName = path.join(this.debugDir, workspace, task.participant, `${task.blobId}_${Date.now()}`)
        await writeFile(blName + '.wav', wavData)
        await writeFile(blName + '.json', JSON.stringify({ result, finalText }))
      }

      if (finalText.trim() === '') {
        this.ctx.info('Empty transcription after overlap correction', {
          workspace,
          blobId: task.blobId,
          participant: task.participant,
          originalText: result.text.substring(0, 100)
        })
        await this.cleanupStorage(wsInfo.wsIds, task.blobId)
        return
      }

      // Send transcript to platform
      await this.sendTranscript(
        ctx,
        workspace,
        task.roomName,
        task.participant,
        finalText,
        task.startTimeSec,
        task.endTimeSec
      )

      const elapsed = Date.now() - startTime
      this.ctx.info('Transcription task completed', {
        workspace,
        blobId: task.blobId,
        participant: task.participant,
        textLength: finalText.length,
        durationSec: task.durationSec,
        elapsedMs: elapsed

      })

      // Cleanup storage
      await this.cleanupStorage(wsInfo.wsIds, task.blobId)
    } catch (err: any) {
      const elapsed = Date.now() - startTime
      this.ctx.error('Transcription task failed', {
        workspace,
        blobId: task.blobId,
        participant: task.participant,
        error: err.message,
        elapsedMs: elapsed
      })
    }
  }

  /**
   * Load blob from storage
   */
  @withContext('loadFromStorage')
  private async loadFromStorage (
    ctx: MeasureContext,
    wsIds: WorkspaceIds,
    blobId: string
  ): Promise<Buffer | undefined> {
    try {
      return Buffer.concat(await this.storageAdapter.read(ctx, wsIds, blobId))
    } catch (err: any) {
      this.ctx.error('Storage read error', { blobId, error: err.message })
      return undefined
    }
  }

  /**
   * Remove blob from storage after processing
   */
  private async cleanupStorage (
    wsIds: WorkspaceIds,
    blobId: string
  ): Promise<void> {
    try {
      await this.storageAdapter.remove(this.ctx, wsIds, [blobId])
    } catch (err: any) {
      this.ctx.error('Storage cleanup error', { blobId, error: err.message })
    }
  }

  /**
   * Get provider name for monitoring
   */
  getProviderName (): string {
    return this.provider?.name ?? 'none'
  }

  /**
   * Check if consumer is ready
   */
  isReady (): boolean {
    return this.provider !== undefined
  }
}

/**
 * Create transcription consumer instance
 */
export function createTranscriptionConsumer (
  ctx: MeasureContext,
  config: TranscriptionConfig,
  storageAdapter: StorageAdapter,
  getWorkspaceStorage: GetWorkspaceStorageCallback,
  sendTranscript: TranscriptSendCallback,
  debugDir?: string
): TranscriptionConsumer {
  return new TranscriptionConsumer(ctx, config, storageAdapter, getWorkspaceStorage, sendTranscript, debugDir)
}
