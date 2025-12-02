//
// Copyright © 2024-2025 Hardcore Engineering Inc.
//
// Licensed under the Eclipse Public License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License. You may
// obtain a copy of the License at https://www.eclipse.org/legal/epl-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
// See the License for the specific language governing permissions and
// limitations under the License.
//
import { setMetadata } from '@hcengineering/platform'
import serverClient, { withRetry } from '@hcengineering/server-client'
import {
  ConsumerHandle,
  initStatisticsContext,
  QueueTopic,
  QueueWorkspaceEvent,
  QueueWorkspaceMessage
} from '@hcengineering/server-core'
import serverToken, { generateToken } from '@hcengineering/server-token'

import { getClient as getAccountClient } from '@hcengineering/account-client'
import { AIEventRequest } from '@hcengineering/ai-bot'
import { createOpenTelemetryMetricsContext, SplitLogger } from '@hcengineering/analytics-service'
import { newMetrics, type SocialId, type WorkspaceUuid, type Ref } from '@hcengineering/core'
import { type Room } from '@hcengineering/love'
import { type Person } from '@hcengineering/contact'
import { getPlatformQueue } from '@hcengineering/kafka'
import { join } from 'path'
import { updateDeepgramBilling } from './billing'
import config from './config'
import { AIControl } from './controller'
import { registerLoaders } from './loaders'
import { createServer, listen } from './server/server'
import { getAccountUuid } from './utils/account'
import { TranscriptionTask } from './types'
import {
  createTranscriptionConsumer,
  TranscriptionConfig,
  TranscriptionQueueTask
} from './transcription'
import { existsSync } from 'fs'
import { mkdir } from 'fs/promises'

export const start = async (): Promise<void> => {
  setMetadata(serverToken.metadata.Secret, config.ServerSecret)
  setMetadata(serverToken.metadata.Service, 'ai-bot-service')
  setMetadata(serverClient.metadata.UserAgent, config.ServiceID)
  setMetadata(serverClient.metadata.Endpoint, config.AccountsURL)

  registerLoaders()

  const queue = getPlatformQueue(QueueTopic.AIQueue)

  const ctx = initStatisticsContext('ai-bot-service', {
    factory: () =>
      createOpenTelemetryMetricsContext(
        'ai-bot-service',
        {},
        {},
        newMetrics(),
        new SplitLogger('ai-bot-service', {
          root: join(process.cwd(), 'logs'),
          enableConsole: (process.env.ENABLE_CONSOLE ?? 'true') === 'true'
        })
      )
  })
  ctx.info('AI Bot Service started', { firstName: config.FirstName, lastName: config.LastName })

  const personUuid = await withRetry(
    async () => await getAccountUuid(ctx),
    (_, attempt) => attempt >= 5,
    5000
  )()

  if (personUuid === undefined) {
    ctx.error('AI Bot Service failed to start. No person found.')
    process.exit()
  }
  ctx.info('AI person uuid', { personUuid })

  const socialIds: SocialId[] = await getAccountClient(
    config.AccountsURL,
    generateToken(personUuid, undefined, { service: 'aibot' })
  ).getSocialIds()

  const aiControl = new AIControl(personUuid, socialIds, ctx)

  // Create a workspace consumer
  // Create queue consumer's
  //
  const workspaceConsumer = queue.createConsumer<QueueWorkspaceMessage>(
    ctx,
    QueueTopic.Workspace,
    'ai-bot',
    async (ctx, message, control) => {
      try {
        if (message.value.type === QueueWorkspaceEvent.Up) {
          await aiControl.connect(message.workspace)
        }
      } catch (err: any) {
        ctx.error('failed to handle operation', { error: err.message })
      }
    }
  )

  const aiEventConsumer = queue.createConsumer<AIEventRequest>(
    ctx,
    QueueTopic.AIQueue,
    'ai-bot',
    async (ctx, message, control) => {
      try {
        await aiControl.processEvent(message.workspace, [message.value], control)
      } catch (err: any) {
        ctx.error('failed to handle ai event', { error: err.message })
      }
    }
  )

  // Set up transcription queue producer
  const transcriptionProducer = queue.getProducer<TranscriptionTask>(ctx, QueueTopic.TranscriptionQueue)
  aiControl.setTranscriptionProducer(transcriptionProducer)

  // Set up transcription configuration from environment
  const transcriptionConfig: TranscriptionConfig = {
    provider: config.SttProvider,
    url: config.SttUrl,
    apiKey: config.SttApiKey,
    model: config.SttModel,
    vadRmsThreshold: config.VadRmsThreshold,
    vadSpeechRatioThreshold: config.VadSpeechRatioThreshold
  }

  ctx.info('Transcription config', {
    provider: transcriptionConfig.provider,
    url: transcriptionConfig.url,
    vadRmsThreshold: transcriptionConfig.vadRmsThreshold,
    vadSpeechRatioThreshold: transcriptionConfig.vadSpeechRatioThreshold
  })

  let transcriptionConsumer: ConsumerHandle | undefined

  try {
    if (config.DebugDir !== '' && config.DebugDir != null) {
      // We need to store chunk and transcription to testing file.
      if (!existsSync(config.DebugDir)) {
        await mkdir(config.DebugDir, { recursive: true })
      }
    }
    // Create transcription consumer with real implementation
    const transcriptionHandler = createTranscriptionConsumer(
      ctx,
      transcriptionConfig,
      aiControl.storageAdapter,
      // Callback to get workspace storage info
      async (workspace: WorkspaceUuid) => {
        const wsClient = await aiControl.getWorkspaceClient(workspace)
        if (wsClient === undefined) {
          return undefined
        }
        return { wsIds: wsClient.wsIds }
      },
      // Callback to send transcript to platform
      async (ctx, workspace: WorkspaceUuid, roomName: string, participant: string, transcript: string, _startTimeSec: number, _endTimeSec: number) => {
        const wsClient = await aiControl.getWorkspaceClient(workspace)
        if (wsClient === undefined) {
          ctx.error('Failed to get workspace client for sending transcript', { workspace })
          return
        }
        // Parse roomId from roomName (format: workspaceUuid_roomId)
        const parsed = roomName.split('_')
        const roomId = parsed[parsed.length - 1] as Ref<Room>

        // participant identity from LiveKit is Ref<Person> as string
        await wsClient.processLoveTranscript(ctx, transcript, participant as Ref<Person>, roomId)
      },
      config.DebugDir
    )

    if (!transcriptionHandler.isReady()) {
      ctx.warn('Transcription consumer not ready - check provider configuration', {
        provider: transcriptionConfig.provider
      })
    }

    // Set up queue consumer for transcription tasks
    transcriptionConsumer = queue.createConsumer<TranscriptionTask>(
      ctx,
      QueueTopic.TranscriptionQueue,
      'ai-bot-transcription',
      async (ctx, message, control) => {
        const task = message.value as unknown as TranscriptionQueueTask
        const workspace = message.workspace

        const int = setInterval(() => {
          void control.heartbeat()
        }, 1000)

        try {
          await transcriptionHandler.processTask(ctx, workspace, task)
        } catch (err: any) {
          ctx.error('Failed to process transcription task', { error: err.message, workspace, blobId: task.blobId })
        } finally {
          clearInterval(int)
        }
      }
    )
  } catch (err: any) {
    ctx.info('Failed to create transcription consumer', { error: err.message })
  }

  const app = createServer(aiControl, ctx)
  const server = listen(app, config.Port)

  let billingIntervalId: any | undefined
  if (config.BillingUrl !== '') {
    billingIntervalId = setInterval(
      () => {
        try {
          void updateDeepgramBilling(ctx)
        } catch {}
      },
      config.DeepgramPollIntervalMinutes * 60 * 1000
    )
    try {
      void updateDeepgramBilling(ctx)
    } catch {}
  }

  const onClose = (): void => {
    void workspaceConsumer.close()
    void aiEventConsumer.close()
    void transcriptionConsumer?.close()
    void transcriptionProducer.close()
    if (billingIntervalId !== undefined) {
      clearInterval(billingIntervalId)
    }
    void aiControl.close()
    server.close(() => process.exit())
  }

  process.on('SIGINT', onClose)
  process.on('SIGTERM', onClose)
  process.on('uncaughtException', (e: Error) => {
    console.error(e)
  })
  process.on('unhandledRejection', (e: Error) => {
    console.error(e)
  })
}
