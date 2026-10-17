//
// Copyright © 2026 Intabia Fusion.
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

// The whole stt-worker path on a real broker: Kafka -> batch consumer -> ServerProvider ->
// clisr -> worker -> back. Own topic and group per run, so the stand's topics stay untouched.

// Stub config so importing the provider chain does not run the env-validating IIFE.
jest.mock('../config', () => ({ __esModule: true, default: {} }))

/* eslint-disable import/first */
import { MeasureMetricsContext, newMetrics } from '@hcengineering/core'
import { createPlatformQueue, parseQueueConfig } from '@hcengineering/kafka'
import type { PlatformQueue } from '@hcengineering/server-core'
import { kafkaBrokers } from '@hcengineering/test-containers'
import { runScenario } from './kafka-clisr-scenario'
/* eslint-enable import/first */

describe('e2e: kafka -> clisr -> worker', () => {
  const ctx = new MeasureMetricsContext('queue-e2e', {}, {}, newMetrics())
  let brokers: string
  let queue: PlatformQueue

  beforeAll(async () => {
    brokers = await kafkaBrokers()
    queue = createPlatformQueue(parseQueueConfig(brokers, 'clisr-e2e', ''))
  })

  afterAll(async () => {
    await queue.shutdown()
  })

  jest.setTimeout(600000)

  it('carries every task through the whole chain', async () => {
    const run = await runScenario(ctx, queue, { tasks: 200, batchSize: 16, capacity: 4, workMs: 1 })
    // runScenario only resolves once all `tasks` transcripts came back, and each one is checked
    // against its own id, so reaching here is the assertion. Depth is the interesting part.
    expect(run.maxQueueBatch).toBeGreaterThan(1)
    expect(run.maxInFlight).toBe(4)
  })

  // Batch size 1 uses the single-message consumer, so only one task is ever in the air and clisr
  // capacity has nothing to work with; capacity 1 caps a batch the same way.
  it('keeps one task in flight with batch size 1 or capacity 1', async () => {
    const single = await runScenario(ctx, queue, { tasks: 20, batchSize: 1, capacity: 4, workMs: 1 })
    expect(single.maxInFlight).toBe(1)
    const capped = await runScenario(ctx, queue, { tasks: 20, batchSize: 8, capacity: 1, workMs: 1 })
    expect(capped.maxInFlight).toBe(1)
  })
})
