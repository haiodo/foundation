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

import contact from '@hcengineering/contact'
import { type TxOperations } from '@hcengineering/core'
import { expect, test } from '../fixtures'
import {
  connectTracker,
  createIssue,
  deleteIssuesByTitlePrefix,
  getProjectContext,
  type ProjectContext
} from '../API/TrackerApi'
import { IssuesPage } from '../model/tracker/issues-page'
import { PlatformSetting, PlatformURI, generateId } from '../utils'

test.use({
  storageState: PlatformSetting
})

// An issue without an assignee puts null among the ids the search is limited to, and Elastic
// rejected the whole query for it: the search found nobody.
test.describe('Tracker filter search', () => {
  let client: TxOperations
  let ctx: ProjectContext
  const titlePrefix = `filter-search-${generateId()}-`

  test.beforeAll(async () => {
    client = (await connectTracker()).client
    ctx = await getProjectContext(client)
  })

  test.afterAll(async () => {
    if (client === undefined) return
    await deleteIssuesByTitlePrefix(client, titlePrefix)
    await client.close()
  })

  test('Assignee filter finds a person by name next to unassigned issues', async ({ page }) => {
    const issuesPage = new IssuesPage(page)
    const assignee = await client.findOne(contact.class.Person, { name: 'Appleseed,John' })
    if (assignee === undefined) throw new Error('No person Appleseed John in sanity-ws')

    const assigned = `${titlePrefix}assigned`
    const unassigned = `${titlePrefix}unassigned`
    await createIssue(client, ctx, { title: assigned, status: 'Backlog', assignee: assignee._id })
    await createIssue(client, ctx, { title: unassigned, status: 'Backlog' })

    const projectPath = encodeURIComponent(ctx.project._id)
    await (await page.goto(`${PlatformURI}/workbench/sanity-ws/tracker/${projectPath}/issues`))?.finished()
    await issuesPage.clickModelSelectorAll()

    await test.step('Search the assignee', async () => {
      await issuesPage.selectFilter('Assignee')
      await expect(issuesPage.menuPopupItemButton('Not selected')).toBeVisible()

      await issuesPage.inputSearch().fill('Appleseed')
      await expect(issuesPage.menuPopupItemButton('Appleseed John')).toBeVisible({ timeout: 15000 })
    })

    await test.step('Filter by the found assignee', async () => {
      await issuesPage.menuPopupItemButton('Appleseed John').click()
      await page.keyboard.press('Escape')
      await issuesPage.checkFilter('Assignee', 'is')
      await issuesPage.checkFilteredIssueExist(assigned)
      await issuesPage.checkFilteredIssueNotExist(unassigned)
    })
  })
})
