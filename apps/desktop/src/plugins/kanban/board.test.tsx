import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness drives the host's request scope, as a connection switch does.
// eslint-disable-next-line no-restricted-imports
import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'
import type { KanbanBoard } from './types'

vi.mock('./board-switcher', () => ({ BoardSwitcher: () => null }))
vi.mock('./drawer', () => ({ TaskDrawer: () => null }))
vi.mock('./orchestration', () => ({ OrchestrationPanel: () => null }))
vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

let board: KanbanBoard
let failNextCompletion = false

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/board') {
    return board
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  if (path === '/orchestration') {
    return { default_assignee: '' }
  }

  if (path === '/boards') {
    return { boards: [], current: '' }
  }

  if (path === '/projects') {
    return { projects: [] }
  }

  if (path === '/tasks' && _options?.method === 'POST') {
    return { task: { id: 't_new', title: 'New task', status: 'ready' } }
  }

  if (path.startsWith('/tasks/') && _options?.method === 'PATCH') {
    if (failNextCompletion && _options.body && (_options.body as { status?: string }).status === 'done') {
      failNextCompletion = false
      throw new Error('completion failed')
    }

    return { ok: true }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

beforeEach(() => {
  failNextCompletion = false
  board = {
    columns: [
      { name: 'todo', tasks: [{ id: 't_empty', title: 'Empty human task', status: 'todo', result: null }] },
      { name: 'done', tasks: [] }
    ],
    tenants: [],
    assignees: [],
    latest_event_id: 0,
    now: 0
  }
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openBoard() {
  render(
    <QueryClientProvider client={client}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )
}

describe('manual board completion', () => {
  it('asks for a summary before a drag/drop to Done and sends it with the move', async () => {
    openBoard()
    const card = await screen.findByText('Empty human task')
    const doneLane = await screen.findByRole('button', { name: /expand done/i })
    let draggedId = ''

    const dataTransfer = {
      effectAllowed: 'move',
      getData: () => draggedId,
      setData: (_type: string, value: string) => {
        draggedId = value
      },
      setDragImage: vi.fn()
    } as unknown as DataTransfer

    fireEvent.dragStart(card.closest('[draggable="true"]')!, { dataTransfer, offsetX: 1, offsetY: 1 })
    fireEvent.drop(doneLane, { dataTransfer })

    expect(await screen.findByRole('textbox', { name: /result/i })).toBeTruthy()
    expect(rest).not.toHaveBeenCalledWith('/tasks/t_empty', expect.anything())
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('textbox', { name: /result/i })).toBeNull())
    expect(rest).not.toHaveBeenCalledWith('/tasks/t_empty', expect.anything())

    fireEvent.dragStart(card.closest('[draggable="true"]')!, { dataTransfer, offsetX: 1, offsetY: 1 })
    fireEvent.drop(doneLane, { dataTransfer })
    fireEvent.change(screen.getByRole('textbox', { name: /result/i }), {
      target: { value: 'Approved the completed work.' }
    })
    fireEvent.click(screen.getByRole('button', { name: /complete task/i }))

    await waitFor(() => {
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_empty',
        expect.objectContaining({
          method: 'PATCH',
          body: { status: 'done', result: 'Approved the completed work.', summary: 'Approved the completed work.' }
        })
      )
    })
  })

  it('preserves an existing result when a card is dragged to Done', async () => {
    board.columns[0].tasks[0].result = 'Existing evidence.'
    openBoard()
    const card = await screen.findByText('Empty human task')
    const doneLane = await screen.findByRole('button', { name: /expand done/i })
    let draggedId = ''

    const dataTransfer = {
      effectAllowed: 'move',
      getData: () => draggedId,
      setData: (_type: string, value: string) => {
        draggedId = value
      },
      setDragImage: vi.fn()
    } as unknown as DataTransfer

    fireEvent.dragStart(card.closest('[draggable="true"]')!, { dataTransfer, offsetX: 1, offsetY: 1 })
    fireEvent.drop(doneLane, { dataTransfer })

    await waitFor(() => {
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_empty',
        expect.objectContaining({ method: 'PATCH', body: { status: 'done', result: 'Existing evidence.' } })
      )
    })
    expect(screen.queryByRole('textbox', { name: /result/i })).toBeNull()
  })

  it('asks before creating a new task directly in Done', async () => {
    board.columns[1].tasks = [{ id: 't_done', title: 'Already done', status: 'done', result: 'Stored result.' }]
    openBoard()
    fireEvent.click(await screen.findByRole('button', { name: /new task in done/i }))
    fireEvent.change(await screen.findByPlaceholderText('Title'), { target: { value: 'New task' } })

    const create = screen.getByRole('button', { name: /create task/i }) as HTMLButtonElement
    expect(create.disabled).toBe(true)
    fireEvent.change(screen.getByRole('textbox', { name: /result/i }), { target: { value: 'Completed manually.' } })
    fireEvent.click(create)

    await waitFor(() => {
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_new',
        expect.objectContaining({
          method: 'PATCH',
          body: { status: 'done', result: 'Completed manually.', summary: 'Completed manually.' }
        })
      )
    })
  })

  it('does not create a task when create-to-Done is cancelled', async () => {
    board.columns[1].tasks = [{ id: 't_done', title: 'Already done', status: 'done', result: 'Stored result.' }]
    openBoard()
    fireEvent.click(await screen.findByRole('button', { name: /new task in done/i }))
    fireEvent.change(await screen.findByPlaceholderText('Title'), { target: { value: 'New task' } })
    fireEvent.change(screen.getByRole('textbox', { name: /result/i }), { target: { value: 'Completed manually.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(rest.mock.calls.some(([path, options]) => path === '/tasks' && options?.method === 'POST')).toBe(false)
  })

  it('retries a failed create-to-Done completion without creating a duplicate task', async () => {
    board.columns[1].tasks = [{ id: 't_done', title: 'Already done', status: 'done', result: 'Stored result.' }]
    openBoard()
    fireEvent.click(await screen.findByRole('button', { name: /new task in done/i }))
    fireEvent.change(await screen.findByPlaceholderText('Title'), { target: { value: 'New task' } })
    fireEvent.change(screen.getByRole('textbox', { name: /result/i }), { target: { value: 'Completed manually.' } })
    failNextCompletion = true
    fireEvent.click(screen.getByRole('button', { name: /create task/i }))

    expect(await screen.findByText(/task was created but could not be moved to done/i)).toBeTruthy()
    expect(rest.mock.calls.filter(([path, options]) => path === '/tasks' && options?.method === 'POST')).toHaveLength(1)

    fireEvent.click(screen.getByRole('button', { name: /retry completion/i }))

    await waitFor(() => {
      expect(rest.mock.calls.filter(([path, options]) => path === '/tasks' && options?.method === 'POST')).toHaveLength(
        1
      )
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_new',
        expect.objectContaining({
          method: 'PATCH',
          body: { status: 'done', result: 'Completed manually.', summary: 'Completed manually.' }
        })
      )
    })
  })

  it('completes a mixed bulk selection without replacing stored results', async () => {
    board.columns[0].tasks = [
      { id: 't_existing', title: 'Existing result task', status: 'todo', result: 'Original result.' },
      { id: 't_empty', title: 'Empty result task', status: 'todo', result: null },
      { id: 't_review', title: 'Review task', status: 'review', result: null }
    ]
    openBoard()
    const existing = await screen.findByText('Existing result task')
    const empty = await screen.findByText('Empty result task')
    const review = await screen.findByText('Review task')
    fireEvent.click(existing.closest('[draggable="true"]')!, { ctrlKey: true })
    fireEvent.click(empty.closest('[draggable="true"]')!, { ctrlKey: true })
    fireEvent.click(review.closest('[draggable="true"]')!, { ctrlKey: true })
    fireEvent.pointerDown(await screen.findByRole('button', { name: 'Move to' }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Done' }))

    await screen.findByRole('textbox', { name: /result/i })
    expect(screen.getByText(/used for selected tasks that do not already have a result/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('textbox', { name: /result/i })).toBeNull())
    expect(rest.mock.calls.some(([path, options]) => path.startsWith('/tasks/') && options?.method === 'PATCH')).toBe(
      false
    )

    fireEvent.pointerDown(await screen.findByRole('button', { name: 'Move to' }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Done' }))
    fireEvent.change(await screen.findByRole('textbox', { name: /result/i }), {
      target: { value: 'Batch confirmation.' }
    })
    fireEvent.click(screen.getByRole('button', { name: /complete task/i }))

    await waitFor(() => {
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_existing',
        expect.objectContaining({ method: 'PATCH', body: { status: 'done', result: 'Original result.' } })
      )
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_empty',
        expect.objectContaining({
          method: 'PATCH',
          body: { status: 'done', result: 'Batch confirmation.', summary: 'Batch confirmation.' }
        })
      )
      expect(rest).toHaveBeenCalledWith(
        '/tasks/t_review',
        expect.objectContaining({ method: 'PATCH', body: { status: 'done' } })
      )
    })
  })
})
