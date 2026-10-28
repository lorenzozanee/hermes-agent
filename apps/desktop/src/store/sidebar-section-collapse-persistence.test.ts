import { beforeEach, describe, expect, it, vi } from 'vitest'

async function loadLayout() {
  return await import('./layout')
}

describe('sidebar section collapse persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('persists Pinned collapse independently across reloads', async () => {
    const initial = await loadLayout()

    expect(initial.$sidebarPinsOpen.get()).toBe(true)
    expect(initial.$sidebarRecentsOpen.get()).toBe(true)

    initial.setSidebarPinsOpen(false)

    vi.resetModules()
    const reloaded = await loadLayout()

    expect(reloaded.$sidebarPinsOpen.get()).toBe(false)
    expect(reloaded.$sidebarRecentsOpen.get()).toBe(true)
  })

  it('persists Sessions collapse independently across reloads', async () => {
    const initial = await loadLayout()

    expect(initial.$sidebarPinsOpen.get()).toBe(true)
    expect(initial.$sidebarRecentsOpen.get()).toBe(true)

    initial.setSidebarRecentsOpen(false)

    vi.resetModules()
    const reloaded = await loadLayout()

    expect(reloaded.$sidebarPinsOpen.get()).toBe(true)
    expect(reloaded.$sidebarRecentsOpen.get()).toBe(false)
  })
})
