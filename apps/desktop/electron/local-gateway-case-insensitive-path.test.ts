import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { mintLocalGatewayTicket } from './local-gateway'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

test.skipIf(process.platform === 'win32')(
  'macOS accepts case-only profile path differences without changing profile identity',
  async () => {
    const home = await shortSocketTmpDir('desktop-case-profile-')
    const socketPath = path.join(home, 'gateway.sock')

    const endpoint = {
      profile_id: home,
      instance_id: 'owner',
      authority_epoch: 1,
      runtime_protocol: 1,
      api_origin: 'http://127.0.0.1:1234',
      capabilities: ['session-authority-v1'],
      supervisor: 'none'
    }

    const requests: unknown[] = []

    const server = net.createServer(socket =>
      socket.once('data', chunk => {
        requests.push(JSON.parse(chunk.toString()))
        socket.end(
          JSON.stringify({
            protocol: 1,
            id: 1,
            ok: true,
            result: { profile_id: home, instance_id: 'owner', ticket: 'grant' }
          }) + '\n'
        )
      })
    )

    const platform = Object.getOwnPropertyDescriptor(process, 'platform')!

    const realpath = vi.spyOn(fs, 'realpath').mockResolvedValue(home.toUpperCase())

    Object.defineProperty(process, 'platform', { value: 'darwin' })

    let listening = false

    try {
      await new Promise<void>(resolve => server.listen(socketPath, resolve))
      listening = true
      await fs.chmod(socketPath, 0o600)
      await expect(mintLocalGatewayTicket(endpoint)).resolves.toBe('grant')
      expect(requests).toEqual([
        {
          protocol: 1,
          id: 1,
          verb: 'session-ticket',
          params: {
            profile_id: home,
            instance_id: 'owner',
            purpose: 'interactive',
            scope: 'host'
          }
        }
      ])
    } finally {
      Object.defineProperty(process, 'platform', platform)
      realpath.mockRestore()

      if (listening) {
        await new Promise<void>(resolve => server.close(() => resolve()))
      }

      await fs.rm(home, { recursive: true, force: true })
    }
  }
)

test.skipIf(process.platform === 'win32')(
  'macOS rejects canonical profile path differences beyond letter casing',
  async () => {
    const home = await shortSocketTmpDir('desktop-case-other-')

    const endpoint = {
      profile_id: home,
      instance_id: 'owner',
      authority_epoch: 1,
      runtime_protocol: 1,
      api_origin: 'http://127.0.0.1:1234',
      capabilities: ['session-authority-v1'],
      supervisor: 'none'
    }

    const platform = Object.getOwnPropertyDescriptor(process, 'platform')!

    const realpath = vi.spyOn(fs, 'realpath').mockResolvedValue(path.join(home, 'other'))

    Object.defineProperty(process, 'platform', { value: 'darwin' })

    try {
      await expect(mintLocalGatewayTicket(endpoint)).rejects.toThrow('Noncanonical gateway profile')
    } finally {
      Object.defineProperty(process, 'platform', platform)
      realpath.mockRestore()
      await fs.rm(home, { recursive: true, force: true })
    }
  }
)

test.skipIf(process.platform === 'win32')(
  'macOS still rejects symlinked profile paths with case-only realpath differences',
  async () => {
    const root = await shortSocketTmpDir('desktop-case-symlink-')
    const home = path.join(root, 'home')
    const alias = path.join(root, 'alias')
    await fs.mkdir(home)
    await fs.symlink(home, alias)

    const endpoint = {
      profile_id: alias,
      instance_id: 'owner',
      authority_epoch: 1,
      runtime_protocol: 1,
      api_origin: 'http://127.0.0.1:1234',
      capabilities: ['session-authority-v1'],
      supervisor: 'none'
    }

    const platform = Object.getOwnPropertyDescriptor(process, 'platform')!

    const realpath = vi.spyOn(fs, 'realpath').mockResolvedValue(alias.toUpperCase())

    Object.defineProperty(process, 'platform', { value: 'darwin' })

    try {
      await expect(mintLocalGatewayTicket(endpoint)).rejects.toThrow('Noncanonical gateway profile')
    } finally {
      Object.defineProperty(process, 'platform', platform)
      realpath.mockRestore()
      await fs.rm(root, { recursive: true, force: true })
    }
  }
)
