import { createHash } from 'node:crypto'
import { put } from '@vercel/blob'
import worker from '../lib/demo.mjs'

const DB = {
  prepare() {
    return {
      bind(name, email, organization, role) {
        return {
          async run() {
            const key = createHash('sha256').update(JSON.stringify([email, organization])).digest('hex')
            const body = JSON.stringify({ name, email, organization, role, requested_at: new Date().toISOString() })
            try {
              await put('demo-requests/' + key + '.json', body, {
                access: 'private', contentType: 'application/json', allowOverwrite: false,
              })
            } catch (error) {
              if (!/blob.*already exists/i.test(error.message)) throw error
            }
          },
        }
      },
    }
  },
}

export default async function handler(req, res) {
  const host = req.headers.host
  const headers = new Headers()
  for (const key of ['origin', 'content-type']) {
    if (typeof req.headers[key] === 'string') headers.set(key, req.headers[key])
  }
  const contentType = headers.get('content-type')
  const body = req.method === 'POST'
    ? typeof req.body === 'string' || Buffer.isBuffer(req.body) ? req.body
      : contentType?.startsWith('application/x-www-form-urlencoded') ? new URLSearchParams(req.body).toString()
      : JSON.stringify(req.body ?? {})
    : undefined
  const response = await worker.fetch(new Request('https://' + host + '/api/demo', {
    method: req.method, headers, body,
  }), { DB })
  for (const [key, value] of response.headers) res.setHeader(key, value)
  res.status(response.status).send(await response.text())
}
