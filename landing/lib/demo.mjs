const assets = {}

const headers = {
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Content-Security-Policy': "default-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'",
}

function message(text, status = 200) {
  return Response.json({message: text}, {status, headers: {...headers, 'Cache-Control': 'no-store'}})
}

async function requestDemo(request, env, origin) {
  if (request.method !== 'POST') return message('Method not allowed', 405)
  if (request.headers.get('Origin') !== origin) return message('Forbidden', 403)
  const contentType = request.headers.get('Content-Type')?.split(';')[0]
  if (!['application/json', 'application/x-www-form-urlencoded'].includes(contentType)) {
    return message('Unsupported format', 415)
  }
  let data
  try {
    const reader = request.body.getReader()
    const decoder = new TextDecoder()
    let raw = ''
    let length = 0
    for (;;) {
      const {value, done} = await reader.read()
      if (done) break
      length += value.byteLength
      if (length > 4096) {
        await reader.cancel()
        return message('Request too large', 413)
      }
      raw += decoder.decode(value, {stream: true})
    }
    raw += decoder.decode()
    data = contentType === 'application/json' ? JSON.parse(raw) : Object.fromEntries(new URLSearchParams(raw))
  } catch {
    return message('Invalid request', 400)
  }
  const field = key => typeof data?.[key] === 'string' ? data[key].trim() : ''
  const name = field('name')
  const email = field('email').toLowerCase()
  const organization = field('organization')
  const role = field('role')
  if (field('website')) return message('Invalid request', 400)
  if (!name || name.length > 100 || !organization || organization.length > 160 ||
      role.length > 100 || email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return message('Please provide a name, work email and organization.', 422)
  }
  try {
    await env.DB.prepare(
      'INSERT INTO demo_requests (name, email, organization, role) VALUES (?, ?, ?, ?) ON CONFLICT (email, organization) DO NOTHING'
    ).bind(name, email, organization, role).run()
  } catch {
    return message('The request could not be saved. Please try again.', 503)
  }
  if (contentType === 'application/x-www-form-urlencoded') {
    return new Response('<!doctype html><html lang="en"><meta charset="utf-8"><title>Demo request received</title><h1>Demo request received</h1><p>The Arkay team will follow up by email.</p><a href="/">Back to Arkay</a></html>', {
      headers: {...headers, 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store'},
    })
  }
  return message('Demo request received', 201)
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/api/demo') return requestDemo(request, env, url.origin)
    if (!['GET', 'HEAD'].includes(request.method)) return message('Method not allowed', 405)
    const asset = assets[url.pathname]
    if (!asset) return new Response('Page not found', {status: 404, headers})
    const [type, body, binary] = asset
    return new Response(request.method === 'HEAD' ? null : binary ? Uint8Array.from(atob(body), char => char.charCodeAt(0)) : body, {
      headers: {...headers, 'Content-Type': type},
    })
  },
}
