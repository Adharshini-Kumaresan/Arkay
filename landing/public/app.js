const canvas = document.querySelector('#letters')
const context = canvas.getContext('2d')
const motion = document.querySelector('#motion')
const preference = matchMedia('(prefers-reduced-motion: reduce)')
const alphabet = document.querySelector('#alphabet')
const phonetics = Array.from('ɑɹkeɪʃəɛθʊŋʒæðɪʌɡ')
const letters = Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
let symbols = phonetics
let width, height, particles, spacing, frame, lastTime = 0, elapsed = 0
let paused = false
let ripple = null

function paint(time) {
  if (lastTime) elapsed += Math.min(time - lastTime, 60)
  lastTime = time
  const phase = elapsed * .00055
  context.clearRect(0, 0, width, height)
  for (const {x, y, seed} of particles) {
    const tileX = x / spacing
    const tileY = y / spacing
    let field = Math.sin(tileX * .32 + tileY * .19 - phase)
      + Math.cos(tileY * .49 - tileX * .08 + phase * 1.4)
      + Math.sin((tileX - tileY) * .17 + phase * .64)
    if (ripple) {
      const age = elapsed - ripple.time
      const delta = Math.hypot(x - ripple.x, y - ripple.y) - age * .32
      field += Math.max(0, 1 - Math.abs(delta) / 70) * 2.3
      if (age > 4000) ripple = null
    }
    const bright = field > .18
    context.fillStyle = bright ? '#f597e0' : '#200d3d'
    context.fillRect(x, y, spacing, spacing)
    const index = (seed + Math.floor(elapsed / 900 + field)) % symbols.length
    context.fillStyle = bright ? '#200d3d' : '#f597e0'
    context.fillText(symbols[(index + symbols.length) % symbols.length], x + spacing / 2, y + spacing / 2)
  }
  if (!paused && !preference.matches) frame = requestAnimationFrame(paint)
}

function refresh() {
  cancelAnimationFrame(frame)
  lastTime = 0
  motion.hidden = preference.matches
  motion.textContent = paused ? 'Resume motion' : 'Pause motion'
  paint(0)
}

function resize() {
  const bounds = canvas.getBoundingClientRect()
  width = bounds.width
  height = bounds.height
  const scale = Math.min(devicePixelRatio || 1, 2)
  canvas.width = Math.round(width * scale)
  canvas.height = Math.round(height * scale)
  context.setTransform(scale, 0, 0, scale, 0, 0)
  spacing = width < 650 ? 18 : 24
  context.font = (spacing * .56) + 'px monospace'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  particles = []
  for (let y = 0; y < height; y += spacing) {
    for (let x = 0; x < width; x += spacing) {
      particles.push({x, y, seed: Math.floor(Math.random() * 260)})
    }
  }
  refresh()
}

if (context) {
  new ResizeObserver(resize).observe(canvas)
  preference.addEventListener('change', refresh)
  motion.addEventListener('click', () => { paused = !paused; refresh() })
  alphabet.hidden = false
  alphabet.addEventListener('click', () => {
    symbols = symbols === phonetics ? letters : phonetics
    alphabet.textContent = symbols === phonetics ? 'Show letters' : 'Show IPA'
    refresh()
  })
  canvas.parentElement.addEventListener('click', event => {
    if (event.target !== canvas.parentElement || paused || preference.matches) return
    const bounds = canvas.getBoundingClientRect()
    ripple = {x: event.clientX - bounds.left, y: event.clientY - bounds.top, time: elapsed}
  })
}

const form = document.querySelector('#request')
const notice = document.querySelector('#notice')
const submit = form.querySelector('button')

form.addEventListener('submit', async event => {
  event.preventDefault()
  submit.disabled = true
  notice.dataset.state = ''
  notice.textContent = 'Sending your request…'
  try {
    const response = await fetch(form.action, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
    })
    if (!response.ok) throw new Error('Request failed')
    notice.dataset.state = 'success'
    notice.textContent = 'Demo request received. The Arkay team will follow up by email to schedule a time.'
    form.reset()
  } catch {
    notice.dataset.state = 'error'
    notice.textContent = 'We couldn’t send your request. Please try again.'
  } finally {
    submit.disabled = false
  }
})

const originStage = document.querySelector('.origin-stage')
const chapters = Array.from(document.querySelectorAll('.origin-chapters article'))
let pendingScene = false
function updateOrigin() {
  const center = window.innerHeight * .63
  const chapter = chapters.find(card => card.getBoundingClientRect().bottom > center) ?? chapters.at(-1)
  originStage.dataset.scene = chapter.dataset.scene
  pendingScene = false
}
window.addEventListener('scroll', () => {
  if (pendingScene) return
  pendingScene = true
  requestAnimationFrame(updateOrigin)
}, {passive:true})
window.addEventListener('resize', updateOrigin)
originStage.addEventListener('pointermove', event => {
  if (event.pointerType === 'touch' || preference.matches) return
  const bounds = originStage.getBoundingClientRect()
  const tilt = ((event.clientX - bounds.left) / bounds.width - .5) * 18
  originStage.style.setProperty('--tilt', tilt + 'deg')
})
originStage.addEventListener('pointerleave', () => originStage.style.removeProperty('--tilt'))
preference.addEventListener('change', () => originStage.style.removeProperty('--tilt'))
updateOrigin()
