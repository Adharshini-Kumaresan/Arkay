# Security: exposed browser-side AI API credential

## Finding

`src/App.tsx` initializes `GoogleGenAI` with a literal API key in browser-delivered code. The `vite.config.ts` configuration also maps `GEMINI_API_KEY` to a client-side `define` value. Neither approach keeps the key secret. Treat the current credential as compromised even if the key is later deleted from Git history.

## Immediate action

1. Revoke and rotate the exposed credential in the relevant Google console, and review usage and billing for abuse.
2. Remove the literal key from `src/App.tsx` and the `process.env.GEMINI_API_KEY` client define in `vite.config.ts`. Moving the key to `VITE_*` variables is **not** a security fix.
3. Move all Google AI calls behind a server-side API endpoint, with the secret loaded only from a server environment variable or secrets manager.
4. Validate requests, implement authentication and server-side rate limiting, set usage limits, and avoid logging patient information.
5. Update the React client to call the server endpoint rather than invoking the provider with browser-visible credentials.
6. Verify the built frontend contains no API keys; test the prediction flow end-to-end after migration.

## Verification checklist

- [ ] Old credential revoked and provider usage reviewed
- [ ] No client-visible credential in source or build output
- [ ] AI requests routed through an authenticated, rate-limited backend
- [ ] Privacy and consent reviewed for patient-related data
- [ ] Functional tests pass after migration

**Scope:** This document records the issue and remediation steps; it does not itself remove the credential or secure the application. Do not deploy the affected code publicly before remediation.
