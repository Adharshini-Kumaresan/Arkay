# Security status and credential rotation

## What this branch fixes

The React client no longer embeds a Google AI API key, and Vite no longer injects `GEMINI_API_KEY` into browser JavaScript. The existing fixed duration-to-phrase mapping is now performed locally without a third-party AI request. This preserves the prototype's deterministic mapping, but **does not provide genuine lip-reading or reliable clinical interpretation**.

## Urgent account action

The previously committed Google AI credential must be treated as compromised: **revoke it now** in the relevant Google console and review usage, quotas, and billing. Removing it from current source does not remove it from Git history, caches, or previously deployed bundles. Do not reuse the same credential. If a future model is introduced, keep all provider credentials server-side behind authenticated, rate-limited endpoints.

## Validation checklist

- [x] Remove hardcoded credential and direct Google AI calls from `src/App.tsx`
- [x] Remove Vite's client-side environment-secret injection
- [x] Preserve deterministic duration mapping locally, without network calls
- [ ] Revoke exposed credential and check provider activity (account owner)
- [ ] Run `npm install && npm run lint && npm run build` and inspect output
- [ ] Verify any existing deployment is rebuilt and no longer serves the old bundle
- [ ] Confirm prototype behavior manually and review clinical safety/consent before patient use

**Clinical caution:** The duration-based phrase output is a demonstration, not speech recognition or a medical device. In particular, a phrase such as “I cannot breathe” is not inferred from patient intent. Do not rely on this prototype for clinical decisions or emergencies.
