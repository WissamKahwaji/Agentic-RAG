---
paths:
  - "frontend/**"
---

# Frontend conventions

- This is Next.js 16 with Cache Components enabled. APIs differ from older versions: read the
  relevant guide in `frontend/node_modules/next/dist/docs/` before using a Next.js API.
- The browser never calls FastAPI directly. Add a Route Handler under `src/app/api/` that uses
  `forward()` from `src/lib/backend.ts`; the backend URL stays server-side (`BACKEND_URL`).
- `src/lib/types.ts` mirrors the backend's Pydantic response models. Change both in the same
  commit when an API shape changes.
- Server Components by default; add `"use client"` only to components that need state, effects
  or event handlers.
- Styling uses the design tokens in `src/app/globals.css` (`fog`, `paper`, `ink`, `slate`, `rule`,
  `cobalt`, `highlighter`, `rust`; fonts `display`, `sans`, `mono`). No raw hex values in
  components. The highlighter is reserved for "this passage is being inspected".
- Every interactive element is reachable and operable by keyboard with a visible focus ring, and
  motion respects `prefers-reduced-motion`.
- Pure logic (parsing, formatting) lives in `src/lib/` with Vitest tests next to it.
- Before calling UI work done: `make web-check`, then look at it in a browser at desktop and
  375px widths, including loading, error and empty states.
