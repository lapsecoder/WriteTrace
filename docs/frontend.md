# WriteTrace Frontend

The Astro frontend for WriteTrace. It is a **document-level analysis workspace**
built around a paper-and-forensics editorial aesthetic, and it renders only what
the backend actually returns.

- Stack: **Astro 7** (static output), **TypeScript 5.9**, **Tailwind CSS 4**.
- No UI framework: the editing + results flow is driven by one small,
  framework-free controller (`src/lib/workspace.ts`) and one isomorphic
  renderer (`src/lib/render-result.ts`). Total shipped JavaScript is ~20 KB.
- **No remote assets.** Typography uses system font stacks; the paper grain is
  an inline SVG turbulence filter; there are no web-font, image, or script
  requests at runtime.

## What it does

1. The user writes or pastes text into a large, accessible editor with live
   word/character counts.
2. On submit, the page validates locally (mirroring the backend rules) and then
   calls `POST /api/v1/analyze`.
3. The real response is rendered as an editorial "reading": label, calibrated
   probabilities, decision, confidence, evidence state, warnings, provenance and
   the signals the backend returned.

## What it deliberately does NOT do

- **No segment/paragraph verdicts.** `AnalyzeResponse.segments` is always
  `[]`. Phase 5A (`docs/experiments/phase5a-report.md`) trained and evaluated a
  dedicated segment model and found segment scores unreliable outside the
  training distribution (OOD segment AUC near chance; short segments
  over-confident). No paragraph highlighting or per-paragraph authorship is
  shown or implied.
- **No fabricated data.** Probabilities, confidences, signals and warnings are
  rendered verbatim from the API. Nothing is simulated, sampled, or delayed.
- **No tracking.** No analytics, telemetry, cookies, `localStorage`,
  `sessionStorage`, or third-party API calls. Submitted text lives only in the
  page's memory and is sent to the configured backend.

## Configuration

| Variable | Scope | Default | Meaning |
| -------- | ----- | ------- | ------- |
| `PUBLIC_WRITETRACE_API_BASE` | build-time (client-visible) | *(unset → same-origin)* | Backend **origin**. Endpoint paths are always the full `/api/v1/...`, so the default request is `/api/v1/analyze`. A trailing `/api` is accepted and normalised away, so `/api`, `""` and unset are equivalent. Set e.g. `http://127.0.0.1:8000` to call a backend on another origin. |
| `WRITETRACE_BACKEND_ORIGIN` | dev/preview server only | `http://127.0.0.1:8000` | Target the local `/api` proxy forwards to. |

### URL composition

One rule: **the endpoint path is always the full `/api/v1/...` path and
`PUBLIC_WRITETRACE_API_BASE` is the origin it is appended to.** A trailing
`/api` on the base is stripped, so the `/api` segment can never be duplicated
(this previously produced `/api/api/v1/...`, which 404s).

| `PUBLIC_WRITETRACE_API_BASE` | Resulting request |
| ---------------------------- | ----------------- |
| *(unset)*                    | `/api/v1/analyze` |
| `""`                         | `/api/v1/analyze` |
| `/api`                       | `/api/v1/analyze` |
| `http://127.0.0.1:8000`      | `http://127.0.0.1:8000/api/v1/analyze` |
| `http://127.0.0.1:8000/api`  | `http://127.0.0.1:8000/api/v1/analyze` |

Both supported deployments share these path semantics: the Astro dev/preview
proxy forwards `/api/v1/x` to the backend **without** stripping the prefix, and
a same-origin host routes `/api/v1/x` to the API alongside the static site.
No deployment URL is ever hardcoded in the client bundle.

No secret is required or exposed: the frontend is a static page and the backend
is local, unauthenticated and stateless.

### CORS and local development

The recommended local setup avoids CORS entirely by letting the dev/preview
server proxy same-origin `/api` requests to the local FastAPI server. This is
configured in `astro.config.mjs`. If you instead point the built frontend at an
absolute backend URL that differs from the page origin, the **backend** must
allow that origin — WriteTrace deliberately ships **no permissive CORS policy**,
so add the specific origin only if you need it (do not use `*`).

## Commands

```bash
# from the repository root
cd frontend
npm install

npm run dev        # dev server on :4321 (proxies /api -> :8000)
npm run build      # type-check (astro check) + production build into dist/
npm run preview    # serve dist/ with the same /api proxy
npm run typecheck  # tsc --noEmit
npm run check      # astro check
npm test           # vitest (59 tests)
```

Start the backend in a second terminal:

```bash
# repository root
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Architecture

```
frontend/
├─ astro.config.mjs           # static build, Tailwind plugin, dev/preview /api proxy
├─ vitest.config.ts           # jsdom unit tests
├─ public/favicon.svg         # inline SVG mark (no network)
├─ src/
│  ├─ layouts/BaseLayout.astro
│  ├─ pages/index.astro       # the workspace page (static shell + island)
│  ├─ components/             # Masthead, HealthNote, ProbabilityBar,
│  │                          # SignalList, ResultsPanel, Disclaimer
│  ├─ lib/
│  │  ├─ types.ts             # TypeScript mirrors of backend/schemas.py
│  │  ├─ api.ts               # fetch client; typed errors; no inference
│  │  ├─ validate.ts          # local validation mirroring the backend
│  │  ├─ format.ts            # presentation-only formatting helpers
│  │  ├─ render-result.ts     # isomorphic result renderer
│  │  └─ workspace.ts         # the analysis controller (the only client JS)
│  └─ styles/global.css       # paper design system (Tailwind + custom layer)
└─ tests/                     # vitest suites + real backend fixtures
```

`types.ts` is the single source of the client-side contract and is hand-synced
with `backend/schemas.py` / `docs/api.md`.

## Tests

`npm test` runs 59 unit tests covering: formatting (no fabricated values),
local validation, the API client (endpoint, payload, 422 / network / malformed
handling), the controller (counts, validation, verbatim submission, duplicate
submission prevention, text preservation on failure, re-analysis), and the
result renderer (normal / abstained / mixed states, null confidence, signal
rendering, XSS escaping, and that no segment verdicts appear).

Fixtures in `tests/fixtures/real-responses.json` were captured from the real
backend (`final-v1`) so the UI is tested against actual responses. Regenerate
with the backend running if the contract changes.

## Accessibility & responsiveness

- Semantic landmarks, a single `h1`, ordered headings, a skip link, associated
  labels, `aria-live` status regions and `role="status"`/`role="alert"` for async
  updates.
- Visible `:focus-visible` outlines in the accent color; full keyboard
  operation.
- All motion (entrance, stamp reveal, ink spread) is restrained and disabled
  under `prefers-reduced-motion`.
- Layout is fluid from mobile to desktop with no fixed pixel widths; the editor
  and results remain usable on narrow screens.

## Design notes

Warm off-white paper, ink-black serif headings, monospaced micro-labels,
graphite rules and a single muted red accent (used only for the "machine"
direction and annotations). Probability is shown as a labelled two-segment bar
with text values and an `aria-label`, so the information never depends on color
alone. Signals read as margin annotations, never as proof.