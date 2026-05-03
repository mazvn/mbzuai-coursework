# Collaborative Document Editor

**Task:** Team project over the semester. Build a Google Docs style editor where people edit the same document live, with sharing and roles, comments, version history and an AI writing assistant inside the editor.

**What we did:** A pnpm monorepo with 4 services:
- `apps/web`: React and Vite with a Tiptap editor
- `apps/api`: Express and Prisma on Postgres, for auth, documents, permissions, invites, comments, versions, audit logs and exports
- `apps/realtime`: Socket.IO and Yjs for live editing, cursors and presence
- `apps/ai-service`: runs AI jobs (rewrite, summarize etc.) on a local model through LM Studio and streams the result back

Shared Zod schemas and types are in `packages/contracts`, so the frontend and backend use the same shapes. Auth uses a short-lived access token and a refresh token cookie. AI suggestions are never applied directly. The user accepts or rejects them and that gets logged.

## Running

It needs Node, pnpm and Docker. Create a `.env` from `.env.example` in each app folder, then:

```
pnpm install
./run.sh
```

`run.sh` starts Postgres, runs the migrations and starts all services. The web app is on http://localhost:5173. `JWT_SECRET` and `REALTIME_INTERNAL_SECRET` must be the same in the API and realtime configs.

Tests: `pnpm test:unit`, `pnpm test:integration` and `pnpm test:e2e` (Playwright).
