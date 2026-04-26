# collaborative document editor

**task:** team project over the semester. build a google-docs style editor where people edit the same document live, with sharing/roles, comments, version history, and an AI writing assistant inside the editor.

**what we did:** pnpm monorepo with 4 services:
- `apps/web` - react + vite, tiptap editor
- `apps/api` - express + prisma (postgres). auth, documents, permissions, invites, comments, versions, audit log, exports
- `apps/realtime` - socket.io + yjs for the live editing, cursors and presence
- `apps/ai-service` - runs the AI jobs (rewrite, summarize etc.) on a local model through LM Studio and streams the result back

shared zod schemas/types are in `packages/contracts` so front and back use the same shapes. auth is short access token + refresh token cookie. AI suggestions are never applied directly, the user accepts/rejects them and that gets logged.

## running

needs node, pnpm and docker.

```
pnpm install
cp apps/api/.env.example apps/api/.env          # same for ai-service, realtime, web
./run.sh
```
`run.sh` starts postgres, runs the migrations and starts everything. web is on http://localhost:5173. `JWT_SECRET` and `REALTIME_INTERNAL_SECRET` need to be the same in api and realtime.

tests: `pnpm test:unit`, `pnpm test:integration`, `pnpm test:e2e` (playwright)
