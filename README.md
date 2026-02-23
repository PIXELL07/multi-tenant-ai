# Multi-Tenant RAG Platform

A production-grade Retrieval-Augmented Generation (RAG) backend in Go.

```
┌─────────────┐   JWT    ┌──────────────────────────────────────────────────────┐
│  HTTP Client│ ──────►  │  API Layer  (internal/api)                           │
└─────────────┘          │  • loggingMiddleware → authMiddleware → handlers     │
                         └──────────┬────────────────┬─────────────────┬────────┘
                                    │                │                 │
                         ┌──────────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
                         │ TenantService   │  │ DocService │  │  RAGService  │
                         │ register/login  │  │ upload     │  │  query (SSE) │
                         └──────────┬──────┘  └─────┬──────┘  └──────┬───────┘
                                    │               │                │
                         ┌──────────▼──────┐  ┌────▼──────────────┐  │
                         │  users/orgs     │  │ goroutine workers │  │
                         │  (PostgreSQL)   │  │ chunk→embed→store │  │
                         └─────────────────┘  └───────────────────┘  │
                                                                     │
                         ┌─────────────────────────────────────────────▼──────┐
                         │  PgVectorStore  (internal/retrieval)               │
                         │  HNSW cosine search scoped to org_id               │
                         └────────────────────────────────────────────────────┘
                                              │
                         ┌────────────────────▼──────────────────┐
                         │  OpenAI Embeddings + Chat (streaming) │
                         └───────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Copy env file
cp .env.example .env
# Edit OPENAI_API_KEY and JWT_SECRET

# 2. Start everything
docker compose -f docker/docker-compose.yml up --build

# 3. Register a new org + admin user
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"org_name":"Acme","email":"admin@acme.com","password":"secret123"}'

# → { "token": "<JWT>", "user": {...}, "org": {...} }

# 4. Upload a document (async ingestion)
curl -X POST http://localhost:8080/api/v1/documents \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"name":"Go Tour","content":"Go is an open source programming language..."}'

# 5. Stream a query (SSE)
curl -N http://localhost:8080/api/v1/query \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Go used for?","top_k":5}'

# 6. Non-streaming query (simpler for testing)
curl -X POST http://localhost:8080/api/v1/query/sync \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Go used for?"}'
```

---

## Architecture Deep-Dive

### 1. Multi-Tenancy

Every request carries a JWT with `org_id` embedded in the claims.  
The `authMiddleware` extracts and validates it, then stores the `Claims` struct in
the request context. All queries to Postgres always include `WHERE org_id = $1`,
so one tenant can **never** see another's data.

The database schema enforces this at the storage level too: both `documents` and
`document_chunks` have foreign keys back to `organizations`, and the HNSW index
search always starts with the `idx_chunks_org` B-tree index to narrow candidates
before the expensive vector scan.

### 2. Async Ingestion Pipeline

```
HTTP POST /documents
  └─ Service.Upload()
       ├─ INSERT document (status=pending)
       ├─ Enqueue ingestJob (buffered channel, cap=256)
       └─ Return 202 Accepted immediately

Worker goroutine (4 running permanently)
  └─ Dequeue job
       ├─ UPDATE status=processing
       ├─ splitIntoChunks()         ← 512-word sliding window, 64-word overlap
       ├─ Embed in batches of 100   ← OpenAI text-embedding-3-small
       ├─ UpsertVectors() in TX     ← pgvector, ON CONFLICT upsert
       └─ UPDATE status=ready
```

The buffered channel acts as an in-process queue. For production at scale,
replace it with a proper queue (Redis Streams, SQS, etc.) and a separate
worker process so you can scale them independently.

### 3. pgvector and HNSW

```sql
-- The magic query: cosine ANN search scoped to tenant
SELECT content, 1 - (embedding <=> $1::vector) AS score
FROM document_chunks
WHERE org_id = $2
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

`<=>` is pgvector's cosine distance operator.  
The HNSW index (`WITH (m=16, ef_construction=64)`) makes this O(log n) instead
of O(n) — critical for achieving sub-200ms retrieval even with millions of chunks.

Tenant isolation works because Postgres evaluates the `org_id = $2` predicate
using `idx_chunks_org` first, dramatically shrinking the candidate set before
the HNSW scan.

### 4. SSE Streaming

The `/api/v1/query` endpoint streams tokens back using Server-Sent Events:

```
Client                          Server
  │                               │
  │── POST /api/v1/query ────────►│
  │                               │── goroutine: RAGService.Query()
  │                               │     ├─ Vector search (~5-50ms)
  │                               │     └─ OpenAI stream → chan string
  │◄── data: The ─────────────────│
  │◄── data: answer ──────────────│
  │◄── data:  is ─────────────────│
  │◄── data: [DONE] ──────────────│
```

The LLM client opens an SSE connection to OpenAI, parses each `data:` line,
and forwards tokens to an internal Go channel. The HTTP handler reads from that
channel and writes `data: <token>\n\n` to the response, flushing after each
token. This gives real-time streaming with ~10ms additional latency per token.

### 5. JWT Authentication

```
Register → bcrypt(password) → INSERT user → Sign JWT(org_id, user_id, role)
Login    → bcrypt.Compare   → Sign JWT(org_id, user_id, role)

Every request → Parse Bearer token → Embed Claims in context
             → Handler reads claims.OrgID for data isolation
```

JWTs are HS256-signed with a secret from env. The `role` claim (`admin`/`member`)
can be used to gate admin-only operations like deleting documents.

---

## Project Layout

```
.
├── cmd/server/main.go          # Entry point, wiring, graceful shutdown
├── internal/
│   ├── api/router.go           # HTTP mux, middleware, all handlers
│   ├── auth/jwt.go             # JWT generation & verification
│   ├── tenant/tenant.go        # Org + user domain, repo, service
│   ├── document/document.go    # Document domain, chunking, async ingestion
│   ├── embedding/embedder.go   # Embedder interface + OpenAI implementation
│   ├── retrieval/retrieval.go  # PgVectorStore + RAGService
│   └── llm/openai.go           # OpenAI chat with SSE streaming
├── migrations/
│   └── 001_initial_schema.sql  # pgvector, HNSW index, multi-tenant tables
├── docker/
│   └── docker-compose.yml
├── Dockerfile                  # Multi-stage, distroless runtime
└── go.mod
|__ go.sum
