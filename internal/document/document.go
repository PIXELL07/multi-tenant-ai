package document

import (
	"context"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pixell07/multi-tenant-ai/internal/embedding"
	"github.com/pixell07/multi-tenant-ai/internal/retrieval"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

type Status string

const (
	StatusPending    Status = "pending"
	StatusProcessing Status = "processing"
	StatusReady      Status = "ready"
	StatusFailed     Status = "failed"
)

type Document struct {
	ID         string    `json:"id"`
	OrgID      string    `json:"org_id"`
	Name       string    `json:"name"`
	Content    string    `json:"-"` // raw text, not exposed in listings
	Status     Status    `json:"status"`
	ChunkCount int       `json:"chunk_count"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type Repository struct {
	db *pgxpool.Pool
}

func NewRepository(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

func (r *Repository) Create(ctx context.Context, doc *Document) error {
	_, err := r.db.Exec(ctx,
		`INSERT INTO documents (id, org_id, name, content, status, chunk_count, created_at, updated_at)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
		doc.ID, doc.OrgID, doc.Name, doc.Content, doc.Status,
		doc.ChunkCount, doc.CreatedAt, doc.UpdatedAt,
	)
	return err
}

func (r *Repository) UpdateStatus(ctx context.Context, id string, status Status, chunkCount int) error {
	_, err := r.db.Exec(ctx,
		`UPDATE documents SET status=$1, chunk_count=$2, updated_at=$3 WHERE id=$4`,
		status, chunkCount, time.Now(), id,
	)
	return err
}

func (r *Repository) ListByOrg(ctx context.Context, orgID string) ([]*Document, error) {
	rows, err := r.db.Query(ctx,
		`SELECT id, org_id, name, status, chunk_count, created_at, updated_at
		 FROM documents WHERE org_id=$1 ORDER BY created_at DESC`,
		orgID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []*Document
	for rows.Next() {
		d := &Document{}
		if err := rows.Scan(&d.ID, &d.OrgID, &d.Name, &d.Status,
			&d.ChunkCount, &d.CreatedAt, &d.UpdatedAt); err != nil {
			return nil, err
		}
		docs = append(docs, d)
	}
	return docs, rows.Err()
}

func (r *Repository) Delete(ctx context.Context, id, orgID string) error {
	_, err := r.db.Exec(ctx,
		`DELETE FROM documents WHERE id=$1 AND org_id=$2`, id, orgID,
	)
	return err
}

func splitDocument(doc *Document) ([]schema.Document, error) {
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(512),
		textsplitter.WithChunkOverlap(64),
	)

	// CreateDocuments handles splitting + metadata attachment in one call
	return textsplitter.CreateDocuments(
		splitter,
		[]string{doc.Content},
		[]map[string]any{
			{
				"org_id":      doc.OrgID,
				"document_id": doc.ID,
				"doc_name":    doc.Name,
			},
		},
	)
}

type Service struct {
	repo        *Repository
	vectorStore *retrieval.LangChainVectorStore
	embedder    embedding.Embedder
	// Buffered channel acts as an in-process job queue.
	// In production replace with Redis Streams / SQS / NATS.
	jobs chan ingestJob
}

type ingestJob struct {
	doc *Document
}

func NewService(repo *Repository, vs *retrieval.LangChainVectorStore, embedder embedding.Embedder) *Service {
	s := &Service{
		repo:        repo,
		vectorStore: vs,
		embedder:    embedder,
		jobs:        make(chan ingestJob, 256),
	}
	// Fixed pool of goroutine workers — each owns its own context and runs forever
	// for i := range 4
	// s.jobs { ... } This will NOT compile in Go
	for i := 0; i < 4; i++ {
		go s.worker(i)
	}
	return s
}

type UploadRequest struct {
	OrgID   string
	Name    string
	Content string
}

// Upload persists the document metadata and enqueues async embedding.
// Returns immediately with status="pending" so the HTTP caller isn't blocked.
func (s *Service) Upload(ctx context.Context, req UploadRequest) (*Document, error) {
	doc := &Document{
		ID:        uuid.NewString(),
		OrgID:     req.OrgID,
		Name:      req.Name,
		Content:   req.Content,
		Status:    StatusPending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.repo.Create(ctx, doc); err != nil {
		return nil, err
	}

	// Non-blocking enqueue: if the queue is full the doc stays "pending"
	// and can be retried by a background sweep (not implemented here).
	select {
	case s.jobs <- ingestJob{doc: doc}:
	default:
		slog.Warn("ingestion queue full, document queued as pending", "doc_id", doc.ID)
	}

	return doc, nil
}

func (s *Service) List(ctx context.Context, orgID string) ([]*Document, error) {
	return s.repo.ListByOrg(ctx, orgID)
}

func (s *Service) Delete(ctx context.Context, id, orgID string) error {
	if err := s.vectorStore.DeleteByDocument(ctx, id); err != nil {
		return err
	}
	return s.repo.Delete(ctx, id, orgID)
}

// worker is the goroutine that consumes ingest jobs.
func (s *Service) worker(id int) {
	slog.Info("ingestion worker started", "worker_id", id)
	for job := range s.jobs {
		s.ingest(job.doc)
	}
}

// ingest is the full pipeline for one document:
//  1. langchaingo textsplitter → []schema.Document (chunks with metadata)
//  2. langchaingo pgvector store → AddDocuments (embed + store in one call)
func (s *Service) ingest(doc *Document) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if err := s.repo.UpdateStatus(ctx, doc.ID, StatusProcessing, 0); err != nil {
		slog.Error("status update failed", "doc_id", doc.ID, "error", err)
		return
	}

	// S1: Split with langchaingo RecursiveCharacter splitter
	chunks, err := splitDocument(doc)
	if err != nil || len(chunks) == 0 {
		slog.Error("text splitting failed", "doc_id", doc.ID, "error", err)
		_ = s.repo.UpdateStatus(ctx, doc.ID, StatusFailed, 0)
		return
	}

	// S2: AddDocuments via langchaingo pgvector store
	// langchaingo handles batching and embedding internally.
	if err := s.vectorStore.AddDocuments(ctx, chunks); err != nil {
		slog.Error("vector store add failed", "doc_id", doc.ID, "error", err)
		_ = s.repo.UpdateStatus(ctx, doc.ID, StatusFailed, 0)
		return
	}

	if err := s.repo.UpdateStatus(ctx, doc.ID, StatusReady, len(chunks)); err != nil {
		slog.Error("status update to ready failed", "doc_id", doc.ID, "error", err)
	}

	slog.Info("document ingested", "doc_id", doc.ID, "chunks", len(chunks))
}
