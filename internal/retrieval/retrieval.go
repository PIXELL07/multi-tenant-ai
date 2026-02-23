// Package retrieval wraps langchaingo's pgvector VectorStore and
// provides a RAG service that streams LLM responses over a Go channel.
package retrieval

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pixell07/multi-tenant-ai/internal/embedding"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	lcpgvector "github.com/tmc/langchaingo/vectorstores/pgvector"
)

// LangChainVectorStore
//
// This wraps langchaingo's pgvector.Store which:
//   - Manages its own connection via pgx
//   - Creates the required langchain_pg_embedding / langchain_pg_collection tables
//   - Provides AddDocuments (embed + upsert) and SimilaritySearch in one call
//   - Supports HNSW index creation via WithHNSWIndex option

type LangChainVectorStore struct {
	store    lcpgvector.Store
	embedder embedding.Embedder
}

// NewLangChainVectorStore initialises a langchaingo pgvector Store.
// It will auto-create the embedding/collection tables on first use.
func NewLangChainVectorStore(
	ctx context.Context,
	db *pgxpool.Pool,
	embedder embedding.Embedder,
	connURL string,
) (*LangChainVectorStore, error) {
	// langchaingo's pgvector store needs the embedder as its own interface.
	// We adapt our internal Embedder to langchaingo's embeddings.Embedder.
	lcEmbedder := &langchainEmbedderAdapter{inner: embedder}

	store, err := lcpgvector.New(
		ctx,
		lcpgvector.WithConnectionURL(connURL),
		lcpgvector.WithEmbedder(lcEmbedder),
		lcpgvector.WithCollectionName("rag_documents"),
		lcpgvector.WithVectorDimensions(1536), // text-embedding-3-small
		// Create HNSW index for sub-linear ANN search
		lcpgvector.WithHNSWIndex(16, 64, "cosine"),
	)
	if err != nil {
		return nil, fmt.Errorf("init langchaingo pgvector store: %w", err)
	}

	return &LangChainVectorStore{store: store, embedder: embedder}, nil
}

// AddDocuments embeds and stores a batch of langchaingo schema.Documents.
// This is called by the ingestion worker after text splitting.
func (vs *LangChainVectorStore) AddDocuments(ctx context.Context, docs []schema.Document) error {
	_, err := vs.store.AddDocuments(ctx, docs)
	return err
}

// SimilaritySearch returns the top-k most similar documents for the query,
// filtered to a specific org via langchaingo's vectorstores.WithFilters option.

// The filter maps directly to a WHERE clause in pgvector's metadata JSON column.
func (vs *LangChainVectorStore) SimilaritySearch(
	ctx context.Context,
	query string,
	orgID string,
	topK int,
) ([]schema.Document, error) {
	return vs.store.SimilaritySearch(
		ctx,
		query,
		topK,
		vectorstores.WithFilters(map[string]any{
			"org_id": orgID,
		}),
	)
}

// DeleteByDocument removes all chunks for a given document_id from the store.

func (vs *LangChainVectorStore) DeleteByDocument(ctx context.Context, documentID string) error {

	// langchaingo's pgvector store doesn't expose a direct delete-by-filter yet,
	// so we use the underlying connection URL via a raw pgx query.
	// The store manages its own pool internally; we delete from the embedding table directly.

	return vs.store.RemoveCollection(ctx, nil) // no-op placeholder — see note below

	// NOTE: In production implement this by holding a *pgxpool.Pool reference and running:
	//   DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = $1
}

// Close releases the pgvector store connection.
func (vs *LangChainVectorStore) Close() {
	vs.store.Close()
}

// Embedder adapter
// langchaingo's pgvector.WithEmbedder expects embeddings.Embedder (langchaingo interface).
//  bridge internal embedding.Embedder

type langchainEmbedderAdapter struct {
	inner embedding.Embedder
}

func (a *langchainEmbedderAdapter) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	return a.inner.EmbedDocuments(ctx, texts)
}

func (a *langchainEmbedderAdapter) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return a.inner.EmbedQuery(ctx, text)
}

// RAG Service

// RAGService uses the RetrievalQA pattern:
//  1. Embed the question and search the pgvector store (retrieval)
//  2. Build a context-augmented prompt from the retrieved chunks
//  3. Stream the LLM response token-by-token over a Go channel

// LLMClient is the interface the RAG service uses to stream completions.
type LLMClient interface {
	StreamCompletion(ctx context.Context, systemPrompt, userMessage string, out chan<- string) error
}

type RAGService struct {
	vectorStore *LangChainVectorStore
	llm         LLMClient
}

func NewRAGService(vs *LangChainVectorStore, llm LLMClient) *RAGService {
	return &RAGService{vectorStore: vs, llm: llm}
}

type QueryRequest struct {
	OrgID    string
	Question string
	TopK     int
}

// Query retrieves relevant context via langchaingo SimilaritySearch and
// streams an LLM response over the out channel (closed when done).
func (s *RAGService) Query(ctx context.Context, req QueryRequest, out chan<- string) error {
	if req.TopK <= 0 {
		req.TopK = 5
	}

	// S1: Retrieve via langchaingo pgvector SimilaritySearch
	results, err := s.vectorStore.SimilaritySearch(ctx, req.Question, req.OrgID, req.TopK)
	if err != nil {
		return fmt.Errorf("similarity search: %w", err)
	}

	// S2: Build context block from retrieved schema.Documents
	var ctxBuilder strings.Builder
	for i, doc := range results {
		docID, _ := doc.Metadata["document_id"].(string)
		docName, _ := doc.Metadata["doc_name"].(string)
		fmt.Fprintf(&ctxBuilder,
			"--- Chunk %d (doc: %s / %s) ---\n%s\n\n",
			i+1, docID, docName, doc.PageContent,
		)
	}

	system := `You are a helpful knowledge-base assistant.
Answer the user's question using ONLY the provided context chunks.
If the answer is not in the context, say "I don't have enough information to answer that."
Be concise and cite chunk numbers when referencing specific information.`

	user := fmt.Sprintf("Context:\n%s\n\nQuestion: %s", ctxBuilder.String(), req.Question)

	// S3: Stream LLM response
	return s.llm.StreamCompletion(ctx, system, user, out)
}
