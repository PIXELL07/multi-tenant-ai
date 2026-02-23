// As we know Package embedding wraps langchaingo's embeddings.Embedder so the rest of the
// code can depend on a clean interface instead of the langchaingo type directly.
package embedding

import (
	"context"

	"github.com/tmc/langchaingo/embeddings"
	lcopenai "github.com/tmc/langchaingo/llms/openai"
)

// Embedder is the interface the rest of the app depends on.
type Embedder interface {
	EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// LangChainEmbedder wraps langchaingo's embeddings.EmbedderImpl.
type LangChainEmbedder struct {
	inner *embeddings.EmbedderImpl
}

// NewOpenAIEmbedder creates a new embedder backed by OpenAI's
// text-embedding-3-small model via langchaingo.
func NewOpenAIEmbedder(apiKey string) (*LangChainEmbedder, error) {
	// langchaingo's openai.New() reads OPENAI_API_KEY automatically;
	// WithToken lets us pass it explicitly so callers don't have to set env vars.
	llm, err := lcopenai.New(
		lcopenai.WithToken(apiKey),
		lcopenai.WithEmbeddingModel("text-embedding-3-small"),
	)
	if err != nil {
		return nil, err
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		return nil, err
	}

	return &LangChainEmbedder{inner: embedder}, nil
}

// EmbedDocuments embeds a batch of texts.
func (e *LangChainEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	return e.inner.EmbedDocuments(ctx, texts)
}

// EmbedQuery embeds a single query string.
func (e *LangChainEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return e.inner.EmbedQuery(ctx, text)
}
