package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	// need to initialize pgxpool before any other pgx imports to avoid issues with multiple versions
	// open.ai import llm and llm import pgxpool, so we need to ensure pgxpool is initialized first

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pixell07/multi-tenant-ai/internal/api"
	"github.com/pixell07/multi-tenant-ai/internal/auth"
	"github.com/pixell07/multi-tenant-ai/internal/document"
	"github.com/pixell07/multi-tenant-ai/internal/embedding"
	"github.com/pixell07/multi-tenant-ai/internal/llm" // to fix circular import with retrieval
	"github.com/pixell07/multi-tenant-ai/internal/retrieval"
	"github.com/pixell07/multi-tenant-ai/internal/tenant"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := loadConfig()
	ctx := context.Background()

	// Database connection pool
	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		slog.Error("failed to ping database", "error", err)
		os.Exit(1)
	}
	slog.Info("connected to database")

	// langchaingo OpenAI embedder
	embedder, err := embedding.NewOpenAIEmbedder(cfg.OpenAIKey)
	if err != nil {
		slog.Error("failed to create embedder", "error", err)
		os.Exit(1)
	}

	// langchaingo pgvector vector store
	vectorStore, err := retrieval.NewLangChainVectorStore(ctx, pool, embedder, cfg.DatabaseURL)
	if err != nil {
		slog.Error("failed to init vector store", "error", err)
		os.Exit(1)
	}
	defer vectorStore.Close()
	slog.Info("langchaingo pgvector store ready")

	// Wire remaining dependencies
	tenantRepo := tenant.NewRepository(pool)
	docRepo := document.NewRepository(pool)
	llmClient := llm.NewOpenAIClient(cfg.OpenAIKey, cfg.LLMModel)
	jwtManager := auth.NewJWTManager(cfg.JWTSecret, cfg.JWTExpiry)

	tenantSvc := tenant.NewService(tenantRepo, jwtManager)
	docSvc := document.NewService(docRepo, vectorStore, embedder)
	ragSvc := retrieval.NewRAGService(vectorStore, llmClient)

	// HTTP router
	router := api.NewRouter(api.RouterDeps{
		TenantService:   tenantSvc,
		DocumentService: docSvc,
		RAGService:      ragSvc,
		JWTManager:      jwtManager,
		Logger:          logger,
	})

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second, // longer for SSE streaming
		IdleTimeout:  120 * time.Second,
	}

	// Graceful shutdown
	go func() {
		slog.Info("server starting", "addr", cfg.ListenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	slog.Info("shutting down server...")
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("forced shutdown", "error", err)
	}
	slog.Info("server stopped")
}

type Config struct {
	DatabaseURL string
	OpenAIKey   string
	LLMModel    string
	JWTSecret   string
	JWTExpiry   time.Duration
	ListenAddr  string
}

func loadConfig() Config {
	return Config{
		DatabaseURL: getEnv("DATABASE_URL", "postgres://postgres:password@localhost:5432/ragdb"),
		OpenAIKey:   mustEnv("OPENAI_API_KEY"),
		LLMModel:    getEnv("LLM_MODEL", "gpt-4o-mini"),
		JWTSecret:   mustEnv("JWT_SECRET"),
		JWTExpiry:   24 * time.Hour,
		ListenAddr:  getEnv("LISTEN_ADDR", ":8080"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		slog.Error("required environment variable not set", "key", key)
		os.Exit(1)
	}
	return v
}
