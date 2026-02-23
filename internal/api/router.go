package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/pixell07/multi-tenant-ai/internal/auth"
	"github.com/pixell07/multi-tenant-ai/internal/document"
	"github.com/pixell07/multi-tenant-ai/internal/retrieval"
	"github.com/pixell07/multi-tenant-ai/internal/tenant"
)

type contextKey string

const claimsKey contextKey = "claims"

type RouterDeps struct {
	TenantService   *tenant.Service
	DocumentService *document.Service
	RAGService      *retrieval.RAGService
	JWTManager      *auth.JWTManager
	Logger          *slog.Logger
}

func NewRouter(deps RouterDeps) http.Handler {
	mux := http.NewServeMux()

	h := &handlers{deps: deps}

	// Public routes
	mux.HandleFunc("POST /api/v1/auth/register", h.register)
	mux.HandleFunc("POST /api/v1/auth/login", h.login)
	mux.HandleFunc("GET  /api/v1/health", h.health)

	// Protected routes (wrapped with auth middleware)
	protected := http.NewServeMux()
	protected.HandleFunc("GET  /api/v1/documents", h.listDocuments)
	protected.HandleFunc("POST /api/v1/documents", h.uploadDocument)
	protected.HandleFunc("DELETE /api/v1/documents/{id}", h.deleteDocument)
	protected.HandleFunc("POST /api/v1/query", h.query)          // SSE streaming
	protected.HandleFunc("POST /api/v1/query/sync", h.querySync) // one-shot for testing

	mux.Handle("/api/v1/", h.authMiddleware(protected))

	return h.loggingMiddleware(mux)
}

// Handlers

type handlers struct {
	deps RouterDeps
}

func (h *handlers) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "time": time.Now().Format(time.RFC3339)})
}

func (h *handlers) register(w http.ResponseWriter, r *http.Request) {
	var req tenant.RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	resp, err := h.deps.TenantService.Register(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, resp)
}

func (h *handlers) login(w http.ResponseWriter, r *http.Request) {
	var req tenant.LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	resp, err := h.deps.TenantService.Login(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) listDocuments(w http.ResponseWriter, r *http.Request) {
	claims := claimsFromCtx(r.Context())

	docs, err := h.deps.DocumentService.List(r.Context(), claims.OrgID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list documents")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "count": len(docs)})
}

func (h *handlers) uploadDocument(w http.ResponseWriter, r *http.Request) {
	claims := claimsFromCtx(r.Context())

	var body struct {
		Name    string `json:"name"`
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if body.Name == "" || body.Content == "" {
		writeError(w, http.StatusBadRequest, "name and content are required")
		return
	}

	doc, err := h.deps.DocumentService.Upload(r.Context(), document.UploadRequest{
		OrgID:   claims.OrgID,
		Name:    body.Name,
		Content: body.Content,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to upload document")
		return
	}
	writeJSON(w, http.StatusAccepted, doc)
}

func (h *handlers) deleteDocument(w http.ResponseWriter, r *http.Request) {
	claims := claimsFromCtx(r.Context())
	docID := r.PathValue("id")

	if err := h.deps.DocumentService.Delete(r.Context(), docID, claims.OrgID); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to delete document")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// query handles SSE streaming of RAG responses.
// The client receives a stream of "data: <token>\n\n" events.
func (h *handlers) query(w http.ResponseWriter, r *http.Request) {
	claims := claimsFromCtx(r.Context())

	var body struct {
		Question string `json:"question"`
		TopK     int    `json:"top_k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if body.Question == "" {
		writeError(w, http.StatusBadRequest, "question is required")
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable Nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	out := make(chan string, 64)

	go func() {
		if err := h.deps.RAGService.Query(r.Context(), retrieval.QueryRequest{
			OrgID:    claims.OrgID,
			Question: body.Question,
			TopK:     body.TopK,
		}, out); err != nil {
			// If context was cancelled (client disconnected), that's fine
			if r.Context().Err() == nil {
				h.deps.Logger.Error("RAG query error", "error", err)
			}
		}
	}()

	for token := range out {
		// SSE format: "data: <content>\n\n"
		payload := strings.ReplaceAll(token, "\n", "\\n") // escape newlines in token
		fmt.Fprintf(w, "data: %s\n\n", payload)
		flusher.Flush()
	}

	// Signal end of stream
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// querySync is a non-streaming endpoint for testing/simple clients.
func (h *handlers) querySync(w http.ResponseWriter, r *http.Request) {
	claims := claimsFromCtx(r.Context())

	var body struct {
		Question string `json:"question"`
		TopK     int    `json:"top_k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	out := make(chan string, 256)
	var sb strings.Builder

	go func() {
		_ = h.deps.RAGService.Query(r.Context(), retrieval.QueryRequest{
			OrgID:    claims.OrgID,
			Question: body.Question,
			TopK:     body.TopK,
		}, out)
	}()

	for token := range out {
		sb.WriteString(token)
	}

	writeJSON(w, http.StatusOK, map[string]string{"answer": sb.String()})
}

//  Middleware

func (h *handlers) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if !strings.HasPrefix(authHeader, "Bearer ") {
			writeError(w, http.StatusUnauthorized, "missing bearer token")
			return
		}

		token := strings.TrimPrefix(authHeader, "Bearer ")
		claims, err := h.deps.JWTManager.Verify(token)
		if err != nil {
			writeError(w, http.StatusUnauthorized, "invalid or expired token")
			return
		}

		ctx := context.WithValue(r.Context(), claimsKey, claims)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (h *handlers) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		h.deps.Logger.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.status,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}

// Helpers

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func claimsFromCtx(ctx context.Context) *auth.Claims {
	c, _ := ctx.Value(claimsKey).(*auth.Claims)
	return c
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(status int) {
	rw.status = status
	rw.ResponseWriter.WriteHeader(status)
}
