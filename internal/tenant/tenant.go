package tenant

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pixell07/multi-tenant-ai/internal/auth"
	"golang.org/x/crypto/bcrypt"
)

type Organization struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
}

type User struct {
	ID           string    `json:"id"`
	OrgID        string    `json:"org_id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	Role         string    `json:"role"`
	CreatedAt    time.Time `json:"created_at"`
}

type Repository struct {
	db *pgxpool.Pool
}

func NewRepository(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

func (r *Repository) CreateOrg(ctx context.Context, name string) (*Organization, error) {
	org := &Organization{
		ID:        uuid.NewString(),
		Name:      name,
		CreatedAt: time.Now(),
	}
	_, err := r.db.Exec(ctx,
		`INSERT INTO organizations (id, name, created_at) VALUES ($1, $2, $3)`,
		org.ID, org.Name, org.CreatedAt,
	)
	return org, err
}

func (r *Repository) CreateUser(ctx context.Context, u *User) error {
	_, err := r.db.Exec(ctx,
		`INSERT INTO users (id, org_id, email, password_hash, role, created_at)
		 VALUES ($1, $2, $3, $4, $5, $6)`,
		u.ID, u.OrgID, u.Email, u.PasswordHash, u.Role, u.CreatedAt,
	)
	return err
}

func (r *Repository) FindUserByEmail(ctx context.Context, email string) (*User, error) {
	u := &User{}
	err := r.db.QueryRow(ctx,
		`SELECT id, org_id, email, password_hash, role, created_at
		 FROM users WHERE email = $1`,
		email,
	).Scan(&u.ID, &u.OrgID, &u.Email, &u.PasswordHash, &u.Role, &u.CreatedAt)
	if err != nil {
		return nil, err
	}
	return u, nil
}

type Service struct {
	repo *Repository
	jwt  *auth.JWTManager
}

func NewService(repo *Repository, jwt *auth.JWTManager) *Service {
	return &Service{repo: repo, jwt: jwt}
}

type RegisterRequest struct {
	OrgName  string `json:"org_name"`
	Email    string `json:"email"`
	Password string `json:"password"`
}

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type AuthResponse struct {
	Token string        `json:"token"`
	User  *User         `json:"user"`
	Org   *Organization `json:"org"`
}

func (s *Service) Register(ctx context.Context, req RegisterRequest) (*AuthResponse, error) {
	if req.Email == "" || req.Password == "" || req.OrgName == "" {
		return nil, errors.New("all fields required")
	}

	org, err := s.repo.CreateOrg(ctx, req.OrgName)
	if err != nil {
		return nil, err
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	user := &User{
		ID:           uuid.NewString(),
		OrgID:        org.ID,
		Email:        req.Email,
		PasswordHash: string(hash),
		Role:         "admin",
		CreatedAt:    time.Now(),
	}
	if err := s.repo.CreateUser(ctx, user); err != nil {
		return nil, err
	}

	token, err := s.jwt.Generate(org.ID, user.ID, user.Role)
	if err != nil {
		return nil, err
	}

	return &AuthResponse{Token: token, User: user, Org: org}, nil
}

// Login authenticates a user and returns a JWT.
func (s *Service) Login(ctx context.Context, req LoginRequest) (*AuthResponse, error) {
	user, err := s.repo.FindUserByEmail(ctx, req.Email)
	if err != nil {
		return nil, errors.New("invalid credentials")
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
		return nil, errors.New("invalid credentials")
	}

	token, err := s.jwt.Generate(user.OrgID, user.ID, user.Role)
	if err != nil {
		return nil, err
	}

	return &AuthResponse{Token: token, User: user}, nil
}
