package main

import (
    "context"
    "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "io"
    "net/http"
    "strings"
    "time"

    "github.com/google/uuid"
    "golang.org/x/crypto/bcrypt"
)

type authKey struct{}

func currentUser(r *http.Request) string {
    value, _ := r.Context().Value(authKey{}).(string)
    return value
}

func digestToken(token string) string {
    hash := sha256.Sum256([]byte(token))
    return hex.EncodeToString(hash[:])
}

func (s *server) authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if !strings.HasPrefix(r.URL.Path, "/api/") ||
            strings.HasPrefix(r.URL.Path, "/api/auth/login") ||
            strings.HasPrefix(r.URL.Path, "/api/auth/register") {
            next.ServeHTTP(w, r)
            return
        }
        cookie, err := r.Cookie("interview_session")
        if err != nil { fail(w, http.StatusUnauthorized, "login required"); return }
        var userID string
        err = s.db.QueryRow(r.Context(), `SELECT user_id::text FROM login_sessions
            WHERE token_hash=$1 AND expires_at>now()`, digestToken(cookie.Value)).Scan(&userID)
        if err != nil { fail(w, http.StatusUnauthorized, "session expired"); return }
        next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), authKey{}, userID)))
    })
}

type authRequest struct {
    Email string `json:"email"`
    Password string `json:"password"`
    DisplayName string `json:"display_name"`
}

func readAuth(r *http.Request) (authRequest, bool) {
    var input authRequest
    err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&input)
    input.Email = strings.ToLower(strings.TrimSpace(input.Email))
    input.DisplayName = strings.TrimSpace(input.DisplayName)
    return input, err == nil && len(input.Email) >= 5 && len(input.Email) <= 254 &&
        strings.Contains(input.Email, "@") && len(input.Password) >= 10 && len(input.Password) <= 128
}

func (s *server) issueLogin(w http.ResponseWriter, r *http.Request, userID, displayName string) {
    raw := make([]byte, 32)
    if _, err := rand.Read(raw); err != nil { fail(w, 500, "login unavailable"); return }
    token := hex.EncodeToString(raw)
    expires := time.Now().Add(14 * 24 * time.Hour)
    _, err := s.db.Exec(r.Context(), `INSERT INTO login_sessions(token_hash,user_id,expires_at)
        VALUES ($1,$2,$3)`, digestToken(token), userID, expires)
    if err != nil { fail(w, 500, "login unavailable"); return }
    secure := r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https"
    http.SetCookie(w, &http.Cookie{Name: "interview_session", Value: token, Path: "/",
        HttpOnly: true, SameSite: http.SameSiteLaxMode, Secure: secure,
        Expires: expires, MaxAge: 14 * 24 * 3600})
    jsonWrite(w, 200, map[string]string{"user_id": userID, "display_name": displayName})
}

func (s *server) register(w http.ResponseWriter, r *http.Request) {
    input, ok := readAuth(r)
    if !ok || len(input.DisplayName) < 1 || len(input.DisplayName) > 60 {
        fail(w, 400, "valid email, display name and password of at least 10 characters required"); return
    }
    hash, err := bcrypt.GenerateFromPassword([]byte(input.Password), bcrypt.DefaultCost)
    if err != nil { fail(w, 500, "registration unavailable"); return }
    id := uuid.NewString()
    _, err = s.db.Exec(r.Context(), `INSERT INTO app_users(id,email,display_name,password_hash)
        VALUES ($1,$2,$3,$4)`, id, input.Email, input.DisplayName, string(hash))
    if err != nil { fail(w, 409, "email already registered"); return }
    s.issueLogin(w, r, id, input.DisplayName)
}

func (s *server) login(w http.ResponseWriter, r *http.Request) {
    input, ok := readAuth(r)
    if !ok { fail(w, 400, "invalid credentials"); return }
    var id, name, hash string
    err := s.db.QueryRow(r.Context(), `SELECT id::text,display_name,password_hash
        FROM app_users WHERE email=$1`, input.Email).Scan(&id, &name, &hash)
    if err != nil || bcrypt.CompareHashAndPassword([]byte(hash), []byte(input.Password)) != nil {
        fail(w, 401, "invalid credentials"); return
    }
    s.issueLogin(w, r, id, name)
}

func (s *server) logout(w http.ResponseWriter, r *http.Request) {
    if cookie, err := r.Cookie("interview_session"); err == nil {
        _, _ = s.db.Exec(r.Context(), `DELETE FROM login_sessions WHERE token_hash=$1`, digestToken(cookie.Value))
    }
    http.SetCookie(w, &http.Cookie{Name: "interview_session", Path: "/", MaxAge: -1,
        HttpOnly: true, SameSite: http.SameSiteLaxMode})
    jsonWrite(w, 200, map[string]bool{"ok": true})
}

func (s *server) me(w http.ResponseWriter, r *http.Request) {
    var name, email string
    err := s.db.QueryRow(r.Context(), `SELECT display_name,email FROM app_users WHERE id=$1`,
        currentUser(r)).Scan(&name, &email)
    if err != nil { fail(w, 401, "account unavailable"); return }
    jsonWrite(w, 200, map[string]string{"user_id": currentUser(r),
        "display_name": name, "email": email})
}
