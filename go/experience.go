package main

import (
    "context"
    "encoding/json"
    "io"
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"

    "github.com/google/uuid"
)

func (s *server) ownsSession(r *http.Request, sessionID string) bool {
    var owner string
    err := s.db.QueryRow(r.Context(), `SELECT user_id FROM sessions WHERE id=$1`, sessionID).Scan(&owner)
    return err == nil && owner == currentUser(r)
}

func (s *server) trace(userID, sessionID, kind, title string, detail any) {
    body, _ := json.Marshal(detail)
    _, _ = s.db.Exec(context.Background(), `INSERT INTO trace_events(id,user_id,session_id,kind,title,detail)
        VALUES ($1,$2,$3,$4,$5,$6)`, uuid.NewString(), userID, sessionID, kind, title, body)
}

func (s *server) getTrace(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    if !s.ownsSession(r, id) { fail(w, 404, "session not found"); return }
    rows, err := s.db.Query(r.Context(), `SELECT kind,title,detail,created_at FROM trace_events
        WHERE session_id=$1 ORDER BY created_at,id LIMIT 300`, id)
    if err != nil { fail(w, 500, "trace unavailable"); return }
    defer rows.Close()
    items := []map[string]any{}
    for rows.Next() {
        var kind, title string
        var detail []byte
        var at time.Time
        if rows.Scan(&kind, &title, &detail, &at) == nil {
            var value any
            _ = json.Unmarshal(detail, &value)
            items = append(items, map[string]any{"kind": kind, "title": title,
                "detail": value, "created_at": at})
        }
    }
    jsonWrite(w, 200, map[string]any{"events": items})
}

func (s *server) preferenceMarkdown(r *http.Request) string {
    userID := currentUser(r)
    var notes string
    _ = s.db.QueryRow(r.Context(), `SELECT notes FROM user_preference_docs WHERE user_id=$1`, userID).Scan(&notes)
    rows, err := s.db.Query(r.Context(), `SELECT content FROM memories WHERE user_id=$1
        AND layer='user' AND kind IN ('communication_preference','preference')
        AND status='active' AND (expires_at IS NULL OR expires_at>now())
        ORDER BY updated_at DESC LIMIT 30`, userID)
    b := strings.Builder{}
    b.WriteString("# 交流偏好\n\n")
    if err == nil {
        for rows.Next() {
            var content string
            if rows.Scan(&content) == nil { b.WriteString("- " + strings.ReplaceAll(content, "\n", " ") + "\n") }
        }
        rows.Close()
    }
    b.WriteString("\n# 用户补充\n\n")
    b.WriteString(strings.TrimSpace(notes))
    b.WriteString("\n")
    markdown := b.String()
    root := env("USER_DATA_ROOT", "/data/users")
    path := filepath.Join(root, userID, "PREFERENCES.md")
    if err := os.MkdirAll(filepath.Dir(path), 0700); err == nil {
        tmp := path + ".tmp"
        if os.WriteFile(tmp, []byte(markdown), 0600) == nil { _ = os.Rename(tmp, path) }
    }
    return markdown
}

func (s *server) getPreferences(w http.ResponseWriter, r *http.Request) {
    var notes string
    _ = s.db.QueryRow(r.Context(), `SELECT notes FROM user_preference_docs WHERE user_id=$1`,
        currentUser(r)).Scan(&notes)
    jsonWrite(w, 200, map[string]string{"markdown": s.preferenceMarkdown(r), "notes": notes})
}

func (s *server) putPreferences(w http.ResponseWriter, r *http.Request) {
    var input struct { Notes string `json:"notes"` }
    if json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&input) != nil ||
        len(input.Notes) > 5000 { fail(w, 400, "invalid preference notes"); return }
    _, err := s.db.Exec(r.Context(), `INSERT INTO user_preference_docs(user_id,notes)
        VALUES ($1,$2) ON CONFLICT(user_id) DO UPDATE SET notes=excluded.notes,
        updated_at=now()`, currentUser(r), strings.TrimSpace(input.Notes))
    if err != nil { fail(w, 500, "could not save preferences"); return }
    jsonWrite(w, 200, map[string]string{"markdown": s.preferenceMarkdown(r)})
}

func (s *server) loadDemo(w http.ResponseWriter, r *http.Request) {
    var result map[string]any
    err := s.mediaPostJSON(r.Context(), "/users/"+currentUser(r)+"/demo/load",
        map[string]any{}, &result)
    if err != nil { fail(w, 502, "demo import unavailable"); return }
    jsonWrite(w, 200, result)
}
