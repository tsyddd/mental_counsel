package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

type server struct {
	db *pgxpool.Pool
	rdb *redis.Client
	media string
	modelURL string
	model string
	key string
	client *http.Client
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" { return value }; return fallback
}

func jsonWrite(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func fail(w http.ResponseWriter, status int, message string) {
	jsonWrite(w, status, map[string]string{"error": message})
}

func (s *server) proxy(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/"), "/")
	if len(parts) >= 2 && parts[0] == "users" && parts[1] != currentUser(r) {
		fail(w, 403, "user scope mismatch"); return
	}
	target := s.media + strings.TrimPrefix(r.URL.Path, "/api")
	if r.URL.RawQuery != "" { target += "?" + r.URL.RawQuery }
	req, err := http.NewRequestWithContext(r.Context(), r.Method, target, r.Body)
	if err != nil { fail(w, 400, err.Error()); return }
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	if value := r.Header.Get("Range"); value != "" { req.Header.Set("Range", value) }
	resp, err := s.client.Do(req)
	if err != nil { fail(w, 502, "media service unavailable"); return }
	defer resp.Body.Close()
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	for _, name := range []string{"Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition"} {
		if value := resp.Header.Get(name); value != "" { w.Header().Set(name, value) }
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (s *server) jobEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok { return }
	last := ""
	for {
		select { case <-r.Context().Done(): return; default: }
		resp, err := s.client.Get(s.media + "/jobs/" + id)
		if err == nil {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			resp.Body.Close()
			if string(body) != last {
				fmt.Fprintf(w, "event: progress\ndata: %s\n\n", body)
				flusher.Flush(); last = string(body)
			}
			var status struct{ Status string `json:"status"` }
			_ = json.Unmarshal(body, &status)
			if status.Status == "completed" || status.Status == "failed" || resp.StatusCode == 404 { return }
		}
		select { case <-r.Context().Done(): return; case <-time.After(time.Second): }
	}
}

type sessionRequest struct {
	UserID string `json:"user_id"`
	VideoID string `json:"video_id"`
	Mode string `json:"mode"`
}

func (s *server) createSession(w http.ResponseWriter, r *http.Request) {
	var input sessionRequest
	if json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&input) != nil { fail(w, 400, "invalid request"); return }
	input.UserID = currentUser(r)
	if input.Mode == "" { input.Mode = "counselor" }
	if input.Mode != "counselor" && input.Mode != "interviewer" { fail(w, 400, "unknown agent mode"); return }
	id := uuid.NewString()
	_, err := s.db.Exec(r.Context(), `INSERT INTO sessions(id,user_id,video_id,agent_mode) VALUES ($1,$2,NULLIF($3,'')::text,$4)`, id, input.UserID, input.VideoID, input.Mode)
	if err != nil { fail(w, 400, "could not create session"); return }
	_, _ = s.db.Exec(r.Context(), `INSERT INTO episodes(session_id,episode_no) VALUES ($1,1)`, id)
	if input.VideoID != "" {
		jobID := uuid.NewString()
		job := map[string]any{"job_id": jobID, "kind": "seed_profile", "video_id": input.VideoID, "user_id": input.UserID}
		payload, _ := json.Marshal(job)
		_, _ = s.db.Exec(r.Context(), `INSERT INTO jobs(id,kind,payload) VALUES ($1,'seed_profile',$2)`, jobID, payload)
		_ = s.rdb.LPush(r.Context(), "interview:jobs", payload).Err()
	}
	jsonWrite(w, 201, map[string]string{"session_id": id, "user_id": input.UserID, "video_id": input.VideoID, "mode": input.Mode})
}

func (s *server) getSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.ownsSession(r, id) { fail(w, 404, "session not found"); return }
	rows, err := s.db.Query(r.Context(), `SELECT role,content,created_at FROM messages WHERE session_id=$1 ORDER BY created_at,id LIMIT 100`, id)
	if err != nil { fail(w, 400, "invalid session"); return }
	defer rows.Close()
	items := []map[string]any{}
	for rows.Next() {
		var role, content string; var at time.Time
		if rows.Scan(&role, &content, &at) == nil { items = append(items, map[string]any{"role": role, "content": content, "created_at": at}) }
	}
	jsonWrite(w, 200, map[string]any{"messages": items})
}

func (s *server) listUserSessions(w http.ResponseWriter, r *http.Request) {
	userID := r.PathValue("user")
	if userID != currentUser(r) { fail(w, 403, "user scope mismatch"); return }
	rows, err := s.db.Query(r.Context(), `SELECT id,COALESCE(video_id,''),agent_mode,created_at,
		COALESCE(last_message_at,created_at) FROM sessions WHERE user_id=$1
		ORDER BY COALESCE(last_message_at,created_at) DESC LIMIT 30`, userID)
	if err != nil { fail(w, 500, "sessions unavailable"); return }
	defer rows.Close()
	items := []map[string]any{}
	for rows.Next() {
		var id, videoID, mode string; var created, active time.Time
		if rows.Scan(&id, &videoID, &mode, &created, &active) == nil {
			items = append(items, map[string]any{"session_id": id, "video_id": videoID,
				"mode": mode, "created_at": created, "last_message_at": active})
		}
	}
	jsonWrite(w, 200, map[string]any{"sessions": items})
}

func (s *server) mediaJSON(ctx context.Context, path string, out any) error {
	req, _ := http.NewRequestWithContext(ctx, "GET", s.media+path, nil)
	resp, err := s.client.Do(req)
	if err != nil { return err }
	defer resp.Body.Close()
	if resp.StatusCode != 200 { return fmt.Errorf("media HTTP %d", resp.StatusCode) }
	return json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(out)
}

func (s *server) mediaPostJSON(ctx context.Context, path string, input any, out any) error {
	body, err := json.Marshal(input)
	if err != nil { return err }
	req, err := http.NewRequestWithContext(ctx, "POST", s.media+path, bytes.NewReader(body))
	if err != nil { return err }
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil { return err }
	defer resp.Body.Close()
	if resp.StatusCode != 200 { return fmt.Errorf("media HTTP %d", resp.StatusCode) }
	return json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(out)
}

type chatRequest struct { Text string `json:"text"` }
type wireMessage struct { Role string `json:"role"`; Content string `json:"content"` }

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var input chatRequest
	if json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&input) != nil || strings.TrimSpace(input.Text) == "" || len(input.Text) > 4000 {
		fail(w, 400, "text required (max 4000 bytes)"); return
	}
	var userID, videoID, agentMode string
	var episodeNo int
	var lastMessageAt *time.Time
	err := s.db.QueryRow(r.Context(), `SELECT user_id,COALESCE(video_id,''),agent_mode,episode_no,last_message_at FROM sessions WHERE id=$1`, id).Scan(&userID, &videoID, &agentMode, &episodeNo, &lastMessageAt)
	if err != nil { fail(w, 404, "session not found"); return }
	if userID != currentUser(r) { fail(w, 404, "session not found"); return }
	if lastMessageAt != nil && time.Since(*lastMessageAt) > 30*time.Minute {
		_, _ = s.db.Exec(r.Context(), `UPDATE episodes SET status='closed',ended_at=now() WHERE session_id=$1 AND episode_no=$2`, id, episodeNo)
		episodeNo++
		_, _ = s.db.Exec(r.Context(), `INSERT INTO episodes(session_id,episode_no) VALUES ($1,$2) ON CONFLICT DO NOTHING`, id, episodeNo)
		_, _ = s.db.Exec(r.Context(), `UPDATE sessions SET episode_no=$2 WHERE id=$1`, id, episodeNo)
	}
	userMessageID := uuid.NewString()
	_, err = s.db.Exec(r.Context(), `INSERT INTO messages(id,session_id,episode_no,role,content) VALUES ($1,$2,$3,'user',$4)`, userMessageID, id, episodeNo, input.Text)
	if err != nil { fail(w, 500, "could not save message"); return }
	s.trace(userID, id, "input", "收到用户消息", map[string]any{"message_id": userMessageID, "characters": len([]rune(input.Text))})
	_, _ = s.db.Exec(r.Context(), `UPDATE sessions SET last_message_at=now() WHERE id=$1`, id)
	var explicit struct { Preferences []struct { ID string `json:"id"`; Content string `json:"content"` } `json:"preferences"` }
	if err := s.mediaPostJSON(r.Context(), "/users/"+userID+"/explicit-preference",
		map[string]any{"text": input.Text, "session_id": id, "message_id": userMessageID}, &explicit); err != nil {
		log.Printf("explicit preference capture unavailable: %v", err)
		s.trace(userID, id, "error", "明确偏好写入失败", map[string]any{"stage": "explicit_preference"})
	} else if len(explicit.Preferences) > 0 {
		s.trace(userID, id, "memory_write", "当轮记录明确偏好", map[string]any{"count": len(explicit.Preferences)})
	}
	// The ordered turns carry the ongoing conversation. Long-term memories
	// supply evidence and preferences; they do not replace this history.
	rows, err := s.db.Query(r.Context(), `SELECT role,content FROM messages
		WHERE session_id=$1 AND episode_no=$2 ORDER BY created_at DESC,id DESC LIMIT 12`, id, episodeNo)
	if err != nil { fail(w, 500, "history unavailable"); return }
	recent := []wireMessage{}
	for rows.Next() {
		var item wireMessage
		if rows.Scan(&item.Role, &item.Content) == nil { recent = append(recent, item) }
	}
	rows.Close()
	var previousSummary string
	_ = s.db.QueryRow(r.Context(), `SELECT content FROM memories WHERE user_id=$1
		AND layer='session' AND scope_session_id=$2 AND status='active'
		AND (expires_at IS NULL OR expires_at>now())
		ORDER BY updated_at DESC LIMIT 1`, userID, id).Scan(&previousSummary)
	// A few explicit communication preferences are small, persistent constraints.
	// This indexed lookup is separate from turn-specific semantic retrieval.
	pinnedPreferences := []string{}
	prefRows, prefErr := s.db.Query(r.Context(), `SELECT content FROM memories WHERE user_id=$1
		AND kind='communication_preference' AND layer='user' AND status='active'
		AND (expires_at IS NULL OR expires_at>now()) ORDER BY updated_at DESC LIMIT 3`, userID)
	if prefErr == nil {
		for prefRows.Next() {
			var content string
			if prefRows.Scan(&content) == nil { pinnedPreferences = append(pinnedPreferences, content) }
		}
		prefRows.Close()
	}
	routingRecent := []wireMessage{}
	for i := len(recent)-1; i >= 0; i-- { routingRecent = append(routingRecent, recent[i]) }
	var memoryContext struct {
		Route string `json:"route"`
		Queries []string `json:"queries"`
		Preferences []struct { Content string `json:"content"` } `json:"preferences"`
		Events []struct { Content string `json:"content"`; SourceType string `json:"source_type"`; SourceID string `json:"source_id"`; SourceWindow []wireMessage `json:"source_window"` } `json:"events"`
		Goals []struct { Content string `json:"content"` } `json:"goals"`
		RelatedEvents []struct { Content string `json:"content"` } `json:"related_events"`
		LinkedIntents []struct { Label string `json:"label"`; Status string `json:"status"`; SupportCount int `json:"support_count"` } `json:"linked_intents"`
	}
	if err := s.mediaPostJSON(r.Context(), "/users/"+userID+"/memory-context",
		map[string]any{"text": input.Text, "recent": routingRecent}, &memoryContext); err != nil {
		log.Printf("memory context unavailable: %v", err)
		s.trace(userID, id, "error", "记忆检索不可用", map[string]any{"stage": "memory_context"})
	}
	s.trace(userID, id, "retrieval", "记忆路由与召回", map[string]any{"route": memoryContext.Route,
		"queries": memoryContext.Queries, "events": len(memoryContext.Events),
		"goals": len(memoryContext.Goals), "intents": len(memoryContext.LinkedIntents)})
	contextText := ""
	if previousSummary != "" { contextText = "此前对话阶段摘要：" + previousSummary + "\n" }
	if len(pinnedPreferences) > 0 {
		contextText += "用户明确提出的交流方式（持续遵守）：" + strings.Join(pinnedPreferences, "；") + "\n"
	}
	if len(memoryContext.Preferences) > 0 {
		contextText += "当前话题相关的交流偏好："
		for _, pref := range memoryContext.Preferences {
			if !slices.Contains(pinnedPreferences, pref.Content) { contextText += "\n- " + pref.Content }
		}
	}
	if len(memoryContext.Goals) > 0 {
		contextText += "\n已确认的目标："
		for _, goal := range memoryContext.Goals { contextText += "\n- " + goal.Content }
	}
	if len(memoryContext.Events) > 0 || len(memoryContext.RelatedEvents) > 0 {
		contextText += "\n相关经历与原话证据："
		for _, event := range memoryContext.Events {
			contextText += fmt.Sprintf("\n- %s（来源 %s:%s）", event.Content, event.SourceType, event.SourceID)
			for _, turn := range event.SourceWindow { contextText += "\n  " + turn.Role + "：" + turn.Content }
		}
		for _, event := range memoryContext.RelatedEvents { contextText += "\n- 关联经历：" + event.Content }
	}
	if len(memoryContext.LinkedIntents) > 0 {
		contextText += "\n关联跟进方向："
		for _, intent := range memoryContext.LinkedIntents {
			if intent.Status == "confirmed" { contextText += "\n- 已确认：" + intent.Label
			} else { contextText += "\n- 待确认、仅供澄清：" + intent.Label }
		}
	}
	var manualNotes string
	_ = s.db.QueryRow(r.Context(), `SELECT notes FROM user_preference_docs WHERE user_id=$1`, userID).Scan(&manualNotes)
	if manualNotes != "" { contextText += "\n用户维护的偏好笔记：\n" + manualNotes + "\n" }
	_ = s.preferenceMarkdown(r)
	rolePrompt := "你是支持性心理咨询师 Agent。先共情、澄清，再提开放问题；根据画像与长期记忆调整对话节奏。"
	if agentMode == "interviewer" { rolePrompt = "你是访谈者 Agent。围绕学习、睡眠、人际和家庭等访谈提纲，每次只问一个开放问题；根据回答决定下一题与追问，不机械按顺序提问。" }
	messages := []wireMessage{{Role: "system", Content: rolePrompt + "按时间顺序承接最近对话；始终遵守用户已明确提出的交流方式。历史记忆只补充与当前话题有关的背景，不要逐条复述。偏好用于调整提问方式；具体经历用于有依据的追问。待确认方向只在当前对话自然涉及且用户尚未拒绝时，最多选择一个，用询问方式请用户确认；不要把推断说成事实。不要做临床诊断。若用户表达当前自伤或其他迫切安全风险，先鼓励立即联系身边可信赖成年人及当地紧急服务，并暂停普通访谈提问。\n" + contextText}}
	for i := len(recent)-1; i >= 0; i-- { messages = append(messages, recent[i]) }
	s.trace(userID, id, "context", "组装本轮上下文", map[string]any{"recent_turns": len(recent),
		"pinned_preferences": len(pinnedPreferences), "manual_notes": manualNotes != "",
		"retrieved_events": len(memoryContext.Events)})
	body, _ := json.Marshal(map[string]any{"model": s.model, "messages": messages, "stream": true, "max_tokens": 1200})
	modelStarted := time.Now()
	req, _ := http.NewRequestWithContext(r.Context(), "POST", strings.TrimRight(s.modelURL, "/")+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+s.key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil { s.trace(userID,id,"error","模型请求失败",map[string]any{"stage":"model"}); fail(w, 502, "model unavailable"); return }
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1000)); log.Printf("model HTTP %d: %s", resp.StatusCode, data)
		s.trace(userID,id,"error","模型服务返回错误",map[string]any{"status":resp.StatusCode})
		fail(w, 502, "model request failed"); return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher); if !ok { return }
	result := strings.Builder{}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 2<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") { continue }
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" { break }
		var chunk struct { Choices []struct { Delta struct { Content string `json:"content"` } `json:"delta"` } `json:"choices"` }
		if json.Unmarshal([]byte(data), &chunk) != nil || len(chunk.Choices) == 0 { continue }
		part := chunk.Choices[0].Delta.Content
		if part != "" {
			result.WriteString(part)
			payload, _ := json.Marshal(map[string]string{"text": part})
			fmt.Fprintf(w, "event: token\ndata: %s\n\n", payload); flusher.Flush()
		}
	}
	lastMessageID := userMessageID
	if result.Len() > 0 {
		lastMessageID = uuid.NewString()
		_, _ = s.db.Exec(context.Background(), `INSERT INTO messages(id,session_id,episode_no,role,content) VALUES ($1,$2,$3,'assistant',$4)`, lastMessageID, id, episodeNo, result.String())
	}
	s.trace(userID, id, "response", "生成咨询师回复", map[string]any{"model": s.model,
		"characters": len([]rune(result.String())), "elapsed_ms": time.Since(modelStarted).Milliseconds()})
	var userTurnCount int
	_ = s.db.QueryRow(context.Background(), `SELECT count(*) FROM messages WHERE session_id=$1 AND role='user'`, id).Scan(&userTurnCount)
	jobID := ""
	if userTurnCount > 0 && userTurnCount%5 == 0 {
		batchNo := userTurnCount / 5
		jobID = uuid.NewString()
		var firstMessageID string
		_ = s.db.QueryRow(context.Background(), `SELECT id FROM messages WHERE session_id=$1 AND role='user'
			ORDER BY created_at DESC,id DESC OFFSET 4 LIMIT 1`, id).Scan(&firstMessageID)
		job := map[string]any{"job_id": jobID, "kind": "extract_memory", "user_id": userID,
			"session_id": id, "batch_no": batchNo, "first_message_id": firstMessageID,
			"last_message_id": lastMessageID}
		jobBytes, _ := json.Marshal(job)
		var accepted string
		err := s.db.QueryRow(context.Background(), `INSERT INTO memory_batches(session_id,batch_no,job_id)
			VALUES ($1,$2,$3) ON CONFLICT DO NOTHING RETURNING job_id`, id, batchNo, jobID).Scan(&accepted)
		if err == nil {
			_, err = s.db.Exec(context.Background(), `INSERT INTO jobs(id,kind,payload) VALUES ($1,'extract_memory',$2)`, jobID, jobBytes)
			if err == nil { _ = s.rdb.LPush(context.Background(), "interview:jobs", jobBytes).Err(); s.trace(userID,id,"memory_write","提交五轮记忆整理",map[string]any{"job_id":jobID,"batch":batchNo}) }
		} else { jobID = "" }
	}
	done, _ := json.Marshal(map[string]any{"memory_job_id": jobID,
		"memory_route": memoryContext.Route, "queries": memoryContext.Queries})
	fmt.Fprintf(w, "event: done\ndata: %s\n\n", done); flusher.Flush()
}

func urlQuery(value string) string {
	return url.QueryEscape(value)
}

func main() {
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, os.Getenv("DATABASE_URL"))
	if err != nil { log.Fatal(err) }
	defer pool.Close()
	rdb := redis.NewClient(&redis.Options{Addr: env("REDIS_URL", "redis:6379")})
	s := &server{db: pool, rdb: rdb, media: env("MEDIA_API_URL", "http://media-api:8000"),
		modelURL: env("OPENAI_BASE_URL", "https://api2.mcben.top/v1"),
		model: env("RELATION_VIDEOARM_MODEL", "deepseek-v4.1-flash"),
		key: os.Getenv("OPENAI_API_KEY"), client: &http.Client{Timeout: 5*time.Minute}}
	if s.key == "" { log.Fatal(errors.New("OPENAI_API_KEY required")) }
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/auth/register", s.register)
	mux.HandleFunc("POST /api/auth/login", s.login)
	mux.HandleFunc("POST /api/auth/logout", s.logout)
	mux.HandleFunc("GET /api/auth/me", s.me)
	mux.HandleFunc("POST /api/sessions", s.createSession)
	mux.HandleFunc("GET /api/sessions/{id}", s.getSession)
	mux.HandleFunc("GET /api/users/{user}/sessions", s.listUserSessions)
	mux.HandleFunc("POST /api/sessions/{id}/messages", s.chat)
	mux.HandleFunc("GET /api/sessions/{id}/trace", s.getTrace)
	mux.HandleFunc("GET /api/preferences", s.getPreferences)
	mux.HandleFunc("PUT /api/preferences", s.putPreferences)
	mux.HandleFunc("POST /api/demo/load", s.loadDemo)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.jobEvents)
	mux.HandleFunc("/api/", s.proxy)
	mux.Handle("/", http.FileServer(http.Dir(filepath.Clean("/web"))))
	log.Fatal(http.ListenAndServe(":8080", s.authMiddleware(mux)))
}
