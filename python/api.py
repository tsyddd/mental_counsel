import json
import math
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .core import (conn, queue, safe_media_path, search_graph, answer_video, review_frames,
                   embed, vector_literal, record_tool_memory)
from .memory_retrieval import retrieve_context
from .demo_case import load_demo

app = FastAPI(title="Interview Media and Memory API")


class VideoRequest(BaseModel):
    video_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    path: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ReportRequest(BaseModel):
    questions: list[str] = []


class SeedRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


class MemoryContextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    recent: list[dict[str, str]] = Field(default_factory=list, max_length=12)


@app.post("/users/{user_id}/memory-context")
def memory_context(user_id: str, req: MemoryContextRequest):
    return retrieve_context(user_id, req.text, req.recent)


@app.post("/users/{user_id}/demo/load")
def demo_load(user_id: str):
    return load_demo(user_id)


def enqueue(kind, **payload):
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "kind": kind, **payload}
    with conn() as db:
        db.execute("INSERT INTO jobs(id,kind,payload) VALUES (%s,%s,%s)",
                   (job_id, kind, json.dumps(job, ensure_ascii=False)))
    queue().lpush("interview:jobs", json.dumps(job, ensure_ascii=False))
    return {"job_id": job_id, "status": "queued"}


@app.get("/health")
def health():
    with conn() as db:
        db.execute("SELECT 1")
    queue().ping()
    return {"status": "ok"}


@app.post("/videos", status_code=202)
def add_video(req: VideoRequest):
    try:
        path = safe_media_path(req.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with conn() as db:
        db.execute("""INSERT INTO videos(id,path,status) VALUES (%s,%s,'queued')
            ON CONFLICT(id) DO UPDATE SET path=excluded.path,status='queued'""",
            (req.video_id, str(path)))
    return enqueue("ingest", video_id=req.video_id, path=str(path))


@app.get("/jobs/{job_id}")
def job_status(job_id: uuid.UUID):
    with conn() as db:
        row = db.execute("SELECT id,kind,status,progress,result,error,updated_at FROM jobs WHERE id=%s",
                         (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "job not found")
    row["id"] = str(row["id"])
    row["updated_at"] = row["updated_at"].isoformat()
    return row


@app.get("/videos/{video_id}/graph")
def graph(video_id: str):
    with conn() as db:
        row = db.execute("SELECT status,graph FROM videos WHERE id=%s", (video_id,)).fetchone()
    if not row:
        raise HTTPException(404, "video not found")
    return row


@app.get("/videos/{video_id}/media")
def video_media(video_id: str):
    with conn() as db:
        row = db.execute("SELECT path FROM videos WHERE id=%s", (video_id,)).fetchone()
    if not row:
        raise HTTPException(404, "video not found")
    try:
        path = safe_media_path(row["path"])
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/videos/{video_id}/search")
def search(video_id: str, req: QuestionRequest, user_id: str | None = None,
           session_id: uuid.UUID | None = None):
    try:
        result = search_graph(video_id, req.question)
        count = len(result.get("macros", []))
        record_tool_memory("hierarchical_video_retrieval", req.question, "succeeded",
                           f"retrieved {count} timestamped Macro events", user_id=user_id, session_id=session_id)
        return result
    except ValueError as exc:
        record_tool_memory("hierarchical_video_retrieval", req.question, "failed", error=str(exc),
                           user_id=user_id, session_id=session_id)
        raise HTTPException(404, str(exc)) from exc


@app.post("/videos/{video_id}/answer")
def answer(video_id: str, req: QuestionRequest, user_id: str | None = None,
           session_id: uuid.UUID | None = None):
    try:
        result = answer_video(video_id, req.question)
        evidence_count = len(result.get("retrieval", {}).get("macros", []))
        record_tool_memory("video_evidence_answer", req.question, "succeeded",
                           f"answered with {evidence_count} timestamped event candidates",
                           user_id=user_id, session_id=session_id)
        return result
    except ValueError as exc:
        record_tool_memory("video_evidence_answer", req.question, "failed", error=str(exc),
                           user_id=user_id, session_id=session_id)
        raise HTTPException(404, str(exc)) from exc


@app.get("/videos/{video_id}/frames")
def frames(video_id: str, start: float, end: float, user_id: str | None = None,
           session_id: uuid.UUID | None = None):
    if start < 0 or end <= start or end - start > 60:
        raise HTTPException(400, "invalid frame window")
    with conn() as db:
        row = db.execute("SELECT path FROM videos WHERE id=%s", (video_id,)).fetchone()
    if not row:
        raise HTTPException(404, "video not found")
    try:
        result = {"frames": review_frames(row["path"], start, end)}
        record_tool_memory("timestamped_frame_review", f"{video_id}:{start:.2f}-{end:.2f}", "succeeded",
                           f"reviewed {len(result['frames'])} frames in the selected interval",
                           user_id=user_id, session_id=session_id)
        return result
    except Exception as exc:
        record_tool_memory("timestamped_frame_review", f"{video_id}:{start:.2f}-{end:.2f}", "failed",
                           error=str(exc), user_id=user_id, session_id=session_id)
        raise


@app.post("/videos/{video_id}/report", status_code=202)
def report_job(video_id: str, req: ReportRequest):
    return enqueue("report", video_id=video_id, questions=req.questions)


@app.get("/videos/{video_id}/report")
def report(video_id: str):
    with conn() as db:
        row = db.execute("SELECT report,profile,created_at FROM reports WHERE video_id=%s",
                         (video_id,)).fetchone()
    if not row:
        raise HTTPException(404, "report not ready")
    row["created_at"] = row["created_at"].isoformat()
    return row


@app.post("/videos/{video_id}/seed-profile", status_code=202)
def seed(video_id: str, req: SeedRequest):
    return enqueue("seed_profile", video_id=video_id, user_id=req.user_id)


@app.get("/users/{user_id}/memories")
def memories(user_id: str, query: str, limit: int = 8,
             layers: str = "user,episodic", session_id: uuid.UUID | None = None):
    vector = vector_literal(embed(["query: " + query])[0])
    allowed = [layer for layer in layers.split(",") if layer in
               {"session", "episodic", "user", "procedural", "failure"}]
    if not allowed:
        allowed = ["user", "episodic"]
    with conn() as db:
        rows = db.execute("""SELECT id,kind,layer,content,source_type,source_id,confidence,metadata,
            salience,reinforcement_count,created_at,updated_at,last_accessed_at,
            1-(embedding <=> %s::vector) AS similarity
            FROM memories WHERE user_id=%s AND status='active' AND layer=ANY(%s)
            AND (layer<>'session' OR scope_session_id=%s)
            AND (expires_at IS NULL OR expires_at>now())
            ORDER BY embedding <=> %s::vector LIMIT 40""",
            (vector, user_id, allowed, session_id, vector)).fetchall()
        now = datetime.now(timezone.utc)
        for row in rows:
            age_days = max(0.0, (now - row["updated_at"]).total_seconds() / 86400)
            recency = math.exp(-age_days / 45.0)
            row["retrieval_score"] = (0.70 * float(row["similarity"]) +
                0.12 * min(1.0, float(row["salience"]) / 2.0) +
                0.08 * min(1.0, math.log1p(row["reinforcement_count"]) / math.log(6)) +
                0.10 * recency)
        rows.sort(key=lambda item: item["retrieval_score"], reverse=True)
        rows = rows[:max(1, min(limit, 20))]
        if rows:
            db.execute("UPDATE memories SET last_accessed_at=now() WHERE id=ANY(%s::uuid[])",
                       ([row["id"] for row in rows],))
    for row in rows:
        row["id"] = str(row["id"])
        row["created_at"] = row["created_at"].isoformat()
        row["updated_at"] = row["updated_at"].isoformat()
        if row["last_accessed_at"]:
            row["last_accessed_at"] = row["last_accessed_at"].isoformat()
        row["similarity"] = float(row["similarity"])
        row["retrieval_score"] = float(row["retrieval_score"])
    return {"memories": rows}


@app.get("/users/{user_id}/intents")
def intents(user_id: str, query: str = ""):
    query = query.strip()
    if query:
        vector = vector_literal(embed(["query: " + query])[0])
        with conn() as db:
            candidates = db.execute("""SELECT i.id,i.label,i.status,i.support_count,
                c.label AS concept_label,1-(c.embedding <=> %s::vector) AS similarity
                FROM intents i JOIN concepts c ON c.id=ANY(i.concept_ids)
                WHERE i.user_id=%s AND i.status='tentative' AND i.support_count>=3
                AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector LIMIT 12""",
                (vector, user_id, vector)).fetchall()
        # A tentative intent is a conversation cue only when the current topic
        # resembles its supporting concept. The threshold is intentionally
        # conservative until calibrated on annotated dialogues.
        rows = [row for row in candidates if float(row["similarity"]) >= 0.75][:2]
        for row in rows:
            row["id"] = str(row["id"])
            row["similarity"] = float(row["similarity"])
        return {"intents": rows}
    with conn() as db:
        rows = db.execute("""SELECT id,label,status,support_count FROM intents
            WHERE user_id=%s AND status='tentative' ORDER BY updated_at DESC LIMIT 5""",
            (user_id,)).fetchall()
    for row in rows:
        row["id"] = str(row["id"])
    return {"intents": rows}


@app.get("/users/{user_id}/memory-graph")
def memory_graph(user_id: str):
    with conn() as db:
        memories = db.execute("""SELECT id,kind,layer,content,source_type,confidence,salience,
            reinforcement_count,created_at FROM memories WHERE user_id=%s AND status='active'
            AND (expires_at IS NULL OR expires_at>now()) ORDER BY created_at DESC LIMIT 60""",
            (user_id,)).fetchall()
        concepts = db.execute("""SELECT id,label,support_count,strength,memory_ids,last_seen_at
            FROM concepts WHERE user_id=%s ORDER BY strength DESC,last_seen_at DESC LIMIT 30""", (user_id,)).fetchall()
        intents = db.execute("""SELECT id,label,status,support_count,concept_ids,updated_at
            FROM intents WHERE user_id=%s ORDER BY updated_at DESC LIMIT 15""", (user_id,)).fetchall()
        tool_counts = db.execute("""SELECT tool_name,status,count(*) AS count FROM tool_runs
            WHERE user_id=%s GROUP BY tool_name,status ORDER BY tool_name,status""", (user_id,)).fetchall()
    for row in memories:
        row["id"] = str(row["id"]); row["created_at"] = row["created_at"].isoformat()
    for row in concepts:
        row["id"] = str(row["id"]); row["memory_ids"] = [str(x) for x in row["memory_ids"]]
        row["last_seen_at"] = row["last_seen_at"].isoformat()
    for row in intents:
        row["id"] = str(row["id"]); row["concept_ids"] = [str(x) for x in row["concept_ids"]]
        row["updated_at"] = row["updated_at"].isoformat()
    return {"memories": memories, "concepts": concepts, "intents": intents, "tool_runs": tool_counts}
