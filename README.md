# Interview Memory Platform

An interview-video and counseling demo built around the existing
`relation-event-videoarm` graph builder. The graph is built bottom-up
(ASR caption blocks → Macro → Super → Root) and queried top-down
(Root → Super → Macro → timestamped captions and frame review).

The Python API/worker handles media, evidence, expert-skill reports, and
memory consolidation. A Go gateway serves the video timeline and two chat
roles (interviewer and counselor), and streams model responses. PostgreSQL
with pgvector stores graph, reports, and memories; Redis carries asynchronous
jobs. The report profile seeds long-term user memory. New dialogue is separated
into user facts/preferences and person-centered episodic events; repeated event
concepts can form a tentative follow-up intent. Session, episodic, user,
procedural-tool, and failure memories have separate scopes and retrieval paths.
See [MEMORY_DESIGN.md](MEMORY_DESIGN.md) for the write/update/retrieval rules
and how they map to Mem0 and CogniFold.

The current web app adds account registration and login, user-scoped sessions,
an event/concept/intent graph inspector, a Markdown preference snapshot, and
per-turn trace events for retrieval, context assembly, response, and memory
writeback. A curated synthetic two-session case can be loaded from the UI;
it is distinct from the real video evidence workflow. No interview video,
student report, credential, or runtime database is included in this repository.

## Start

Copy `.env.example` to `.env` and fill the model key locally. Keep `.env`
private. Set `PGDATA_HOST_PATH` and `USER_DATA_HOST_PATH` to directories on ext4, not NTFS; PostgreSQL
requires POSIX file ownership. Set `DEMO_VIDEO_PATH` to a file inside `./media`,
then:

```bash
docker compose up --build -d
curl -X POST http://127.0.0.1:8080/api/videos \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"interview-demo","path":"/media/interview.mp4"}'
```

Open `http://127.0.0.1:8080` on the server, register an account, and sign in.
From another machine, forward port 8080 over SSH and open the forwarded
localhost port. Poll `/api/jobs/{id}` or subscribe to
`/api/jobs/{id}/events`. The UI accepts a video ID, evidence question,
report-generation request, and interviewer or counselor chat. It shows a
clickable Super/Macro timeline, frame review, evidence IDs and tentative
follow-up intents.

The ASR worker uses `faster-whisper` and creates timestamped caption blocks.
The first run downloads its configured model unless it is already cached.
The language/vision model is configured through an OpenAI-compatible URL;
image support depends on that model endpoint. If image input is rejected,
frame review remains available to the user and the answer marks visual
verification as unavailable.

The vector encoder uses Qdrant FastEmbed's ONNX runtime with a 384-dimensional
multilingual E5 small model, so the CPU deployment avoids a full PyTorch/CUDA
stack. The demo result and verification notes are in [CASE_STUDY.md](CASE_STUDY.md).

The development deployment binds the gateway to localhost. If exposing it to
other users, add HTTPS, consent and retention policies, encryption, and a
human reviewer for any psychological interpretation.

## Memory timing

Every fifth completed user turn queues a batched memory-routing and extraction job.
The first four turns remain in the ordered conversation history while the next
batch is pending. A return after
30 minutes starts a new episode under the same session ID. The worker also
consolidates an idle episode into a short, expiring summary after 30 minutes;
explicit corrections supersede old facts while preserving their record. Three
distinct supporting facts under one concept can create a **tentative** intent.
The counselor asks before following that suggestion. Initial profile facts
carry report evidence IDs as their source.

The demo ASR and graph can also be prepared before the service starts:

```bash
ASR_MODEL=small .case-venv/bin/python scripts/asr_case.py
set -a; . ./.env; set +a
.case-venv/bin/python scripts/build_case_graph.py
```

The synthetic long-dialogue evaluation is `scripts/long_dialogue_eval.py`.
It uses DeepSeek for a fictional visitor and the counselor across two sessions
for one isolated user, waits for each five-turn memory batch, and saves dialogue,
routes, memory counts, and front/back-end checks in `checkpoints/long-dialogue-eval.json`.
The roleplay is synthetic; it is not a transcript or claim about the person in
the video.

The browser can create multiple sessions for one anonymous user and reopen their
history. Each counselor response displays the memory route and rewritten query;
the fifth user turn also displays the queued memory consolidation job.
