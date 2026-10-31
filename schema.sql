CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS app_users (
  id uuid PRIMARY KEY, email text NOT NULL UNIQUE, display_name text NOT NULL,
  password_hash text NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_preference_docs (
  user_id uuid PRIMARY KEY REFERENCES app_users(id) ON DELETE CASCADE,
  notes text NOT NULL DEFAULT '', updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS demo_imports (
  user_id uuid PRIMARY KEY REFERENCES app_users(id) ON DELETE CASCADE,
  latest_session_id uuid, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS login_sessions (
  token_hash text PRIMARY KEY, user_id uuid NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
  expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS login_sessions_user ON login_sessions(user_id,expires_at);
CREATE TABLE IF NOT EXISTS videos (
  id text PRIMARY KEY, path text NOT NULL, status text NOT NULL DEFAULT 'queued',
  captions jsonb, graph jsonb, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY, kind text NOT NULL, status text NOT NULL DEFAULT 'queued',
  progress integer NOT NULL DEFAULT 0, payload jsonb NOT NULL,
  result jsonb, error text, created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS graph_nodes (
  video_id text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  id text NOT NULL, kind text NOT NULL, parent_id text,
  content text NOT NULL, start_sec double precision, end_sec double precision,
  data jsonb NOT NULL DEFAULT '{}', embedding vector(384),
  PRIMARY KEY(video_id,id)
);
CREATE INDEX IF NOT EXISTS graph_nodes_vec ON graph_nodes USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS graph_edges (
  video_id text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  src text NOT NULL, dst text NOT NULL, relation text NOT NULL,
  evidence text, PRIMARY KEY(video_id,src,dst,relation)
);
CREATE TABLE IF NOT EXISTS reports (
  video_id text PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
  report jsonb NOT NULL, profile jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sessions (
  id uuid PRIMARY KEY, user_id text NOT NULL, video_id text REFERENCES videos(id),
  created_at timestamptz NOT NULL DEFAULT now(), last_message_at timestamptz,
  episode_no integer NOT NULL DEFAULT 1,
  agent_mode text NOT NULL DEFAULT 'counselor'
);
CREATE TABLE IF NOT EXISTS trace_events (
  id uuid PRIMARY KEY, user_id text NOT NULL, session_id uuid REFERENCES sessions(id) ON DELETE CASCADE,
  kind text NOT NULL, title text NOT NULL, detail jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trace_events_session ON trace_events(session_id,created_at,id);
CREATE TABLE IF NOT EXISTS episodes (
  session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  episode_no integer NOT NULL, status text NOT NULL DEFAULT 'active',
  started_at timestamptz NOT NULL DEFAULT now(), ended_at timestamptz,
  consolidated_at timestamptz,
  PRIMARY KEY(session_id,episode_no)
);
CREATE TABLE IF NOT EXISTS messages (
  id uuid PRIMARY KEY, session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  episode_no integer NOT NULL DEFAULT 1, role text NOT NULL, content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_time ON messages(session_id,created_at);
CREATE TABLE IF NOT EXISTS memories (
  id uuid PRIMARY KEY, user_id text NOT NULL, kind text NOT NULL,
  layer text NOT NULL DEFAULT 'user', scope_session_id uuid REFERENCES sessions(id) ON DELETE CASCADE,
  content text NOT NULL, status text NOT NULL DEFAULT 'active',
  source_type text NOT NULL, source_id text, confidence real NOT NULL DEFAULT 0.5,
  salience real NOT NULL DEFAULT 1.0, reinforcement_count integer NOT NULL DEFAULT 1,
  retrieval_count integer NOT NULL DEFAULT 0,
  last_accessed_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}', embedding vector(384),
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz
);
CREATE INDEX IF NOT EXISTS memories_user ON memories(user_id,status);
CREATE INDEX IF NOT EXISTS memories_vec ON memories USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS tool_runs (
  id uuid PRIMARY KEY, user_id text, session_id uuid REFERENCES sessions(id) ON DELETE SET NULL,
  tool_name text NOT NULL, input_digest text NOT NULL, status text NOT NULL,
  summary text NOT NULL DEFAULT '', error_summary text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_runs_scope ON tool_runs(user_id,session_id,created_at DESC);
CREATE TABLE IF NOT EXISTS concepts (
  id uuid PRIMARY KEY, user_id text NOT NULL, label text NOT NULL,
  keywords text[] NOT NULL DEFAULT '{}', embedding vector(384), strength real NOT NULL DEFAULT 1.0,
  support_count integer NOT NULL DEFAULT 1, memory_ids uuid[] NOT NULL DEFAULT '{}',
  last_seen_at timestamptz NOT NULL DEFAULT now(), UNIQUE(user_id,label)
);
CREATE TABLE IF NOT EXISTS intents (
  id uuid PRIMARY KEY, user_id text NOT NULL, label text NOT NULL,
  status text NOT NULL DEFAULT 'tentative', concept_ids uuid[] NOT NULL DEFAULT '{}',
  support_count integer NOT NULL DEFAULT 1,
  retrieval_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(user_id,label)
);

-- Idempotent upgrades for databases created by earlier demo versions.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS layer text NOT NULL DEFAULT 'user';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS scope_session_id uuid REFERENCES sessions(id) ON DELETE CASCADE;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS salience real NOT NULL DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS reinforcement_count integer NOT NULL DEFAULT 1;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed_at timestamptz;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS retrieval_count integer NOT NULL DEFAULT 0;
ALTER TABLE intents ADD COLUMN IF NOT EXISTS retrieval_count integer NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS memory_batches (
  session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  batch_no integer NOT NULL,
  job_id uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(session_id,batch_no)
);
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS keywords text[] NOT NULL DEFAULT '{}';
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS strength real NOT NULL DEFAULT 1.0;
CREATE INDEX IF NOT EXISTS memories_scope_layer ON memories(user_id,layer,status);
CREATE INDEX IF NOT EXISTS concepts_vec ON concepts USING hnsw (embedding vector_cosine_ops);
