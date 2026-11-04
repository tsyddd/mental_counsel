-- Additive migration: legacy videos and three-layer memory remain intact.
CREATE TABLE IF NOT EXISTS mt_events (
 id uuid PRIMARY KEY, user_id text NOT NULL, content text NOT NULL,
 subject text NOT NULL DEFAULT 'user', occurred_at text, source_ids jsonb NOT NULL,
 source_session uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 start_turn integer NOT NULL, end_turn integer NOT NULL, embedding vector(384),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mt_events_user ON mt_events(user_id);
CREATE TABLE IF NOT EXISTS mt_intents (
 id uuid PRIMARY KEY, user_id text NOT NULL, label text NOT NULL,
 category text NOT NULL, status text NOT NULL DEFAULT 'tentative',
 source_session uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 extracted_turn integer NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mt_intents_scope ON mt_intents(user_id,status);
CREATE TABLE IF NOT EXISTS mt_intent_events (
 intent_id uuid REFERENCES mt_intents(id) ON DELETE CASCADE,
 event_id uuid REFERENCES mt_events(id) ON DELETE CASCADE,
 relation_type text NOT NULL DEFAULT 'support', PRIMARY KEY(intent_id,event_id)
);
CREATE TABLE IF NOT EXISTS mt_snapshots (
 user_id text PRIMARY KEY, version bigint NOT NULL DEFAULT 1,
 session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 processed_turn integer NOT NULL, content text NOT NULL,
 event_ids jsonb NOT NULL, intent_ids jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS mt_confirmations (
 id uuid PRIMARY KEY, user_id text NOT NULL, kind text NOT NULL,
 payload jsonb NOT NULL, question text NOT NULL, status text NOT NULL DEFAULT 'pending',
 source_session uuid REFERENCES sessions(id) ON DELETE CASCADE,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS mt_jobs (
 id uuid PRIMARY KEY, user_id text NOT NULL,
 session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 start_turn integer NOT NULL, end_turn integer NOT NULL,
 consolidate boolean NOT NULL DEFAULT false, status text NOT NULL DEFAULT 'queued',
 attempts integer NOT NULL DEFAULT 0, error text, result jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(session_id,end_turn)
);
CREATE INDEX IF NOT EXISTS mt_jobs_pending ON mt_jobs(user_id,status,created_at);
CREATE TABLE IF NOT EXISTS mt_session_state (
 session_id uuid PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
 processed_turn integer NOT NULL DEFAULT 0, closed boolean NOT NULL DEFAULT false
);
