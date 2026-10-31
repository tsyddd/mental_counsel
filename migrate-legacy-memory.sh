set -e
docker exec -i interview-platform-postgres-1 psql -v ON_ERROR_STOP=1 -U interview -d interview <<'SQL'
UPDATE memories m
SET layer='session',
    scope_session_id=(m.metadata->>'session_id')::uuid,
    expires_at=COALESCE(m.expires_at,m.created_at+interval '30 days')
WHERE m.kind IN ('episode_summary','state')
  AND m.metadata ? 'session_id'
  AND EXISTS (SELECT 1 FROM sessions s WHERE s.id=(m.metadata->>'session_id')::uuid);
SQL
