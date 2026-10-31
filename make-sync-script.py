import base64
from pathlib import Path, PurePosixPath

root = Path(__file__).parent
remote = PurePosixPath("/media/4t_2/hanhp/interview-platform")
paths = [
    "schema.sql", "docker-compose.yml", ".env.example", "python/api.py", "python/core.py", "python/worker.py", "python/demo_case.py",
    "python/memory_retrieval.py",
    "go/main.go", "go/auth.go", "go/experience.go", "web/index.html", "web/style.css", "web/app.js",
    "README.md", "CASE_STUDY.md", "MEMORY_DESIGN.md",
    "LONG_DIALOGUE_EVAL_REPORT.md",
    "scripts/roleplay_memory_demo.py", "scripts/long_dialogue_eval.py",
    "scripts/summarize_long_eval.py", "scripts/eval_cross_session.py",
    "scripts/intent_ablation.py",
    "scripts/validate_intent_merge.py",
    "scripts/validate_pinned_preference.py",
]
lines = ["set -euo pipefail"]
for name in paths:
    data = base64.b64encode((root / name).read_bytes()).decode("ascii")
    target = str(remote / name)
    lines += [f"mkdir -p '{str((remote / name).parent)}'",
              f"cat > /tmp/interview-sync-{Path(name).name}.b64 <<'CODEX_B64'",
              data, "CODEX_B64",
              f"base64 -d /tmp/interview-sync-{Path(name).name}.b64 > '{target}'",
              f"rm -f /tmp/interview-sync-{Path(name).name}.b64"]

# The Go build context is on ext4 to avoid the Docker VFS/NTFS slowdown.
for name in ["main.go", "auth.go", "experience.go"]:
    source = base64.b64encode((root / "go" / name).read_bytes()).decode("ascii")
    lines += ["mkdir -p /home/dashan/interview-platform-build/go",
              f"cat > /tmp/interview-{name}.b64 <<'CODEX_B64'", source, "CODEX_B64",
              f"base64 -d /tmp/interview-{name}.b64 > /home/dashan/interview-platform-build/go/{name}",
              f"rm -f /tmp/interview-{name}.b64"]
(root / "sync-dashan.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")

