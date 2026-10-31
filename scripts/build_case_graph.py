"""Build the real demo graph without starting Docker."""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from relation_videoarm.api import APIClient
from relation_videoarm.graph import build_graph

checkpoint = root / "checkpoints" / "interview-demo"
captions = checkpoint / "captions.json"
if not captions.exists():
    raise SystemExit("ASR captions are not ready")

graph = build_graph(APIClient(), captions, checkpoint_dir=checkpoint / "graph")
(checkpoint / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"macro_events": len(graph["macro_events"]),
                  "super_events": len(graph["super_events"]),
                  "duration": graph["duration"],
                  "graph": str(checkpoint / "graph.json"),
                  "usage": {k: graph["build_usage"][k] for k in ("calls", "total_tokens")}},
                 ensure_ascii=False))
