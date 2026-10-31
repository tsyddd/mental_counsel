"""Prepare reusable timestamped subtitles for the supplied demo video."""
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel

root = Path(__file__).resolve().parents[1]
video = root / "media" / "interview.mp4"
if not video.is_file():
    raise SystemExit("missing media/interview.mp4")
target = root / "checkpoints" / "interview-demo"
target.mkdir(parents=True, exist_ok=True)
model_name = os.getenv("ASR_MODEL", "small")
device = os.getenv("ASR_DEVICE", "cpu")
model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
segments, _ = model.transcribe(str(video), language="zh", vad_filter=True,
                               initial_prompt="这是一次中文师生访谈。")
rows = [{"time_start_secs": round(x.start, 2), "time_end_secs": round(x.end, 2),
         "caption": x.text.strip()} for x in segments if x.text.strip()]
if not rows:
    raise SystemExit("no speech segments")
(target / "captions.json").write_text(json.dumps({"data": rows}, ensure_ascii=False), encoding="utf-8")
signature = {"size": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns, "model": model_name}
(target / "asr.meta.json").write_text(json.dumps(signature), encoding="utf-8")
print(json.dumps({"segments": len(rows), "duration_sec": rows[-1]["time_end_secs"],
                  "captions": str(target / "captions.json")}, ensure_ascii=False))
