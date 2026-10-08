"""Pre-download the two free local models used by E2/E3 (no paid APIs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

for mid in ("t5-small", "Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"fetching {mid} ...", flush=True)
    AutoTokenizer.from_pretrained(mid)
    AutoModelForSeq2SeqLM.from_pretrained(mid) if mid == "t5-small" else AutoModelForCausalLM.from_pretrained(mid)
    print(f"  ok {mid}", flush=True)
print("models cached")
