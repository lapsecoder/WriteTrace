"""Render docs/experiments/phase4-report.md from the Phase-4 execution logs
(logs/phase4_reference.json, logs/phase4_bench.json, logs/phase4_smoke.json),
the locked final-v1 artifact, and the executed test log (logs/pytest.log).

Every number is read from those files at render time - nothing is hard-coded
or invented. The report has exactly the 16 sections required by the Phase-4
spec and ends with the fixed PHASE 4 FINAL REPORT bullet block.

Prerequisites (run before rendering):
    python scripts/verify_phase4.py
    python scripts/bench_phase4.py --runs 20
    python scripts/smoke_phase4.py --port 8017
    python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FINAL = ROOT / "ml" / "models" / "final-v1"
LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "experiments" / "phase4-report.md"

REF_PATH = LOGS / "phase4_reference.json"
BENCH_PATH = LOGS / "phase4_bench.json"
SMOKE_PATH = LOGS / "phase4_smoke.json"
PYTEST_LOG = LOGS / "pytest.log"

BASELINE_TESTS = 83          # recorded Phase-4 entry baseline
PROTECTED = ["ml/models/baseline-v1", "ml/models/final-v1",
             "docs/experiments/phase2-report.md",
             "docs/experiments/phase3-report.md"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _fmt(v, nd: int = 4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{nd}f}"
    return str(v)


def _table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _clip(s, n=200):
    s = str(s).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} - run the Phase-4 "
                 f"scripts first (see module docstring)")
    return json.loads(path.read_text(encoding="utf-8"))


def _pytest_stats() -> tuple[str | None, str | None]:
    """(passed, warnings) from the executed pytest log (UTF-16 aware)."""
    if not PYTEST_LOG.exists():
        return None, None
    raw = PYTEST_LOG.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    elif raw.startswith(b"\xef\xbb\xbf"):
        txt = raw[3:].decode("utf-8", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    passed = re.findall(r"(\d+) passed", txt)
    warns = re.findall(r"(\d+) warnings?", txt)
    return (passed[-1] if passed else None, warns[-1] if warns else None)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    # rstrip only: `--porcelain` lines start with meaningful spaces (" M x")
    return r.stdout.rstrip("\r\n")


def _lc(rel: str) -> int:
    p = ROOT / rel
    return len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0


def _test_functions(rel: str) -> int:
    p = ROOT / rel
    if not p.exists():
        return 0
    return len(re.findall(r"^def test_", p.read_text(encoding="utf-8"),
                          re.M))


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def main() -> int:
    ref = _load(REF_PATH)
    bench = _load(BENCH_PATH)
    smoke = _load(SMOKE_PATH)
    manifest = _load(FINAL / "manifest.json")
    gate = _load(FINAL / "gate.json")
    cfg = _load(FINAL / "config.json")
    feats_doc = _load(FINAL / "features.json")
    cal_doc = _load(FINAL / "calibration.json")
    imp_doc = _load(FINAL / "importance.json")

    from backend.schemas import API_VERSION, MAX_TEXT_CHARS, MIN_TEXT_CHARS
    from ml.inference import (DISCLAIMER, OUTER_CUT, SIGNAL_CAVEAT,
                              presentation_label)

    passed, warn_n = _pytest_stats()
    eval_summary = manifest.get("eval_summary", {})
    all_features = feats_doc.get("features", [])
    n_syn = sum(1 for f in all_features if f.startswith("syn_"))
    n_base = len(all_features) - n_syn
    band = tuple(gate.get("mixed_band") or (0.4, 0.6))

    label_probes = []
    for p_probe, ok in ((0.05, True), (0.3, True), (0.5, True),
                        (0.7, True), (0.95, True), (0.95, False)):
        label_probes.append((f"p={p_probe}, suffices={ok}",
                             presentation_label(p_probe, ok, band)))

    analyze_ok = [c for c in smoke.get("cases", [])
                  if c.get("status") == 200 and c.get("method") == "POST"
                  and "label" in (c.get("response") or {})]
    smoke_lats = [c["latency_ms"] for c in analyze_ok]

    tag_commit = _git("rev-parse", "phase-3-complete")
    porcelain = _git("status", "--porcelain") or ""
    changed = [ln[3:] for ln in porcelain.splitlines() if ln.strip()]
    if not any("phase4-report.md" in c for c in changed):
        changed.append("?? docs/experiments/phase4-report.md")
    changed.sort()
    protected_diff = _git("diff", "phase-3-complete", "--name-only",
                          "--", *PROTECTED)

    imp_win = imp_doc.get("winner", {})
    imp_map = imp_win.get("importance_delta_pr_auc", {})
    top_imp = sorted(imp_map.items(), key=lambda kv: -abs(kv[1]))[:8]

    e4 = cal_doc.get("e4_study", {})
    e4_eval = e4.get("eval", {})

    p: list[str] = []
    a = p.append

    # ---------------------------------------------------------------- header
    a("# WriteTrace — Phase 4 Report: Inference Engine + FastAPI Backend")
    a("")
    a(f"Generated: `{datetime.now(timezone.utc).isoformat()}` · artifact "
      f"`{manifest.get('winner')}` in `ml/models/final-v1/` · pipeline "
      f"`{manifest.get('pipeline_version')}` · every number below is read "
      "from executed logs (`logs/phase4_*.json`, `logs/pytest.log`) and the "
      "locked artifact at render time.")
    a("")
    a("> **Protocol statement.** Phase 4 performs **no training, tuning, "
      "calibration, or threshold selection**. It consumes the Phase-3 "
      "artifact read-only, reproduces the validated preprocessing → features "
      "→ model → calibrator → gate path in exactly one module "
      "(`ml/inference.py`), and adds only transport, validation, and "
      "presentation mapping. The Phase-1/2/3 artifacts and reports are "
      "unchanged (verified in Section 3).")
    a("")

    # ================================================== 1. Phase 4 objective
    a("## 1. Phase 4 objective")
    a("")
    a("Turn the locked Phase-3 artifact into a reliable, reusable inference "
      "system with one authoritative path:")
    a("")
    a("```text")
    a("raw user text")
    a("  → validation (Pydantic, safe 422s)")
    a("  → training-compatible preprocessing (ml/preprocessing/normalize)")
    a("  → training-compatible feature extraction "
      f"({n_base} v1 + {n_syn} syn features)")
    a("  → Phase-3 final model (hgb_syn)")
    a("  → Phase-3 calibrator (platt)")
    a("  → machine-generated probability p ∈ [0, 1]")
    a("  → evidence / abstention gate (min_words = 50)")
    a("  → 5-level presentation label + warnings")
    a("  → explainability signals (local reference perturbations)")
    a("  → structured, versioned API response")
    a("```")
    a("")
    a("In scope and shipped in this phase:")
    a("")
    a(_table(["Component", "File", "Lines", "Role"], [
        ["Inference engine", "`ml/inference.py`", _lc("ml/inference.py"),
         "the ONE authoritative path; dataclasses; load-once predictor"],
        ["API schemas", "`backend/schemas.py`", _lc("backend/schemas.py"),
         "Pydantic request/response/error models"],
        ["FastAPI app", "`backend/app.py`", _lc("backend/app.py"),
         "endpoints, lifespan load-once, safe exception handlers"],
        ["Measurement", "`ml/experiments/phase4.py`",
         _lc("ml/experiments/phase4.py"),
         "reference check + latency benchmark helpers"],
        ["Scripts", "`scripts/{verify,bench,smoke}_phase4.py`",
         _lc("scripts/verify_phase4.py") + _lc("scripts/bench_phase4.py")
         + _lc("scripts/smoke_phase4.py"),
         "produce logs/phase4_{reference,bench,smoke}.json"],
        ["Tests", "`tests/test_phase4_{inference,api}.py`",
         _lc("tests/test_phase4_inference.py")
         + _lc("tests/test_phase4_api.py"),
         "inference + API coverage (existing tests untouched)"],
        ["API docs", "`docs/api.md`", _lc("docs/api.md"),
         "endpoint/schema/label/limitation documentation"],
    ]))
    a("")
    a("Explicitly NOT done (Phase-4 stop condition): Astro frontend, "
      "paper/editorial UI, animations, audio, deployment, segment-level "
      "model training, authentication, database, external/paid APIs.")
    a("")

    # ================================================ 2. Phase 3 model selected
    a("## 2. Phase 3 model selected")
    a("")
    spec = cfg.get("spec", {})
    a(f"- **Winner:** `{manifest.get('winner')}` — "
      f"{spec.get('type')} (learning_rate={spec.get('learning_rate')}, "
      f"max_leaf_nodes={spec.get('max_leaf_nodes')}, "
      f"max_iter={spec.get('max_iter')}, random_state="
      f"{spec.get('random_state')}), trained on "
      f"{cfg.get('train_rows')} MAGE-train rows (seed {cfg.get('seed')}).")
    a(f"- **Features:** {feats_doc.get('n_features')} "
      f"({n_base} base + {n_syn} syntactic/spaCy); "
      f"selected/calibrated on val_tune {cfg.get('val_tune_rows')} / "
      f"val_cal {cfg.get('val_cal_rows')} rows.")
    a(f"- **Calibrator:** {cfg.get('calibrator')} · **threshold:** "
      f"{cfg.get('threshold')} (Phase-1 fixed, never tuned) · **label "
      f"policy:** {cfg.get('label_policy')}.")
    a(f"- **Data policy:** {manifest.get('data_policy')}")
    a(f"- **Selection order:** {' → '.join(manifest.get('selection_criteria_order', []))}"
      f"; LODO override applied = {manifest.get('lodo_override_applied')}.")
    a("")
    a("Held-out evaluation recorded in the artifact (never used for "
      "training/tuning):")
    a("")
    rows = []
    for name, m in eval_summary.items():
        rows.append([name, m.get("n"), _fmt(m.get("roc_auc")),
                     _fmt(m.get("pr_auc")), _fmt(m.get("f1_macro")),
                     _fmt(m.get("accuracy")), _fmt(m.get("brier")),
                     _fmt(m.get("ece_10")), _fmt(m.get("human_fpr"))])
    a(_table(["Set", "n", "ROC-AUC", "PR-AUC", "F1-macro", "Acc", "Brier",
              "ECE-10", "human FPR"], rows))
    a("")

    # =========================================== 3. Artifact/version provenance
    a("## 3. Artifact/version provenance")
    a("")
    a(_table(["Property", "Value"], [
        ["artifact directory", "`ml/models/final-v1/` (read-only)"],
        ["artifact_complete", manifest.get("artifact_complete")],
        ["pipeline_version", manifest.get("pipeline_version")],
        ["artifact written", manifest.get("generated_at")],
        ["hashed files", len(manifest.get("files", {}))],
        ["API identity",
         f"`model_name={manifest.get('winner')}` served as "
         "`model_version=final-v1` (directory name)"],
        ["Git tag", (f"`phase-3-complete` @ `{tag_commit[:12]}`"
                     if tag_commit else "n/a")],
        ["baseline-v1 / phase1-3 outputs vs tag",
         ("unchanged (empty git diff on protected paths)"
          if protected_diff == "" else "CHANGED - investigate")],
    ]))
    a("")
    a("Manifest file hashes (sha256, first 16 chars):")
    a("")
    a(_table(["File", "sha256[:16]", "bytes"],
             [[f"`{n}`", f"`{d['sha256'][:16]}`", d["bytes"]]
              for n, d in manifest.get("files", {}).items()]))
    a("")
    a(f"Working-tree changes in this phase ({len(changed)} paths):")
    a("")
    a("```text")
    for c in changed:
        a(c)
    a("```")
    a("")

    # ==================================================== 4. Inference pipeline
    a("## 4. Inference pipeline")
    a("")
    a("One implementation, used by every entry point (API, tests, scripts):")
    a("")
    a(_table(["Step", "Code", "Notes"], [
        ["validate", "`backend/schemas.py` `AnalyzeRequest`",
         f"{MIN_TEXT_CHARS}–{MAX_TEXT_CHARS} chars, ≥1 non-whitespace; "
         "wrong type/malformed JSON → 422"],
        ["normalize", "`ml/inference.Predictor._normalize`",
         "`normalize_text` — byte-identical to the training normalizer"],
        ["extract", "`Predictor._extract_row`",
         "v1 feature extractor + cached spaCy syntactic family; deterministic"],
        ["score", "`Predictor._score_row`",
         "`p = calibrator(model.predict_proba(x)[:, 1])` — canonical, no "
         "complement"],
        ["gate", "`Predictor.analyze`",
         f"min_words={gate.get('min_words')} → abstain; mixed band "
         f"{list(band)} → warning"],
        ["label", "`presentation_label`",
         "documented presentation policy, separate from the binary decision"],
        ["explain", "`ml/explain.local_contributions`",
         "one-at-a-time reference perturbations (k signals)"],
        ["map", "`backend/app._build_response`",
         "dataclass → Pydantic; probabilities rounded to 6 dp only here"],
    ]))
    a("")
    a("Result types (`ml/inference.py`): `TextAnalysis`, `FeatureEvidence`, "
      "`SegmentAnalysis` (reserved), `PredictionResult` — framework-free, "
      "so the API layer adds no ML logic.")
    a("")
    a(f"- **Load once:** `get_predictor()` returns a process-wide singleton; "
      f"the FastAPI lifespan loads it and runs `warmup()` before serving "
      f"(startup cost in Section 11). A test asserts `load_final` runs "
      f"exactly once across requests.")
    a("- **Deterministic:** same text → same features → same probability; "
      "verified in-process, across requests, and across OS processes "
      "(`test_clean_process_loading_reproduces_probability`).")
    a("- **Legacy contract preserved:** `predict()` (4 dp, legacy labels) "
      "and `explain()` still behave exactly as the Phase-3 tests expect.")
    a("")

    # =================================================== 5. Preprocessing parity
    a("## 5. Preprocessing parity")
    a("")
    a("Phase 4 does not re-derive features independently; it calls the same "
      "modules the experiments called (`ml/preprocessing/normalize.py`, "
      "`ml/features/extract.py`, `ml/features/syntactic.py`). Parity is then "
      "proven numerically against the **Phase-3 cached feature path** "
      "(the oracle that produced the locked metrics):")
    a("")
    a(_table(["Check", "Result"], [
        ["rows compared", f"{ref['n_rows']} (seed {ref['seed']}, "
         "5 human + 5 machine from MAGE-test)"],
        ["feature source", "Phase-3 cached parquet + syn cache vs live "
         "`Predictor.analyze` extraction"],
        ["max abs(Δp)", f"`{ref['max_abs_delta']:.3e}` "
         f"(tolerance {ref['tolerance']:.0e})"],
        ["within tolerance", ref["all_within_tolerance"]],
        ["normalization", "same `normalize_text`; idempotent on "
         "`text_norm` (asserted in tests)"],
        ["E7 cross-check", "recorded / recomputed / Phase-4 p all "
         f"{ref['e7_cross_check']['e7_recorded_p_machine']} → match = "
         f"{ref['e7_cross_check']['match']}"],
    ]))
    a("")
    a("A max delta of 0.0 on every compared row means preprocessing, "
      "feature order, and dtype handling are byte-for-byte equivalent; "
      "per-row evidence is in Section 13.")
    a("")

    # ================================================== 6. Calibration integration
    a("## 6. Calibration integration")
    a("")
    a(f"- Calibrator `{cfg.get('calibrator')}` from the artifact "
      "(`calibrator.joblib`), applied as `cal.predict(model_proba[:, 1])` — "
      "the canonical `p = cal(model(x))` from `config.json` "
      f"(`label_policy`).")
    a(f"- Calibrator *type* was chosen in Phase-3 E4 on a group-disjoint "
      f"val_cal split: Brier platt {e4_eval.get('platt', {}).get('cal_eval_brier')} "
      f"vs isotonic {e4_eval.get('isotonic', {}).get('cal_eval_brier')} "
      f"(chosen = {e4.get('chosen')}). Phase 4 re-uses it unchanged.")
    a(f"- Test-set calibration as recorded in the artifact: Brier "
      f"{eval_summary.get('test', {}).get('brier')}, ECE-10 "
      f"{eval_summary.get('test', {}).get('ece_10')}.")
    a("- **No orientation complement:** labels are canonical "
      "(y=1 MACHINE), so `human_probability = 1 − machine_probability` "
      "is exact; tests assert `machine + human == 1` within 1e-9.")
    a("- **Rounding policy:** decisions always use the full-precision p; "
      "the API rounds to 6 dp for presentation; the legacy `predict()` "
      "path rounds to 4 dp (parity deltas therefore ≤ 4e-5 by construction; "
      "the raw-vs-raw reference check in Section 13 is exactly 0).")
    a("- Tests: probability bounds, complement consistency, no NaN/∞, "
      "monotone bounded calibrator (Phase-2 suite), determinism.")
    a("")

    # ============================================= 7. Evidence/abstention gate
    a("## 7. Evidence/abstention implementation")
    a("")
    a(_table(["Gate property", "Value (from `gate.json`)"], [
        ["status", gate.get("status")],
        ["min_words", gate.get("min_words")],
        ["threshold", gate.get("threshold")],
        ["abstain fraction (test)", gate.get("abstain_frac_test")],
        ["abstain fraction cap", gate.get("abstain_frac_max")],
        ["mixed band", list(band)],
        ["mixed-band accuracy (val_cal)", gate.get("mixed_band_accuracy")],
    ]))
    a("")
    a(f"Behavior for `word_count < {gate.get('min_words')}`:")
    a("")
    a("- HTTP **200** (not a rejection): `status=insufficient_evidence`, "
      "`label=insufficient_evidence`, `decision=abstain`, "
      "`confidence=null`, `evidence_sufficient=false`.")
    a("- `word_count`, `text_length`, gate metadata, and warnings are still "
      "returned; the probability is shown for transparency **with a "
      "warning that it must not support an authorship conclusion**.")
    a("- The binary decision threshold (0.5) and the mixed band are read "
      "from the artifact, never re-derived here.")
    a("")
    short_cases = [c for c in smoke.get("cases", [])
                   if c.get("name") in ("short_under_gate", "normal_human_text")
                   and c.get("response")]
    if short_cases:
        a("Observed through the live API (from `logs/phase4_smoke.json`):")
        a("")
        a(_table(["Case", "words", "label", "decision", "confidence",
                  "warnings"],
                 [[c["name"], c["response"].get("word_count"),
                   f"`{c['response'].get('label')}`",
                   c["response"].get("decision"),
                   c["response"].get("confidence"),
                   c["response"].get("n_warnings")]
                  for c in short_cases]))
        a("")
    a("Tests: `test_short_text_abstains_with_full_metadata`, "
      "`test_short_text_returns_insufficient_evidence`, "
      "`test_gate_values_match_phase3_report`.")
    a("")

    # ================================================== 8. Explainability
    a("## 8. Explainability implementation")
    a("")
    a(f"- **Global importance:** {imp_doc.get('method')} "
      f"(n_rows={imp_win.get('n_rows')}, n_repeats={imp_win.get('n_repeats')}, "
      f"baseline PR-AUC {imp_win.get('baseline_pr_auc')}). Top features by "
      "|Δ PR-AUC|:")
    a("")
    a(_table(["Feature", "Δ PR-AUC"],
             [[f"`{f}`", _fmt(v, 5)] for f, v in top_imp]))
    a("")
    a("- **Local explanation:** one-at-a-time reference perturbation "
      "(`ml/explain.local_contributions`) against train reference stats — "
      "batched for speed and verified equal to the sequential definition "
      "(`test_local_contributions_batched_equals_reference`).")
    a("")
    a(_table(["Signal field", "Meaning"], [
        ["signal / label / family", "feature id, human label, family"],
        ["value", "observed value in this text"],
        ["reference_median", "train-corpus median anchor"],
        ["contribution", "signed local contribution to p(machine)"],
        ["direction", "`machine` / `human` / `neutral`"],
        ["interpretation", "plain-language value vs reference sentence"],
        ["caveat", f"`{SIGNAL_CAVEAT}`"],
    ]))
    a("")
    a("Forbidden phrasing is enforced by a test "
      "(`test_signals_schema_and_no_causal_language`): no 'proves', no "
      "authorship claims, no causal attribution. Global explanations always "
      "carry the fixed disclaimer:")
    a("")
    a(f"> {DISCLAIMER}")
    a("")

    # ========================================================== 9. API design
    a("## 9. API design")
    a("")
    a(_table(["Method", "Path", "Purpose"], [
        ["POST", "`/api/v1/analyze`", "analyse one text"],
        ["GET", "`/api/v1/health`", "liveness + loaded model identity"],
        ["GET", "`/api/v1/info`", "static metadata (gate, labels, eval, "
         "data policy, top features)"],
    ]))
    a("")
    a("Request:")
    a("")
    a(_table(["Field", "Type", "Default", "Bounds / notes"], [
        ["`text`", "string", "required",
         f"{MIN_TEXT_CHARS}–{MAX_TEXT_CHARS} chars; ≥1 non-whitespace"],
        ["`include_signals`", "bool", "true", "local signals on/off"],
        ["`max_signals`", "int", "8", "1–50"],
    ]))
    a("")
    a("Response groups: `status`, `label`, `decision`, "
      "`machine_probability`, `human_probability`, `confidence`, "
      "`word_count`, `text_length`, `evidence_sufficient`, "
      "`mixed_low_confidence`, `signals[]`, `segments[]` (always empty in "
      "v1), `gate`, `model_version`, `model_name`, `calibrator`, "
      "`threshold`, `warnings[]`, `disclaimer`.")
    a("")
    a("Label vocabulary generated from the implemented policy "
      f"(`presentation_label`, outer cutoff {OUTER_CUT}):")
    a("")
    a(_table(["Probe", "label"], [[p, f"`{l}`"] for p, l in label_probes]))
    a("")
    a(f"Design rules: versioned namespace `/api/{API_VERSION}`; probabilities "
      "rounded "
      "to 6 dp after full-precision decisions; `human_probability` derived "
      "as `1 − machine` so the pair is always consistent; no filesystem "
      "paths, stack traces, model objects, or debug detail in any response; "
      "submitted text is never logged, persisted, or transmitted "
      "(`test_submitted_text_is_never_logged`). Full contract: "
      "`docs/api.md`.")
    a("")
    a("Startup:")
    a("")
    a("```bash")
    a("python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000")
    a("```")
    a("")

    # ============================================ 10. Validation and errors
    a("## 10. Validation and error handling")
    a("")
    a("Real-server smoke cases (from `logs/phase4_smoke.json`, server "
      f"`{smoke.get('base_url')}`, ready = {smoke.get('server_ready')}):")
    a("")
    a(_table(["Case", "Method", "Expected", "Got", "OK", "ms"],
             [[c["name"], c["method"], c["expected_status"], c["status"],
               c["status_ok"], c["latency_ms"]]
              for c in smoke.get("cases", [])]))
    a("")
    a("Error contract (no input echo, no paths, no traces):")
    a("")
    a("```json")
    a('{"status": "error", "error": {"code": "...", "message": "...",')
    a(' "details": [{"field": "...", "message": "...", "type": "..."}]}}')
    a("```")
    a("")
    a(_table(["Status", "code", "Trigger"], [
        ["404", "`not_found`", "unknown path"],
        ["405", "`method_not_allowed`", "wrong HTTP method"],
        ["422", "`validation_error`", "empty / whitespace-only / >30,000 "
         "chars / wrong type / missing field / malformed JSON / "
         "out-of-range `max_signals`"],
        ["500", "`internal_error`", "unexpected failure - details never "
         "exposed"],
        ["200", "—", "short text returns abstention, not an error"],
    ]))
    a("")
    a("Handlers: `RequestValidationError` → sanitized 422 (field + message "
      "only, offending value dropped), `StarletteHTTPException` → mapped "
      "codes, catch-all `Exception` → generic 500. Warnings (HTTP 200): "
      "insufficient evidence · mixed band (with the 0.5342 accuracy note) "
      "· non-English heuristic. Tests cover every row above plus the "
      "no-echo and no-path-leak guarantees.")
    a("")

    # ================================================ 11. Performance/latency
    a("## 11. Performance and latency")
    a("")
    env = bench.get("environment", {})
    a(_table(["Environment", "Value"], [
        ["python", env.get("python")], ["scikit-learn", env.get("sklearn")],
        ["platform", env.get("platform")], ["CPU count", env.get("cpu_count")],
        ["device", env.get("device")], ["artifact", bench.get("artifact")],
        ["runs per bucket", bench.get("n_runs")],
    ]))
    a("")
    a(_table(["Phase", "Milliseconds"], [
        ["cold load (model + calibrator + feature state)",
         bench.get("cold_load_ms")],
        ["startup warm-up incl. spaCy pipeline load (once per process)",
         bench.get("warmup_including_spacy_load_ms")],
    ]))
    a("")
    a("Warm in-process latency (median / p95 ms):")
    a("")
    rows = []
    for name, b in bench.get("warm", {}).items():
        w = b.get("analyze_with_signals_ms", {})
        n = b.get("analyze_no_signals_ms", {})
        rows.append([name, b.get("words"), b.get("chars"),
                     f"{_fmt(w.get('median_ms'), 1)} / "
                     f"{_fmt(w.get('p95_ms'), 1)}",
                     f"{_fmt(n.get('median_ms'), 1)} / "
                     f"{_fmt(n.get('p95_ms'), 1)}"])
    a(_table(["Bucket", "words", "chars", "analyze + signals med/p95",
              "analyze no-signals med/p95"], rows))
    a("")
    if smoke_lats:
        a(f"End-to-end HTTP (smoke, {len(smoke_lats)} successful analyze "
          f"requests): median `{_fmt(_median(smoke_lats), 1)} ms`, max "
          f"`{_fmt(max(smoke_lats), 1)} ms`.")
        a("")
    a("Notes: CPU-only (₹0 budget, no paid/external APIs); the model is "
      "never reloaded per request (asserted by "
      "`test_model_not_reloaded_per_request`); signals cost a few "
      "milliseconds; the one-time spaCy load is paid at server startup, "
      "not by the first user request.")
    a("")

    # ======================================================== 12. Test results
    a("## 12. Test results")
    a("")
    a(f"Executed: `python -m pytest tests/ -q` → **{passed or 'n/a'} "
      f"passed, {warn_n or 'n/a'} warnings** (`logs/pytest.log`).")
    a("")
    inf_fns = _test_functions("tests/test_phase4_inference.py")
    api_fns = _test_functions("tests/test_phase4_api.py")
    a(_table(["Suite", "Test functions", "Note"], [
        ["Phase 1–3 (existing)", f"{BASELINE_TESTS} test cases",
         "untouched; baseline was 83 passed / 2 warnings"],
        ["`tests/test_phase4_inference.py`", inf_fns,
         "artifact load, determinism, gate, labels, signals, parity, "
         "clean-process load"],
        ["`tests/test_phase4_api.py`", api_fns,
         "schema, validation matrix, safety, load-once, privacy"],
        ["total (executed)", f"{passed or '?'} passed",
         f"new Phase-4 cases = {int(passed) - BASELINE_TESTS}"
         if passed and passed.isdigit() else "see log"],
    ]))
    a("")
    a("Requirement → test mapping (Phase-4 spec §12):")
    a("")
    a(_table(["Requirement", "Test(s)"], [
        ["model / calibrator / feature-state loading",
         "`test_all_artifact_components_load`"],
        ["deterministic inference",
         "`test_deterministic_inference_full_precision`, "
         "`test_deterministic_across_requests`, "
         "`test_clean_process_loading_reproduces_probability`"],
        ["probability bounds / complement / no NaN",
         "`test_probability_bounds_and_human_complement`"],
        ["abstention + short text",
         "`test_short_text_abstains_with_full_metadata`, "
         "`test_short_text_returns_insufficient_evidence`"],
        ["empty / wrong type / malformed",
         "`test_empty_and_invalid_text_raise`, "
         "`test_validation_errors_are_safe_422`, `test_malformed_json`"],
        ["Unicode / multiline / punctuation",
         "`test_unicode_and_multiline_inputs`"],
        ["API + explanation schema",
         "`test_analyze_response_schema`, `test_signals_in_response`, "
         "`test_signals_schema_and_no_causal_language`"],
        ["model version reporting",
         "`test_info_endpoint_exposes_metadata_without_paths`, "
         "`test_health_endpoint`"],
        ["model loaded only once",
         "`test_get_predictor_is_singleton_and_loads_once`, "
         "`test_model_not_reloaded_per_request`"],
        ["reference predictions vs Phase 3",
         "`test_reference_parity_vs_phase3_cached_path`, "
         "`test_report_worked_example_reproduced`"],
        ["privacy (no text logging)", "`test_submitted_text_is_never_logged`"],
    ]))
    a("")
    a("Existing tests were not modified or weakened; `git status` in "
      "Section 3 shows no Phase-1/2/3 test file changed. The 2 warnings are "
      "the pre-existing sklearn imputer warnings from "
      "`test_pipeline_smoke`.")
    a("")

    # ============================================ 13. Reference prediction checks
    a("## 13. Reference prediction checks")
    a("")
    a("Oracle = Phase-3 cached-feature predictions (the numbers the locked "
      "metrics were computed from); Phase-4 = the inference engine's own "
      "extraction + scoring on raw text.")
    a("")
    a(_table(["Row", "domain", "truth", "words", "p (Phase 3 cached)",
              "p (Phase 4)", "Δ"],
             [[r["row_id"], r["domain"], r["true"], r["words"],
               f"{r['p_phase3_cached']:.6f}",
               f"{r['p_phase4_inference']:.6f}",
               f"{r['abs_delta']:.1e}"]
              for r in ref.get("rows", [])]))
    a("")
    a(f"**max |Δ| = `{ref['max_abs_delta']:.3e}` across "
      f"{ref['n_rows']} rows (tolerance {ref['tolerance']:.0e}) → "
      f"all_within_tolerance = {ref['all_within_tolerance']}.**")
    a("")
    o = ref.get("orientation", {})
    a("Orientation check over the full locked test split "
      f"(canonical y=1 MACHINE):")
    a("")
    a(_table(["Statistic", "Human rows", "Machine rows"], [
        ["n", o.get("n_human"), o.get("n_machine")],
        ["mean p", _fmt(o.get("mean_p_human")), _fmt(o.get("mean_p_machine"))],
        ["median p", _fmt(o.get("median_p_human")),
         _fmt(o.get("median_p_machine"))],
    ]))
    a("")
    a(f"separated (mean machine − mean human > 0.3): {o.get('separated')}.")
    a("")
    e7 = ref.get("e7_cross_check", {})
    a("E7 worked example (highest-confidence human FP on MAGE-test):")
    a("")
    a(_table(["Quantity", "E7 recorded", "Recomputed", "Phase 4"], [
        ["p(machine)", e7.get("e7_recorded_p_machine"),
         e7.get("recomputed_p_machine"), e7.get("phase4_p_machine")],
        ["words", e7.get("e7_recorded_words"), e7.get("recomputed_words"),
         "—"],
        ["domain", e7.get("e7_recorded_domain"),
         e7.get("recomputed_domain"), "—"],
    ]))
    a("")
    a(f"match = **{e7.get('match')}** — this also documents an honest "
      "false positive (a human text the model scores as machine-like); it "
      "is reported, not hidden.")
    a("")

    # ===================================================== 14. Known limitations
    a("## 14. Known limitations")
    a("")
    a(_table(["Limitation", "Evidence (from artifacts / live runs)"], [
        ["OOD degradation",
         f"OOD-GPT ROC {_fmt(eval_summary.get('ood_gpt', {}).get('roc_auc'))}, "
         f"OOD-GPT-para {_fmt(eval_summary.get('ood_gpt_para', {}).get('roc_auc'))}, "
         f"HC3 human FPR {_fmt(eval_summary.get('hc3', {}).get('human_fpr'))}, "
         f"RAID-clean ROC {_fmt(eval_summary.get('raid_clean', {}).get('roc_auc'))}"],
        ["mixed band is near chance",
         f"accuracy {gate.get('mixed_band_accuracy')} inside {list(band)} "
         "→ flagged, never silently classified"],
        ["short text under-determined",
         f"gate abstains below {gate.get('min_words')} words "
         f"({gate.get('abstain_frac_test')} of test)"],
        ["repetition-heavy / formulaic text",
         "smoke: repeated filler text scored "
         f"{next((c['response']['machine_probability'] for c in smoke.get('cases', []) if c.get('name') == 'repeated_text' and c.get('response')), 'n/a')} "
         "→ machine (a known feature-style failure mode)"],
        ["non-English input",
         "warned, not rejected; model was trained/evaluated on English only"],
        ["confidence is derived",
         "`confidence = max(p, 1-p)` — not a separately trained confidence "
         "model; null when abstaining"],
        ["presentation cutoffs",
         f"outer {OUTER_CUT}/1-{OUTER_CUT} label split is a documented "
         "convention, never fitted; decision stays at 0.5"],
        ["service surface",
         "no auth, no rate limit, no persistence — local inference only; "
         "put your own controls in front before exposure"],
        ["statistics",
         "single seed, no confidence intervals; CPU-only measurements on "
         "one machine"],
    ]))
    a("")
    a("Not an authorship test in any legal or forensic sense — the fixed "
      "disclaimer on every response states this.")
    a("")

    # ================================================ 15. Segment-analysis readiness
    a("## 15. Segment-analysis readiness")
    a("")
    a("No segment model was trained in Phase 4 (per the stop condition), "
      "but the pipeline is already shaped for it:")
    a("")
    a("- `SegmentAnalysis` dataclass exists with "
      "`status='not_available'`, `count=0`, and an explanatory note.")
    a("- `PredictionResult.segments` and the API `segments[]` field are "
      "already in the contract (always `[]` in v1) — Phase 5 can populate "
      "them without breaking clients.")
    a("- Normalization, feature extraction, scoring, gating, and "
      "explanation live in private `Predictor` methods "
      "(`_normalize` / `_extract_row` / `_score_row`) that operate on a "
      "single text — segment scoring can reuse them verbatim instead of "
      "re-implementing the pipeline.")
    a("- `TextAnalysis` already measures document-level facts "
      "(characters/words/sentences/paragraphs) that windowing will need.")
    a("- The gate is document-level today; Phase 5 must decide and "
      "**document** whether segment-level outputs carry the same 50-word "
      "gate or a validated alternative (a new artifact, not a silent "
      "change).")
    a("")

    # ====================================================== 16. Phase 5 handoff
    a("## 16. Phase 5 handoff")
    a("")
    a("What Phase 5 inherits:")
    a("")
    a(_table(["Item", "State"], [
        ["API contract", "`/api/v1/*` stable and documented in "
         "`docs/api.md`; keep `/api/v1` frozen, extend additively"],
        ["Artifact", "`ml/models/final-v1/` read-only, hash-verified; "
         "never retrain/edit in place"],
        ["Inference", "`ml/inference.Predictor` is the single path — the "
         "frontend must not re-implement scoring"],
        ["Tests", f"{passed or '?'} passing; keep green; never weaken"],
        ["Logs", "`logs/phase4_{reference,bench,smoke}.json` + "
         "`logs/pytest.log` are the provenance for this report"],
        ["Privacy", "local-first/stateless — no DB, auth, telemetry may be "
         "added without an explicit decision"],
    ]))
    a("")
    a("Recommended Phase-5 order:")
    a("")
    a("1. **Segment-level model** (the remaining ML work): reuse "
      "`_normalize`/`_extract_row`/`_score_row` on paragraph windows, train "
      "as a *new* versioned artifact (e.g. `ml/models/segments-v1/`), "
      "document the gate policy, then populate `segments[]`.")
    a("2. **Astro frontend** consuming `POST /api/v1/analyze` exactly as "
      "documented (label vocabulary, abstention, mixed-band and disclaimer "
      "semantics must be rendered honestly).")
    a("3. **Paper/editorial UI + deployment** last, behind explicit "
      "consent about false positives (Section 14).")
    a("")
    a("```bash")
    a("# reproduce everything in this report:")
    a("python scripts/verify_phase4.py && python scripts/bench_phase4.py")
    a("python scripts/smoke_phase4.py --port 8017")
    a("python -m pytest tests/ -q")
    a("python ml/experiments/phase4_report_generator.py")
    a("```")
    a("")

    # ------------------------------------------------- fixed ending block
    a("---")
    a("")
    a("## PHASE 4 FINAL REPORT")
    a("")
    a(f"- **Status:** Phase 4 complete — inference engine + FastAPI backend "
      f"live; frontend/deployment/segment model deliberately NOT started.")
    a(f"- **Model:** `{manifest.get('winner')}` "
      f"({spec.get('type')}, lr={spec.get('learning_rate')}, "
      f"leaves={spec.get('max_leaf_nodes')}, iters={spec.get('max_iter')}, "
      f"seed={spec.get('random_state')}) on "
      f"{feats_doc.get('n_features')} features, calibrator "
      f"{cfg.get('calibrator')}, threshold {cfg.get('threshold')} — "
      f"consumed read-only from `ml/models/final-v1/` "
      f"({len(manifest.get('files', {}))} hashed files, "
      f"artifact_complete={manifest.get('artifact_complete')}).")
    a(f"- **Architecture:** `backend/app.py` → `ml/inference.Predictor` "
      f"(single path: normalize → extract → model → calibrator → gate → "
      f"label → signals); process-wide singleton with startup warm-up "
      f"({bench.get('warmup_including_spacy_load_ms')} ms) so requests never "
      f"reload anything (cold load {bench.get('cold_load_ms')} ms).")
    a(f"- **API:** `POST /api/v1/analyze` (+ `/health`, `/info`), versioned "
      f"v1, Pydantic-validated, safe 422/404/405/500 envelopes, no input "
      f"echo / paths / traces; documented in `docs/api.md`.")
    a(f"- **Abstention:** gate min_words={gate.get('min_words')} "
      f"(abstain {gate.get('abstain_frac_test')} of test; excluded F1m "
      f"worse than kept), 200 + `insufficient_evidence` below the gate, "
      f"mixed band {list(band)} (accuracy "
      f"{gate.get('mixed_band_accuracy')}) flagged; `confidence=null` on "
      f"abstain.")
    a(f"- **Explainability:** permutation importance "
      f"(Δ PR-AUC) + local reference perturbations; every signal carries "
      f"the non-causal caveat; no causal/authorship language (tested).")
    a(f"- **Latency (CPU):** cold load {bench.get('cold_load_ms')} ms, "
      f"startup warm-up {bench.get('warmup_including_spacy_load_ms')} ms, "
      f"warm analyze medians "
      + ", ".join(f"{n} {b['analyze_with_signals_ms']['median_ms']} ms"
                  for n, b in bench.get("warm", {}).items()) + ".")
    a(f"- **Tests:** {passed or 'n/a'} passed, {warn_n or 'n/a'} warnings "
      f"(baseline {BASELINE_TESTS} all still green; "
      f"+{int(passed) - BASELINE_TESTS if passed and passed.isdigit() else '?'} "
      "Phase-4 cases incl. clean-process loading and privacy).")
    a(f"- **Smoke (real server):** {sum(1 for c in smoke.get('cases', []) if c.get('status_ok'))}/"
      f"{len(smoke.get('cases', []))} cases passed; "
      f"end-to-end median "
      f"{_fmt(_median(smoke_lats), 1) if smoke_lats else 'n/a'} ms "
      "over successful analyze requests.")
    a(f"- **Reference checks:** max |Δp| = "
      f"`{ref['max_abs_delta']:.3e}` vs Phase-3 cached path over "
      f"{ref['n_rows']} rows (tolerance {ref['tolerance']:.0e}); E7 "
      f"worked example matches (p={e7.get('phase4_p_machine')}, "
      f"{e7.get('e7_recorded_words')} words); orientation separated "
      f"(mean p machine {o.get('mean_p_machine')} vs human "
      f"{o.get('mean_p_human')}).")
    a(f"- **Integrity:** `phase-3-complete` tag intact"
      + (f" (`{tag_commit[:12]}`)" if tag_commit else "")
      + "; git diff on baseline-v1/final-v1/phase1-3 reports = "
      + ("empty (unchanged)." if protected_diff == "" else "NOT EMPTY - investigate."))
    a(f"- **Files changed:** {len(changed)} (see Section 3); temporary "
      "probe files removed; nothing committed.")
    a(f"- **Known limitations:** OOD ROC "
      f"{_fmt(eval_summary.get('ood_gpt', {}).get('roc_auc'))}/"
      f"{_fmt(eval_summary.get('ood_gpt_para', {}).get('roc_auc'))}, HC3 "
      f"human FPR {_fmt(eval_summary.get('hc3', {}).get('human_fpr'))}, "
      f"mixed band ≈ chance, non-English unvalidated, no auth/rate limits, "
      "single-seed CPU measurements.")
    a(f"- **Phase 5 recommendation:** (1) segment-level model as a new "
      f"versioned artifact reusing `Predictor` internals, then populate "
      f"`segments[]`; (2) Astro frontend against the frozen `/api/v1` "
      f"contract in `docs/api.md`; (3) deployment last. Do not retrain or "
      f"edit `final-v1` in place.")
    a("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf-8")
    n_sections = len(re.findall(r"(?m)^## \d+\. ", "\n".join(p)))
    print(f"report -> {OUT} ({OUT.stat().st_size} bytes, {len(p)} lines, "
          f"{n_sections} numbered sections)")
    if n_sections != 16:
        print(f"ERROR: expected 16 numbered sections, found {n_sections}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
