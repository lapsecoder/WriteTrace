"""Phase-4 inference engine tests.

Covers: artifact loading, determinism, probability semantics, the Phase-3
evidence gate, presentation policy, explainability schema, reference parity
against the locked Phase-3 numbers, and load-once behavior.
"""
from __future__ import annotations

import json
import math

import pytest

from ml.experiments.phase3 import FINAL_DIR

if not (FINAL_DIR / "manifest.json").exists():
    pytest.skip("final artifact not written yet", allow_module_level=True)

from ml.inference import (  # noqa: E402
    DISCLAIMER, Predictor, get_predictor, presentation_label, reset_predictor,
)

NORMAL_TEXT = (
    "The committee announced yesterday that the new library will open in "
    "March. It cost roughly 4.2 million euros, officials said, and will "
    "stay open seven days a week. \"We are thrilled,\" the mayor told "
    "reporters. I have to admit the delays frustrated quite a few "
    "residents, but the finished building looks wonderful. Bring your "
    "library cards! The project began in 2019 after a survey found that "
    "residents wanted longer opening hours, better accessibility, and a "
    "dedicated children's area."
)

SHORT_TEXT = "Too short to judge."

UNICODE_TEXT = (
    "Le comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars. "
    "Elle a coûté environ 4,2 millions d'euros, ont dit les responsables. "
    "這個項目是本地運行的，不會把文字發送到任何外部服務。 "
    "I have to admit the delays frustrated quite a few residents, but the "
    "finished building looks wonderful. Bring your library cards! 🎉 "
    "Nous sommes ravis, a déclaré la maire, et elle restera ouverte."
)

MULTILINE_TEXT = (
    "First paragraph of the report describes the renovation timeline.\n\n"
    "Second paragraph lists the contractors and the final budget.\n\n"
    "Third paragraph thanks the residents for their patience during "
    "two years of construction work and noise."
)

PUNCT_TEXT = (
    "Wait—really?! Yes; truly... (the \"final\" count) wasn't 42; it was "
    "43!!! #refresh 100% -- see below... Okay? Right. "
    * 6
)


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(FINAL_DIR)


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------

def test_all_artifact_components_load(predictor):
    f = predictor.final
    assert hasattr(f["model"], "predict_proba")
    assert hasattr(f["calibrator"], "predict")
    assert f["state"] is not None
    assert isinstance(f["features"], list) and len(f["features"]) == 89
    assert f["manifest"]["winner"] == "hgb_syn"
    assert f["manifest"]["calibrator"] == "platt"
    assert f["manifest"]["threshold"] == 0.5
    assert f["manifest"].get("artifact_complete") is True
    assert f["config"]["label_policy"].startswith("canonical y=1 MACHINE")
    assert (FINAL_DIR / "reference.json").exists()
    assert (FINAL_DIR / "importance.json").exists()
    assert predictor.needs_syn is True
    assert predictor.model_version == "final-v1"


def test_gate_values_match_phase3_report(predictor):
    gate = json.loads((FINAL_DIR / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "active"
    assert gate["min_words"] == 50 == predictor.min_words
    assert gate["mixed_band"] == [0.4, 0.6]
    assert predictor.mixed_band == (0.4, 0.6)
    assert gate["threshold"] == 0.5
    # values published in docs/experiments/phase3-report.md
    assert gate["abstain_frac_test"] == pytest.approx(0.0933, abs=1e-4)
    assert gate["mixed_band_accuracy"] == pytest.approx(0.5342, abs=1e-4)


# --------------------------------------------------------------------------
# determinism & probability semantics
# --------------------------------------------------------------------------

def test_deterministic_inference_full_precision(predictor):
    p1 = predictor.score(NORMAL_TEXT)
    p2 = predictor.score(NORMAL_TEXT)
    assert p1 == p2                      # exact, not approximately
    r1 = predictor.analyze(NORMAL_TEXT, include_signals=False)
    r2 = predictor.analyze(NORMAL_TEXT, include_signals=False)
    assert r1.machine_probability == r2.machine_probability
    assert r1.label == r2.label and r1.decision == r2.decision


def test_probability_bounds_and_human_complement(predictor):
    for text in (NORMAL_TEXT, UNICODE_TEXT, MULTILINE_TEXT, PUNCT_TEXT,
                 SHORT_TEXT):
        r = predictor.analyze(text, include_signals=False)
        p = r.machine_probability
        assert math.isfinite(p) and 0.0 <= p <= 1.0
        assert math.isfinite(r.human_probability)
        assert r.human_probability == pytest.approx(1.0 - p, abs=1e-15)
        assert r.machine_probability + r.human_probability == \
            pytest.approx(1.0, abs=1e-12)
        if r.confidence is not None:
            assert 0.5 <= r.confidence <= 1.0


def test_orientation_matches_locked_evaluation(predictor, reference_result):
    """Canonical orientation (y=1 MACHINE, p = cal(model(x)), no complement).

    Proven two ways, both from executed numbers: (a) the locked Phase-3
    evaluation summary stored in the artifact (ROC-AUC would collapse to
    ~0.04 instead of ~0.96 if the sign were flipped), (b) mean p(machine)
    over all 51,995 MAGE-test rows, machine rows vs human rows.
    """
    ev = predictor.manifest["eval_summary"]["test"]
    assert ev["roc_auc"] > 0.9
    o = reference_result["orientation"]
    assert o["separated"], o
    assert o["n_human"] + o["n_machine"] == ev["n"]
    assert o["mean_p_machine"] > 0.8
    assert o["mean_p_human"] < 0.4


# --------------------------------------------------------------------------
# evidence gate / abstention
# --------------------------------------------------------------------------

def test_short_text_abstains_with_full_metadata(predictor):
    r = predictor.analyze(SHORT_TEXT, include_signals=False)
    assert r.evidence_sufficient is False
    assert r.decision == "abstain"
    assert r.status == "insufficient_evidence"
    assert r.label == "insufficient_evidence"
    assert r.confidence is None
    assert r.word_count == 4                 # "Too short to judge."
    assert r.gate_min_words == 50
    assert any("insufficient evidence" in w for w in r.warnings)
    assert 0.0 <= r.machine_probability <= 1.0      # reported, flagged
    assert r.disclaimer == DISCLAIMER


def test_empty_and_invalid_text_raise(predictor):
    with pytest.raises(ValueError):
        predictor.analyze("")
    with pytest.raises(ValueError):
        predictor.analyze("    \n\t  ")
    with pytest.raises(ValueError):
        predictor.analyze(None)          # type: ignore[arg-type]
    with pytest.raises(ValueError):
        predictor.score("")


@pytest.mark.parametrize("injected,expected", [(0.45, "human"),
                                               (0.55, "machine")])
def test_mixed_band_flags_low_confidence(predictor, monkeypatch, injected,
                                         expected):
    monkeypatch.setattr(Predictor, "_score_row",
                        lambda self, row, v=injected: v)
    r = predictor.analyze(NORMAL_TEXT, include_signals=False)
    assert r.mixed_low_confidence is True
    assert r.label == "mixed"
    assert r.decision == expected           # binary decision still at 0.5
    assert any("mixed" in w for w in r.warnings)


def test_presentation_label_policy():
    band = (0.4, 0.6)
    assert presentation_label(0.05, True, band) == "likely_human"
    assert presentation_label(0.30, True, band) == "mostly_human"
    assert presentation_label(0.50, True, band) == "mixed"
    assert presentation_label(0.70, True, band) == "mostly_machine"
    assert presentation_label(0.95, True, band) == "likely_machine"
    assert presentation_label(0.95, False, band) == "insufficient_evidence"
    # boundary values
    assert presentation_label(0.4, True, band) == "mixed"
    assert presentation_label(0.6, True, band) == "mixed"
    assert presentation_label(0.2, True, band) == "mostly_human"
    assert presentation_label(0.8, True, band) == "mostly_machine"


def test_presentation_policy_is_separate_from_decision(predictor, monkeypatch):
    """Label vocabulary must never change the threshold-0.5 decision."""
    for injected, expected in ((0.55, "machine"), (0.45, "human")):
        monkeypatch.setattr(Predictor, "_score_row",
                            lambda self, row, v=injected: v)
        r = predictor.analyze(NORMAL_TEXT, include_signals=False)
        assert r.decision == expected
        assert r.threshold == 0.5


# --------------------------------------------------------------------------
# explainability
# --------------------------------------------------------------------------

def test_signals_schema_and_no_causal_language(predictor):
    r = predictor.analyze(NORMAL_TEXT, include_signals=True, max_signals=6)
    assert 1 <= len(r.signals) <= 6
    contribs = [s.contribution for s in r.signals]
    assert contribs == sorted(contribs, key=lambda v: -abs(v))
    for s in r.signals:
        assert s.feature and s.phrase
        assert s.family in {"structure", "lexical", "punctuation",
                            "stylometric", "repetition", "syntactic",
                            "readability", "positional", "syntactic_spacy"}
        assert s.direction in {"machine", "human", "neutral"}
        assert math.isfinite(s.contribution)
        assert math.isfinite(s.reference_median)
        assert "not proof of authorship" in s.caveat
        low = (s.interpretation + s.phrase).lower()
        assert "proves" not in low and "proof that" not in low


def test_signals_can_be_disabled(predictor):
    r = predictor.analyze(NORMAL_TEXT, include_signals=False)
    assert r.signals == []
    r2 = predictor.analyze(NORMAL_TEXT, include_signals=True, max_signals=3)
    assert len(r2.signals) <= 3


def test_legacy_explain_contract(predictor):
    ex = predictor.explain(NORMAL_TEXT, k=5)
    assert set(ex) == {"prediction", "contributions", "text_rendered",
                       "method"}
    assert len(ex["contributions"]) <= 5
    assert "not causal attribution" in ex["method"]
    assert ex["prediction"]["p_machine"] == \
        round(predictor.score(NORMAL_TEXT), 4)
    assert "method:" in ex["text_rendered"]


def test_local_contributions_batched_equals_reference(predictor):
    """The batched perturbation must equal the sequential definition."""
    from ml.explain import local_contributions, load_reference

    nt_row = predictor._extract_row(NORMAL_TEXT)
    x = nt_row[predictor.features].to_numpy()[0]
    ref = load_reference(FINAL_DIR)["train_reference_stats"]
    model, cal = predictor.final["model"], predictor.final["calibrator"]

    p_base = float(cal.predict(model.predict_proba(x.reshape(1, -1))[:, 1])[0])
    sequential: dict[str, float] = {}
    for j, f in enumerate(predictor.features):
        if f not in ref:
            continue
        x2 = x.copy()
        x2[j] = float(ref[f]["median"])
        p2 = float(cal.predict(model.predict_proba(x2.reshape(1, -1))[:, 1])[0])
        sequential[f] = round(p_base - p2, 4)
    batched = {c["feature"]: c["contribution_to_p_machine"]
               for c in local_contributions(model, cal, predictor.features,
                                            x, ref)}
    assert sequential.keys() == batched.keys()
    for f, v in sequential.items():
        assert v == batched[f], f"{f}: sequential {v} vs batched {batched[f]}"


# --------------------------------------------------------------------------
# reference parity against the locked Phase-3 numbers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_result():
    from ml.experiments.phase4 import reference_check
    return reference_check(n_per_class=2, seed=0)


def test_reference_parity_vs_phase3_cached_path(reference_result):
    res = reference_result
    assert res["n_rows"] == 4
    assert res["all_within_tolerance"], res["max_abs_delta"]
    assert res["max_abs_delta"] <= 1e-9
    ec = res["e7_cross_check"]
    assert ec and ec["match"], ec


def test_report_worked_example_reproduced(reference_result):
    """docs/experiments/phase3-report.md Section 14 example must reproduce:
    the highest-confidence MAGE-test human false positive (78 words,
    p(machine) = 1.000 at 3 dp) via the Phase-4 inference path."""
    ec = reference_result["e7_cross_check"]
    assert ec["recomputed_words"] == 78
    assert ec["e7_recorded_words"] == 78
    assert ec["phase4_p_machine"] == ec["e7_recorded_p_machine"]
    assert round(ec["phase4_p_machine"], 3) == 1.000
    assert ec["recomputed_domain"] == "eli5"


# --------------------------------------------------------------------------
# load-once behavior
# --------------------------------------------------------------------------

def test_get_predictor_is_singleton_and_loads_once(monkeypatch):
    import ml.inference as inf
    calls: list = []
    original = inf.load_final

    def counting(final_dir=None):
        calls.append(str(final_dir))
        return original(FINAL_DIR if final_dir is None else final_dir)

    monkeypatch.setattr(inf, "load_final", counting)
    reset_predictor()
    try:
        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2
        assert len(calls) == 1, f"artifact loaded {len(calls)} times"
        # further analyze() calls must not reload anything
        p1.analyze(NORMAL_TEXT, include_signals=False)
        p1.explain(SHORT_TEXT, k=3)
        assert len(calls) == 1
    finally:
        reset_predictor()


def test_spacy_pipeline_loaded_once():
    from ml.features import syntactic
    nlp1 = syntactic._load_nlp()
    nlp2 = syntactic._load_nlp()
    assert nlp1 is nlp2
    assert len(syntactic._NLP_CACHE) >= 1


# --------------------------------------------------------------------------
# clean-process loading (Phase-4 requirement)
# --------------------------------------------------------------------------

def test_clean_process_loading_reproduces_probability(predictor, tmp_path):
    """A brand-new OS process must load final-v1 from scratch and return
    bit-identical probabilities - proves no hidden in-process state leaks
    into the inference path."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "clean_load.py"
    script.write_text(
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from ml.experiments.phase3 import FINAL_DIR\n"
        "from ml.inference import Predictor\n"
        "p = Predictor(FINAL_DIR)\n"
        "r = p.analyze(sys.argv[2], include_signals=False)\n"
        "print(json.dumps({'p': r.machine_probability,\n"
        "                  'model_version': r.model_version,\n"
        "                  'model_name': r.model_name,\n"
        "                  'decision': r.decision}))\n",
        encoding="utf-8")

    expected = predictor.analyze(NORMAL_TEXT, include_signals=False)
    proc = subprocess.run(
        [sys.executable, str(script), str(root), NORMAL_TEXT],
        capture_output=True, text=True, timeout=300, cwd=str(root))
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["model_version"] == "final-v1"
    assert got["model_name"] == expected.model_name
    assert got["decision"] == expected.decision
    assert got["p"] == pytest.approx(expected.machine_probability,
                                     abs=1e-12)
