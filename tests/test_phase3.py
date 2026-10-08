"""Phase-3 unit tests: config validity, orientation (no complement),
selection ranking, gate/mixed-band rules, ablation feature removal,
syntactic determinism, and (when present) final-artifact integrity.
No training and no model downloads required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.experiments.phase3 import (
    EXPERIMENTS, FINAL_DIR, HIGHER_BETTER, ROOT, V1_FEATURES,
    _ablation_features, decide_gate, lexicographic_rank, load_final,
    mixed_band, p_of, selection_pool,
)

CFG = json.loads((ROOT / "ml" / "configs" / "phase3.json").read_text(
    encoding="utf-8"))


# --------------------------------------------------------------------------
# config validity (pre-registration guards)
# --------------------------------------------------------------------------

def test_config_criteria_and_bands_align():
    order = CFG["selection"]["criteria_order"]
    assert order == [
        "robustness_generalization", "calibration", "false_positive_behavior",
        "pr_auc", "interpretability", "computational_practicality",
        "in_distribution_performance",
    ]
    assert set(HIGHER_BETTER) == set(order)
    assert set(CFG["selection"]["tie_bands"]) == set(order)
    # robustness must be first: selection must never optimize MAGE accuracy
    assert order[0] == "robustness_generalization"
    assert order[-1] == "in_distribution_performance"
    assert "accuracy" not in order


def test_config_ablation_families_and_names_exist():
    from ml.features.extract import FEATURE_FAMILIES, feature_names
    known = set(feature_names())
    for ab in CFG["ablations"]:
        if "remove_family" in ab:
            assert ab["remove_family"] in FEATURE_FAMILIES
        for f in ab.get("remove", []):
            assert f in known, f"ablation {ab['key']} removes unknown {f}"


def test_config_pool_names_sensible():
    pool = CFG["selection_pool"]
    for name in pool["primary"]:
        assert name in CFG["candidates"]
    for name in pool["challengers"]:
        assert name.startswith("abl_") or name == "hgb_syn"


def test_experiment_order_ends_with_gate():
    keys = list(EXPERIMENTS)
    assert keys[0] == "E3"
    assert keys[-1] == "E8"
    assert "SELECTION" in keys
    assert keys.index("SELECTION") < keys.index("E7") < keys.index("E8")


# --------------------------------------------------------------------------
# orientation: Phase-3 candidates never complement at inference
# --------------------------------------------------------------------------

class _StubModel:
    def __init__(self, raw):
        self.raw = float(raw)

    def predict_proba(self, X):
        n = len(np.asarray(X))
        return np.column_stack([np.full(n, 1 - self.raw),
                                np.full(n, self.raw)])


class _StubCal:
    def predict(self, p):
        return np.asarray(p, dtype=float) * 0.8 + 0.1


def test_p_of_applies_calibrator_without_complement():
    cand = {"model": _StubModel(0.2), "calibrators": {"isotonic": _StubCal()},
            "default_cal": "isotonic", "features": ["f1", "f2"]}
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    p = p_of(cand, df)
    # raw 0.2 -> cal: 0.2*0.8+0.1 = 0.26 (NOT 1-0.2=0.8, NOT complemented)
    assert np.allclose(p, [0.26])


def test_p_of_respects_default_cal_and_explicit_override():
    class _Other:
        def predict(self, p):
            return np.asarray(p, dtype=float) * 0.5
    cand = {"model": _StubModel(0.4),
            "calibrators": {"isotonic": _StubCal(), "platt": _Other()},
            "default_cal": "platt", "features": ["f1"]}
    df = pd.DataFrame({"f1": [0.0]})
    assert np.allclose(p_of(cand, df), [0.2])
    assert np.allclose(p_of(cand, df, cal="isotonic"), [0.42])


# --------------------------------------------------------------------------
# selection ranking
# --------------------------------------------------------------------------

def test_lexicographic_rank_robustness_beats_in_distribution():
    crit = {
        "weak_id_strong_ood": {"robustness_generalization": 0.90,
                               "calibration": 0.10,
                               "false_positive_behavior": 0.20,
                               "pr_auc": 0.95, "interpretability": 0,
                               "computational_practicality": 10.0,
                               "in_distribution_performance": 0.90},
        "strong_id_weak_ood": {"robustness_generalization": 0.80,
                               "calibration": 0.10,
                               "false_positive_behavior": 0.20,
                               "pr_auc": 0.99, "interpretability": 0,
                               "computational_practicality": 10.0,
                               "in_distribution_performance": 0.99},
    }
    order = CFG["selection"]["criteria_order"]
    bands = CFG["selection"]["tie_bands"]
    ranked, _ = lexicographic_rank(crit, order, bands, HIGHER_BETTER)
    assert ranked[0] == "weak_id_strong_ood"   # robustness first, always


def test_lexicographic_rank_tie_band_falls_through():
    crit = {
        "a": {"robustness_generalization": 0.900, "calibration": 0.10,
              "false_positive_behavior": 0.20, "pr_auc": 0.95,
              "interpretability": 0, "computational_practicality": 5.0,
              "in_distribution_performance": 0.90},
        "b": {"robustness_generalization": 0.898, "calibration": 0.05,
              "false_positive_behavior": 0.20, "pr_auc": 0.95,
              "interpretability": 0, "computational_practicality": 9.0,
              "in_distribution_performance": 0.99},
    }
    order = CFG["selection"]["criteria_order"]
    bands = CFG["selection"]["tie_bands"]
    ranked, _ = lexicographic_rank(crit, order, bands, HIGHER_BETTER)
    # 0.002 diff <= 0.005 band -> tie; b wins on calibration (lower)
    assert ranked[0] == "b"


def test_lexicographic_rank_lr_wins_interpretability_tie():
    base = {"robustness_generalization": 0.90, "calibration": 0.10,
            "false_positive_behavior": 0.20, "pr_auc": 0.95,
            "computational_practicality": 5.0,
            "in_distribution_performance": 0.90}
    crit = {"lr": {**base, "interpretability": 1},
            "hgb": {**base, "interpretability": 0}}
    order = CFG["selection"]["criteria_order"]
    bands = CFG["selection"]["tie_bands"]
    ranked, _ = lexicographic_rank(crit, order, bands, HIGHER_BETTER)
    assert ranked[0] == "lr"


# --------------------------------------------------------------------------
# gate + mixed band rules
# --------------------------------------------------------------------------

def test_decide_gate_picks_smallest_eligible():
    table = [
        {"min_words": 25, "frac_abstained": 0.03, "kept_f1_macro": 0.88,
         "excluded_f1_macro": 0.60},
        {"min_words": 50, "frac_abstained": 0.09, "kept_f1_macro": 0.89,
         "excluded_f1_macro": 0.79},
        {"min_words": 75, "frac_abstained": 0.23, "kept_f1_macro": 0.90,
         "excluded_f1_macro": 0.80},     # over abstain cap
    ]
    out = decide_gate(table, abstain_frac_max=0.15)
    assert out["min_words"] == 25
    assert out["rule_applied"] is True
    assert 75 not in out["eligible"]


def test_decide_gate_requires_excluded_worse_than_kept():
    table = [
        {"min_words": 50, "frac_abstained": 0.09, "kept_f1_macro": 0.85,
         "excluded_f1_macro": 0.90},   # excluded BETTER -> not eligible
    ]
    out = decide_gate(table, abstain_frac_max=0.15)
    assert out["min_words"] is None
    assert out["rule_applied"] is False


def test_decide_gate_skips_notes_and_small_excluded():
    table = [
        {"min_words": 25, "note": "excluded bucket too small",
         "excluded_f1_macro": None, "kept_f1_macro": 0.9,
         "frac_abstained": 0.01},
        {"min_words": 50, "frac_abstained": 0.09, "kept_f1_macro": 0.9,
         "excluded_f1_macro": 0.7},
    ]
    out = decide_gate(table, abstain_frac_max=0.15)
    assert out["min_words"] == 50


def test_mixed_band_straddles_threshold():
    rng = np.random.default_rng(0)
    near = rng.uniform(0.42, 0.58, size=400)
    y_near = (near >= 0.5).astype(int)
    flip = rng.integers(0, 2, size=len(near))   # ~50% accuracy near 0.5
    y_near = np.where(np.abs(near - 0.5) < 0.08, flip, y_near)
    far_hi = np.full(200, 0.97)
    far_lo = np.full(200, 0.03)
    p = np.concatenate([far_lo, near, far_hi])
    y = np.concatenate([np.zeros(200, int), y_near, np.ones(200, int)])
    out = mixed_band(p, y, [i / 10 for i in range(11)], acc_threshold=0.75)
    assert out["band"] is not None
    lo, hi = out["band"]
    assert lo < 0.5 < hi
    assert out["accuracy_inside_band"] < 0.75


def test_mixed_band_none_when_well_calibrated_everywhere():
    p = np.concatenate([np.full(300, 0.95), np.full(300, 0.05)])
    y = np.concatenate([np.ones(300, int), np.zeros(300, int)])
    out = mixed_band(p, y, [i / 10 for i in range(11)], acc_threshold=0.75)
    # every straddling interval includes some error-free mass but the interval
    # [0.4,0.6] is empty -> widest containing bands are the full range with
    # accuracy 1.0 -> no band qualifies
    assert out["band"] is None


# --------------------------------------------------------------------------
# ablations
# --------------------------------------------------------------------------

def test_ablation_minus_lexical_removes_exactly_family():
    from ml.features.extract import FEATURE_FAMILIES
    ab = {"key": "minus_lexical", "remove_family": "lexical"}
    feats, removed = _ablation_features(CFG, ab)
    assert set(removed) == set(FEATURE_FAMILIES["lexical"])
    assert set(feats) == set(V1_FEATURES) - set(FEATURE_FAMILIES["lexical"])
    assert len(feats) == len(V1_FEATURES) - len(FEATURE_FAMILIES["lexical"])


def test_ablation_preregistered_leakage_set_applies():
    ab = [a for a in CFG["ablations"]
          if a["key"] == "minus_leakage_preregistered"][0]
    feats, removed = _ablation_features(CFG, ab)
    assert len(removed) == 16
    assert "rare_token_ratio" in removed and "mean_log_df" in removed
    assert "boilerplate_rate" in removed and "hash_rate" in removed
    assert all(f in V1_FEATURES for f in removed)
    assert "avg_sent_len" in feats    # structure family untouched


def test_ablation_unknown_feature_raises():
    with pytest.raises(ValueError):
        _ablation_features(CFG, {"key": "bad", "remove": ["not_a_feature"]})


# --------------------------------------------------------------------------
# syntactic family determinism (spacy, small)
# --------------------------------------------------------------------------

def test_syntactic_extraction_deterministic_and_short_nan():
    from ml.features.syntactic import SYNTACTIC_FEATURES, extract_syntactic
    texts = ["The cat sat on the mat because it was warm outside.",
             "", "hi"]
    a = extract_syntactic(texts)
    b = extract_syntactic(texts)
    assert list(a.columns) == list(SYNTACTIC_FEATURES)
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[1].isna().all() and a.iloc[2].isna().all()   # <2 words
    assert np.isfinite(a.iloc[0]).all()


# --------------------------------------------------------------------------
# results / artifact integrity (conditional on pipeline progress)
# --------------------------------------------------------------------------

def _results() -> dict:
    p = ROOT / "ml" / "experiments" / "phase3" / "phase3-results.json"
    if not p.exists():
        pytest.skip("phase3 results not written yet")
    return json.loads(p.read_text(encoding="utf-8"))


def test_results_no_failed_experiments():
    res = _results()
    failed = {k: v.get("error") for k, v in res.items()
              if isinstance(v, dict) and v.get("status") == "failed"}
    assert not failed, failed


def test_selection_winner_in_pool_when_present():
    res = _results()
    sel = res.get("SELECTION")
    if not isinstance(sel, dict) or not sel.get("winner"):
        pytest.skip("SELECTION not run yet")
    assert sel["winner"] in sel["pool"]
    order = CFG["selection"]["criteria_order"]
    assert [c["criterion"] for c in sel["trace"]] == order
    # winner must be robustness-best among pool within recorded values
    crit = sel["criteria_values"]
    vals = {n: c["robustness_generalization"] for n, c in crit.items()
            if c.get("robustness_generalization") is not None}
    assert vals[sel["winner"]] == max(vals.values())


def test_final_artifact_integrity():
    if not (FINAL_DIR / "manifest.json").exists():
        pytest.skip("final artifact not written yet")
    from ml.experiments.phase3 import file_sha
    m = json.loads((FINAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    for fn, meta in m["files"].items():
        p = FINAL_DIR / fn
        assert p.exists(), f"missing {fn}"
        assert file_sha(p) == meta["sha256"], f"hash mismatch {fn}"
    feats = json.loads((FINAL_DIR / "features.json").read_text(encoding="utf-8"))
    import joblib
    model = joblib.load(FINAL_DIR / "model.joblib")
    n_in = getattr(model, "n_features_in_", None)
    if n_in is None:
        n_in = model[-1].n_features_in_ if hasattr(model, "__getitem__") else None
    assert n_in == feats["n_features"] == len(feats["features"])
    cfg_doc = json.loads((FINAL_DIR / "config.json").read_text(encoding="utf-8"))
    assert cfg_doc["threshold"] == 0.5
    assert "canonical" in cfg_doc["label_policy"]


def test_final_artifact_gate_rules_honored():
    gate_p = FINAL_DIR / "gate.json"
    if not gate_p.exists():
        pytest.skip("gate not written yet")
    g = json.loads(gate_p.read_text(encoding="utf-8"))
    if g.get("status") != "active":
        pytest.skip("gate still pending")
    if g.get("min_words") is not None:
        assert g["min_words"] in CFG["e8"]["gate_candidates"]
        assert g["abstain_frac_test"] <= CFG["e8"]["abstain_frac_max"]
    if g.get("mixed_band") is not None:
        lo, hi = g["mixed_band"]
        assert lo < 0.5 < hi


def test_predictor_deterministic_and_gated():
    if not (FINAL_DIR / "manifest.json").exists():
        pytest.skip("final artifact not written yet")
    gate_p = FINAL_DIR / "gate.json"
    if gate_p.exists() and json.loads(gate_p.read_text(encoding="utf-8")).get(
            "status") != "active":
        pytest.skip("gate still pending")
    from ml.inference import Predictor
    pred = Predictor(FINAL_DIR)
    text = ("The committee announced yesterday that the new library will "
            "open in March. It cost roughly 4.2 million euros, officials "
            "said, and will stay open seven days a week. \"We are thrilled,"
            "\" the mayor told reporters. I have to admit the delays "
            "frustrated quite a few residents, but the finished building "
            "looks wonderful. Bring your library cards!")
    r1, r2 = pred.predict(text), pred.predict(text)
    assert r1["p_machine"] == r2["p_machine"]
    assert 0.0 <= r1["p_machine"] <= 1.0
    assert r1["label"] in ("machine", "human")
    assert r1["decision"] in ("machine", "human", "abstain")
    short = pred.predict("Too short to judge.")
    if pred.gate.get("min_words"):
        assert short["abstain"] is True
        assert short["decision"] == "abstain"
    ex = pred.explain(text, k=5)
    assert ex["contributions"] and len(ex["contributions"]) <= 5
    cs = [c["contribution_to_p_machine"] for c in ex["contributions"]]
    assert cs == sorted(cs, key=lambda v: -abs(v))
    assert all("phrase" in c for c in ex["contributions"])


def test_manifest_data_policy_declares_eval_only_corpora():
    if not (FINAL_DIR / "manifest.json").exists():
        pytest.skip("final artifact not written yet")
    m = json.loads((FINAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    pol = m["data_policy"]
    assert pol["trained_on"] == "MAGE-train only"
    assert "HC3" in pol["eval_only"] and "RAID" in pol["eval_only"]
    assert "0.5" in pol["threshold"]
