# Phase 6E — Explainability Feature-Scaling Audit

**Project:** WriteTrace · `D:\WriteTrace`
**Date:** 2026-10-10
**Scope:** investigation only. **No code changed, no model altered, nothing
retrained, nothing deployed, committed or tagged.** `ml/models/final-v1/` is
byte-identical.

---

## Conclusion

**PRESENTATION-ONLY ISSUE.** The numbers are correct, internally consistent, and
in the units the model was trained on. Nothing is mis-scaled.

The defect is that the human-readable **labels do not disclose the per-1,000
scaling convention**, so a reader interprets `101.9108` for `pron_i` as "about
102 first-person pronouns" when the text actually contains **16** (10.2 % of
words). The same applies to `syn_pos_adv_ratio` and `comma_rate`.

One secondary finding: for the zero-inflated `pron_i`, comparing against a
**median of 0** dramatically exaggerates the apparent deviation.

---

## 1. Evidence as reported

| Feature | Observed | Reported train median |
| ------- | -------- | --------------------- |
| `pron_i` | 101.9108 | 0 |
| `syn_pos_adv_ratio` | 98.9011 | 41.7096 |
| `comma_rate` | 14.27 | 6.6335 |
| `root_ttr` | 9.4174 | 7.79328 |

---

## 2. Units actually used

All four features are present in the `final-v1` feature list (89 features).

| Feature | Index | Formula | Source | Real unit |
| ------- | ----- | ------- | ------ | --------- |
| `pron_i` | 43 | `_rate(hits["pron_i"], n_words)` | `ml/features/extract.py:304` | first-person pronoun hits **per 1,000 words** |
| `syn_pos_adv_ratio` | 77 | `pos_counts["ADV"] * (1000.0 / n)` | `ml/features/syntactic.py:88` | adverbs **per 1,000 words** |
| `comma_rate` | 28 | `_rate(text.count(","), n_chars)` | `ml/features/extract.py:280` | commas **per 1,000 characters** |
| `root_ttr` | 19 | `unique_roots / math.sqrt(n_words)` | `ml/features/extract.py:261` | Guiraud-style root TTR — **not** a percentage, **not** a plain TTR |

The scale is centralised in one helper:

```python
def _rate(count: float, denom: float, scale: float = 1000.0) -> float:
    return float(count) / denom * scale if denom > 0 else float("nan")
```

The code comments state the convention explicitly (`# punctuation (per 1,000
chars)`, `# stylometric lexicons (per 1,000 words)`) — it is consistent
throughout, just not surfaced to the reader.

---

## 3. Observed and median use the same units — confirmed

`ml/explain.py` computes contributions by **substituting the training median
into the observed feature vector** and re-predicting:

```python
st = ref_stats.get(f)
X[j + 1, j] = float(st["median"])     # same vector, same units
contrib = p_base - float(p_all[j + 1])
```

If the units differed this would be arithmetically meaningless. The four
medians in `ml/models/final-v1/reference.json` match the reported values
**exactly**:

```
pron_i             median=0.0        mean=15.9556  sd=28.5972
comma_rate         median=6.633499   mean= 7.1410  sd= 4.9122
root_ttr           median=7.793275   mean= 8.5202  sd= 3.3958
syn_pos_adv_ratio  median=41.709637  mean=45.3540  sd=27.8400
```

**Training and inference share one code path** (`ml/features/extract.py`),
so no train/serve skew is possible here.

---

## 4. Arithmetic reproduction

Back-solving the deployed values for the stated 157-word input:

| Feature | Raw value | Implied reality |
| ------- | --------- | --------------- |
| `pron_i` | 101.9108 | **exactly 16.00** first-person pronouns = 10.19 % of words |
| `syn_pos_adv_ratio` | 98.9011 | ≈ 15.5 adverbs = 9.89 % of words |
| `comma_rate` | 14.27 | ≈ 13.4 commas = 1.43 % of characters |
| `root_ttr` | 9.4174 | 118 unique roots |

The `pron_i` back-solve lands on a **clean integer (16.00)**, which is strong
confirmation of the per-1,000-words convention rather than a coincidental fit.

**Empirical check** on a synthetic first-person text (139 words, 30 first-person
hits):

```
pron_i = 215.8273      and   30 / 139 x 1000 = 215.8273   <- exact match
```

---

## 5. Is anything mislabelled?

**No unit conversion or unit label is applied anywhere.** Searched
`ml/explain.py`, `backend/app.py`, `backend/schemas.py`,
`frontend/src/lib/format.ts`, `frontend/src/lib/render-result.ts` — none
contains a per-1000 note, a ×100 conversion, or a percent suffix. Values are
passed through raw and formatted as plain decimals.

**But the labels omit the unit**, which is the actual problem:

```python
phrase("pron_i")            -> "first-person singular pronouns"
phrase("syn_pos_adv_ratio") -> "adverb density (spaCy)"
phrase("comma_rate")        -> "comma usage"
phrase("root_ttr")          -> "root type-token ratio"
```

Rendered as *"observed value 101.9108 vs train median 0"* beside the phrase
*"first-person singular pronouns"*, a reader will conclude the text contains
~102 first-person pronouns. It contains 16.

`root_ttr` has a second, subtler label problem: the phrase says "type-token
ratio", but the implementation is `unique_roots / sqrt(n_words)` (Guiraud's
root TTR), which is **not** bounded to 1.0 — so `9.4174` cannot be read as
"9.4 %" or as a plain TTR of 0.094.

---

## 6. Deployed vs local response format

| Aspect | Local | Deployed report | Match |
| ------ | ----- | --------------- | ----- |
| Signal fields | `feature, phrase, family, value, reference_median, contribution, direction, interpretation, caveat` | same | **yes** |
| `value` / `reference_median` | raw floats | raw floats | **yes** |
| Environment | Python 3.13.15, fastapi 0.141.1, spacy 3.8.16, sklearn 1.9.0, numpy 2.5.3 | pinned identically in `requirements.txt` | consistent |

**Limitation:** the exact 157-word input was not available, so byte-identical
value reproduction was not possible. The comparison establishes format identity
and unit convention, not value equality.

---

## 7. Statistical context (why `pron_i` looked alarming)

| Feature | Observed | Mean | SD | z |
| ------- | -------- | ---- | -- | - |
| `pron_i` | 101.9108 | 15.9556 | 28.5972 | **+3.01** |
| `comma_rate` | 14.27 | 7.1410 | 4.9122 | +1.45 |
| `root_ttr` | 9.4174 | 8.5202 | 3.3958 | +0.26 |
| `syn_pos_adv_ratio` | 98.9011 | 45.3540 | 27.8400 | +1.92 |

`pron_i` is genuinely first-person-heavy (+3.01 SD) — a real property of the
text, not a defect. But it is **zero-inflated**: over half of the 155,988-row
training corpus contains no first-person pronoun at all, which is why the median
is `0`. Against a median of 0 the deviation looks catastrophic; against the mean
and SD it is +3.01 — high, but ordinary for first-person prose.

**`root_ttr` and `comma_rate` are unremarkable** (+0.26 and +1.45) and were never
evidence of a problem.

---

## 8. Ruled out

- **Not a train/serve skew** — one extraction function serves both.
- **Not a missing or renamed feature** — all four are in the 89-feature list and
  are consumed by the model.
- **Not an imputation/clipping artefact** — the earlier scikit-learn warning
  `Skipping features without any observed values: [71]` refers to feature index
  71 = **`sent_len_trend`**, unrelated to these four.
- **Not a `torch`/`transformers` lazy-import effect** — those are lazily
  imported inside Phase 2 training functions and never reached by `final-v1`.
- **Not a unit mismatch between observed and median** — proven by the
  median-substitution mechanism and the exact median matches.

---

## 9. Recommended follow-ups (NOT implemented — investigation only)

1. **Disclose the unit in the phrase or the rendered line.** e.g. append
   "per 1,000 words" for the `_rate`-based families and "per 1,000 characters"
   for punctuation. This is the single highest-value fix.
2. **Use mean ± SD alongside the median** in the explanation, or use the mean as
   the perturbation baseline for zero-inflated features.
3. **Rename `root_ttr`'s phrase** to reflect the Guiraud form
   ("root type-token ratio (Guiraud)") so `9.42` is not read as a percentage.
4. Consider exposing the family-level unit once per signal group rather than
   per feature.

Any change must be reviewed against the abstention and abstention-adjacent
logic constraints; none of the above alters inference, calibration, thresholds
or the API contract.

---

*Investigation only. No code change, no model change, no retraining, no
deployment, commit or tag. `ml/models/final-v1/` verified unchanged.*