/**
 * Isomorphic result renderer.
 *
 * Produces the same editorial markup as `ResultsPanel.astro`, but as a DOM
 * node, so the controller can render a live response without a UI framework
 * or a second templating language. Escaping is centralised in `escapeHtml`;
 * values come only from the API response.
 */
import type { AnalyzeResponse, Signal } from "./types";
import {
  DECISION_COPY,
  formatBand,
  formatCount,
  formatFeatureValue,
  formatInterval,
  formatProbability,
  formatSigned,
  labelCopy,
} from "./format";

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const pct = (v: number) =>
  Number.isFinite(v) ? `${(Math.max(0, Math.min(1, v)) * 100).toFixed(1)}%` : "n/a";

function signalRow(s: Signal): string {
  const dirClass =
    s.direction === "machine"
      ? "text-seal"
      : s.direction === "human"
        ? "text-verdigris"
        : "text-graphite";
  const dirLabel =
    s.direction === "machine"
      ? "toward machine"
      : s.direction === "human"
        ? "toward human"
        : "no measurable shift";
  return `
    <li class="py-4" data-signal="${escapeHtml(s.signal)}" data-direction="${escapeHtml(s.direction)}">
      <div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p class="font-display text-[1.0625rem] text-ink">${escapeHtml(s.label)}</p>
        <p class="micro ${dirClass}">${dirLabel} · ${escapeHtml(formatSigned(s.contribution))}</p>
      </div>
      <p class="mt-1 font-mono text-[0.75rem] text-graphite">
        ${escapeHtml(s.signal)} · ${escapeHtml(s.family)} · observed ${escapeHtml(formatFeatureValue(s.value))} · train median ${escapeHtml(formatFeatureValue(s.reference_median))}
      </p>
      <p class="mt-2 text-[0.9375rem] leading-relaxed text-ink-soft">${escapeHtml(s.interpretation)}</p>
      <p class="mt-1.5 text-[0.8125rem] italic leading-relaxed text-graphite">${escapeHtml(s.caveat)}</p>
    </li>`;
}

function signalsBlock(signals: Signal[]): string {
  if (signals.length === 0) {
    return `<p class="text-[0.9375rem] text-graphite" data-testid="signals-empty">The backend returned no signals for this analysis.</p>`;
  }
  return `<ol class="divide-y divide-rule/70 border-y border-rule/70" data-testid="signals">${signals
    .map(signalRow)
    .join("")}</ol>`;
}

function probabilityBlock(r: AnalyzeResponse, tone: string): string {
  const humanWidth = Number.isFinite(r.human_probability)
    ? (Math.max(0, Math.min(1, r.human_probability)) * 100).toFixed(2)
    : "0.00";
  const machineWidth = Number.isFinite(r.machine_probability)
    ? (Math.max(0, Math.min(1, r.machine_probability)) * 100).toFixed(2)
    : "0.00";
  const toneFill =
    tone === "machine"
      ? "var(--color-seal)"
      : tone === "human"
        ? "var(--color-verdigris)"
        : tone === "mixed"
          ? "var(--color-seal-soft)"
          : "var(--color-graphite-light)";
  return `
    <div class="w-full" data-testid="probability-bar">
      <div class="relative flex h-3 w-full overflow-hidden border border-rule-strong bg-paper" role="img"
           aria-label="Human-written ${pct(r.human_probability)}, machine-generated ${pct(r.machine_probability)}">
        <div class="h-full" style="width:${humanWidth}%; background-color: var(--color-verdigris); opacity:0.55;" data-segment="human"></div>
        <div class="animate-ink h-full" style="width:${machineWidth}%; background-color:${toneFill};" data-segment="machine"></div>
      </div>
      <div class="mt-3 flex items-baseline justify-between gap-4">
        <div class="min-w-0">
          <p class="micro">human-written</p>
          <p class="font-display text-2xl tabular-nums text-verdigris" data-testid="probability-human">${pct(r.human_probability)}</p>
        </div>
        <div class="min-w-0 text-right">
          <p class="micro">machine-generated</p>
          <p class="font-display text-2xl tabular-nums text-seal" data-testid="probability-machine">${pct(r.machine_probability)}</p>
        </div>
      </div>
      <p class="mt-2 text-[0.8125rem] text-graphite">Probabilities are calibrated model estimates, not authorship percentages.</p>
    </div>`;
}

/**
 * True when the backend declined to read the text.
 *
 * Covers every signal the v1 contract can use: the explicit flag, an `abstain`
 * decision, and the response status. Treating an abstention as a verdict is the
 * failure this guards against, so the check is deliberately broad.
 */
export function isAbstention(r: AnalyzeResponse): boolean {
  return (
    !r.evidence_sufficient ||
    r.decision === "abstain" ||
    r.status === "insufficient_evidence"
  );
}

/**
 * Abstained readings replace the prominent human-vs-machine bar.
 *
 * The raw score is still shown — the backend deliberately returns it "for
 * transparency" — but as an explicitly de-emphasised, non-comparative figure.
 * A two-segment bar with large coloured percentages reads as a conclusion even
 * when the stamp says "no verdict", and short inputs can produce extreme raw
 * values (a 23-word casual note measured 0.9856 machine-like), so presenting it
 * as a reading would be actively misleading.
 */
function rawEstimateBlock(r: AnalyzeResponse): string {
  return `
    <div class="w-full" data-testid="raw-estimate">
      <p class="micro">raw model output · not interpretable</p>
      <p class="mt-1.5 font-display text-[1.75rem] tabular-nums text-graphite" data-testid="raw-machine-output">${pct(r.machine_probability)}</p>
      <p class="mt-2 max-w-prose text-[0.8125rem] leading-relaxed text-graphite">
        This is the model's raw machine-likeness score, reported unchanged for
        transparency. For an input this short it is not reliable enough to
        interpret: it is not the probability that the text was machine-generated,
        and no human-versus-machine comparison is offered here.
      </p>
    </div>`;
}

export function renderResultHtml(r: AnalyzeResponse): string {
  const copy = labelCopy(r.label);
  const abstained = isAbstention(r);
  const mixed = r.mixed_low_confidence;

  const stamp = abstained
    ? `no verdict`
    : mixed
      ? `uncertain`
      : `assessed`;
  const stampClass = abstained
    ? "text-graphite"
    : mixed
      ? ""
      : "text-ink-soft";

  const insufficient = abstained
    ? `<aside class="annotation mt-7" data-testid="insufficient-evidence" role="note">
         <p class="micro micro--ink">insufficient evidence</p>
         <p class="mt-1.5 text-[0.9375rem] leading-relaxed">
           This text has ${formatCount(r.word_count)} ${r.word_count === 1 ? "word" : "words"}, below the ${formatInterval(r.gate.min_words)} evidence threshold selected during model validation. The model is reporting the probability below for transparency only — it is not offering a human or machine classification, and no confidence is attached.
         </p>
       </aside>`
    : "";

  const mixedNote =
    mixed && !abstained
      ? `<aside class="annotation mt-7" data-testid="mixed-note" role="note">
           <p class="micro micro--ink">mixed band</p>
           <p class="mt-1.5 text-[0.9375rem] leading-relaxed">
             The estimate of ${formatProbability(r.machine_probability)} sits inside the model's mixed band (${formatBand(r.gate.mixed_band)}), where validation accuracy was close to a coin flip. Treat this reading as undecided rather than as evidence either way.
           </p>
         </aside>`
      : "";

  const warnings =
    r.warnings.length > 0
      ? `<div class="mt-7" data-testid="warnings">
           <h3 class="micro">annotations</h3>
           <ul class="mt-3 space-y-3">
             ${r.warnings
               .map(
                 (w) =>
                   `<li class="annotation text-[0.875rem] leading-relaxed">${escapeHtml(w)}</li>`,
               )
               .join("")}
           </ul>
         </div>`
      : "";

  const confidenceCell =
    r.confidence === null
      ? "not reported (insufficient evidence)"
      : formatProbability(r.confidence);

  return `
    <article class="sheet sheet--raised grain animate-rise p-6 sm:p-9" data-testid="results" aria-labelledby="result-headline">
      <header class="flex flex-wrap items-start justify-between gap-6">
        <div class="min-w-0">
          <p class="micro">reading</p>
          <h2 id="result-headline" class="mt-2 font-display text-[2rem] leading-tight text-balance" data-testid="result-label" data-abstained="${abstained}">${escapeHtml(copy.headline)}</h2>
          <p class="mt-3 max-w-prose text-[1.0625rem] leading-relaxed text-ink-soft text-pretty">${escapeHtml(copy.gloss)}</p>
        </div>
        <span class="stamp shrink-0 animate-stamp ${stampClass}" data-testid="decision-stamp">${stamp}</span>
      </header>
      ${insufficient}
      ${mixedNote}
      <div class="mt-8 grid gap-8 border-t border-rule pt-8 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <section aria-label="${abstained ? "Raw model output (not interpretable)" : "Probability estimate"}">
          ${
            abstained
              ? rawEstimateBlock(r)
              : probabilityBlock(r, copy.tone)
          }
          <dl class="mt-7 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3">
            <div><dt class="micro">decision</dt><dd class="mt-1 text-[0.9375rem]" data-testid="decision">${escapeHtml(abstained ? "No verdict — abstained (below the evidence threshold)" : DECISION_COPY[r.decision] ?? r.decision)}</dd></div>
            <div><dt class="micro">confidence</dt><dd class="mt-1 text-[0.9375rem]" data-testid="confidence">${escapeHtml(confidenceCell)}</dd></div>
            <div><dt class="micro">words</dt><dd class="mt-1 text-[0.9375rem] tabular-nums" data-testid="word-count">${formatCount(r.word_count)}</dd></div>
            <div><dt class="micro">characters</dt><dd class="mt-1 text-[0.9375rem] tabular-nums">${formatCount(r.text_length.characters)}</dd></div>
            <div><dt class="micro">sentences</dt><dd class="mt-1 text-[0.9375rem] tabular-nums">${formatCount(r.text_length.sentences)}</dd></div>
            <div><dt class="micro">paragraphs</dt><dd class="mt-1 text-[0.9375rem] tabular-nums">${formatCount(r.text_length.paragraphs)}</dd></div>
          </dl>
          ${
            r.confidence !== null
              ? `<p class="mt-5 text-[0.8125rem] leading-relaxed text-graphite">Confidence here is the leading class probability, rounded for display — it is not a separately trained confidence model.</p>`
              : ""
          }
        </section>
        <aside class="lg:border-l lg:border-rule lg:pl-8">
          <h3 class="micro">provenance</h3>
          <dl class="mt-3 space-y-2 text-[0.875rem]">
            <div class="flex justify-between gap-4"><dt class="text-graphite">model</dt><dd class="font-mono" data-testid="model-name">${escapeHtml(r.model_name)}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-graphite">version</dt><dd class="font-mono" data-testid="model-version">${escapeHtml(r.model_version)}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-graphite">calibration</dt><dd class="font-mono">${escapeHtml(r.calibrator)}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-graphite">threshold</dt><dd class="font-mono tabular-nums">${r.threshold}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-graphite">evidence gate</dt><dd class="font-mono">${formatInterval(r.gate.min_words)}</dd></div>
          </dl>
          ${warnings}
          <p class="mt-7 text-[0.8125rem] italic leading-relaxed text-graphite">${escapeHtml(r.disclaimer)}</p>
        </aside>
      </div>
      <section class="mt-9 border-t border-rule pt-8" aria-label="Stylistic signals">
        <div class="flex flex-wrap items-baseline justify-between gap-3">
          <h3 class="micro">stylistic signals</h3>
          <p class="micro text-graphite/80">statistical evidence · not proof of authorship</p>
        </div>
        <div class="mt-5">${signalsBlock(r.signals)}</div>
      </section>
      <p class="sr-only" data-testid="segment-note">Paragraph-level analysis is intentionally not shown: Phase 5A found segment scores unreliable outside the training distribution.</p>
    </article>`;
}

/** Render the response into a detached DOM node. */
export function renderResult(r: AnalyzeResponse): HTMLElement {
  const template = document.createElement("template");
  template.innerHTML = renderResultHtml(r).trim();
  const node = template.content.firstElementChild;
  return (node ?? document.createElement("div")) as HTMLElement;
}