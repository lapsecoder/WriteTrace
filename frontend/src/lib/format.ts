/**
 * Presentation formatting helpers.
 *
 * These only *format* values returned by the API; they never compute,
 * round-trip or invent probabilities. Percentages are rendered to one decimal
 * place purely for display while the underlying value stays untouched.
 */
import type { Label } from "./types";

export interface LabelCopy {
  /** Short editorial heading, e.g. "Likely machine-generated". */
  headline: string;
  /** One-line explanation of what the label means. */
  gloss: string;
  tone: "human" | "mixed" | "machine" | "neutral";
}

const COPY: Record<Label, LabelCopy> = {
  likely_human: {
    headline: "Likely human-written",
    gloss:
      "The writing patterns resemble human-written text more than " +
      "machine-generated text under the model's evaluation conditions.",
    tone: "human",
  },
  mostly_human: {
    headline: "Mostly human-written",
    gloss:
      "Patterns lean human-written, but the estimate is not decisive " +
      "enough to call with confidence.",
    tone: "human",
  },
  mixed: {
    headline: "Mixed — uncertain",
    gloss:
      "The probability sits inside the model's mixed band, where " +
      "validation accuracy was near chance. Treat this as undecided.",
    tone: "mixed",
  },
  mostly_machine: {
    headline: "Mostly machine-generated",
    gloss:
      "Patterns lean machine-generated, but the estimate is not decisive " +
      "enough to call with confidence.",
    tone: "machine",
  },
  likely_machine: {
    headline: "Likely machine-generated",
    gloss:
      "The writing patterns resemble machine-generated text more than " +
      "human-written text under the model's evaluation conditions.",
    tone: "machine",
  },
  insufficient_evidence: {
    headline: "Insufficient evidence",
    gloss:
      "The text is below the model's evidence threshold, so no conclusion " +
      "is drawn. The probability is shown for transparency only.",
    tone: "neutral",
  },
};

export function labelCopy(label: Label): LabelCopy {
  return COPY[label] ?? {
    headline: "Unlabelled result",
    gloss: "The backend returned a label this interface does not know.",
    tone: "neutral",
  };
}

export const DECISION_COPY: Record<string, string> = {
  machine: "Model decision: machine-generated",
  human: "Model decision: human-written",
  abstain: "Model decision: abstained (below the evidence threshold)",
};

/** 0.9334 → "93.3%". Values outside [0,1] are shown as "n/a", never clamped. */
export function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 0 || value > 1) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatSigned(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "n/a";
  const s = value.toFixed(digits);
  return value > 0 ? `+${s}` : s;
}

export function formatFeatureValue(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 10_000)) {
    return value.toExponential(2);
  }
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded);
}

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

/** Inclusive list formatter for labels like "1–50 words". */
export function formatBand(band: readonly [number, number] | null): string {
  if (!band) return "n/a";
  const [lo, hi] = band;
  return `${formatPercentValue(lo)}–${formatPercentValue(hi)}`;
}

function formatPercentValue(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

export function formatInterval(
  gateMinWords: number | null | undefined,
): string {
  if (gateMinWords === null || gateMinWords === undefined) return "n/a";
  return `${formatCount(gateMinWords)} words`;
}