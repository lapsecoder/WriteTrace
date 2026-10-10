import { describe, expect, it } from "vitest";
import {
  formatBand,
  formatCount,
  formatFeatureValue,
  formatProbability,
  formatSigned,
  labelCopy,
} from "../src/lib/format";
import { okResponse, mixedResponse, abstainResponse } from "./fixtures";

describe("formatProbability", () => {
  it("formats real API values to one decimal place", () => {
    expect(formatProbability(okResponse.machine_probability)).toBe("15.3%");
    expect(formatProbability(okResponse.human_probability)).toBe("84.7%");
  });

  it("returns n/a for null/NaN instead of inventing a value", () => {
    expect(formatProbability(null)).toBe("n/a");
    expect(formatProbability(undefined)).toBe("n/a");
    expect(formatProbability(Number.NaN)).toBe("n/a");
  });

  it("refuses to display values outside the probability range", () => {
    expect(formatProbability(1.4)).toBe("n/a");
    expect(formatProbability(-0.2)).toBe("n/a");
  });

  it("keeps the exact 0 and 1 boundaries", () => {
    expect(formatProbability(0)).toBe("0.0%");
    expect(formatProbability(1)).toBe("100.0%");
  });

  it("never rounds the underlying value it was given", () => {
    const raw = okResponse.machine_probability;
    formatProbability(raw);
    expect(okResponse.machine_probability).toBe(raw);
  });
});

describe("formatSigned", () => {
  it("marks machine-leaning contributions with a leading plus", () => {
    expect(formatSigned(0.5062)).toBe("+0.5062");
  });

  it("keeps human-leaning contributions negative", () => {
    expect(formatSigned(-0.0984)).toBe("-0.0984");
  });

  it("handles non-finite input", () => {
    expect(formatSigned(Number.NaN)).toBe("n/a");
  });
});

describe("formatFeatureValue", () => {
  it("passes through ordinary values", () => {
    expect(formatFeatureValue(29.703)).toBe("29.703");
    expect(formatFeatureValue(0)).toBe("0");
  });

  it("renders null as n/a rather than a fabricated zero", () => {
    expect(formatFeatureValue(null)).toBe("n/a");
  });

  it("uses exponential notation for extreme magnitudes", () => {
    expect(formatFeatureValue(123456)).toMatch(/e\+5/);
  });
});

describe("labelCopy", () => {
  it("has editorial copy for every label the API can return", () => {
    const labels = [
      "likely_human",
      "mostly_human",
      "mixed",
      "mostly_machine",
      "likely_machine",
      "insufficient_evidence",
    ] as const;
    for (const label of labels) {
      const copy = labelCopy(label);
      expect(copy.headline.length).toBeGreaterThan(0);
      expect(copy.gloss.length).toBeGreaterThan(0);
    }
  });

  it("never uses proof language in any label explanation", () => {
    const forbidden = ["proves", "proof", "guarantees", "definitely ai"];
    const labels = [
      "likely_human",
      "mostly_human",
      "mixed",
      "mostly_machine",
      "likely_machine",
      "insufficient_evidence",
    ] as const;
    for (const label of labels) {
      const copy = labelCopy(label);
      const blob = `${copy.headline} ${copy.gloss}`.toLowerCase();
      for (const word of forbidden) {
        expect(blob).not.toContain(word);
      }
    }
  });

  it("describes abstention as a non-verdict", () => {
    expect(labelCopy("insufficient_evidence").headline).toBe(
      "Insufficient evidence",
    );
    expect(labelCopy(abstainResponse.label).tone).toBe("neutral");
  });

  it("marks the real mixed response as mixed", () => {
    expect(labelCopy(mixedResponse.label).tone).toBe("mixed");
  });
});

describe("misc formatting", () => {
  it("formats the mixed band from the API gate", () => {
    expect(formatBand(mixedResponse.gate.mixed_band)).toBe("40%–60%");
    expect(formatBand(null)).toBe("n/a");
  });

  it("formats counts with separators", () => {
    expect(formatCount(51995)).toBe("51,995");
  });
});