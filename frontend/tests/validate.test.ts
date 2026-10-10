import { describe, expect, it } from "vitest";
import { textStats, validateText } from "../src/lib/validate";
import { MAX_TEXT_CHARS } from "../src/lib/types";

describe("textStats", () => {
  it("counts words and characters as the user types", () => {
    expect(textStats("one two three")).toEqual({
      characters: 13,
      words: 3,
      nonWhitespace: 11,
    });
  });

  it("handles empty and whitespace-only input", () => {
    expect(textStats("")).toEqual({ characters: 0, words: 0, nonWhitespace: 0 });
    expect(textStats("   \n\t ")).toEqual({
      characters: 6,
      words: 0,
      nonWhitespace: 0,
    });
  });

  it("counts multiline text across newlines", () => {
    const text = "first line here\n\nsecond line here";
    expect(textStats(text).words).toBe(6);
    expect(textStats(text).characters).toBe(text.length);
  });

  it("handles unicode without dropping characters", () => {
    const text = "café 圖書館 — test";
    expect(textStats(text).characters).toBe(text.length);
    expect(textStats(text).words).toBe(4);
  });
});

describe("validateText", () => {
  it("accepts ordinary text", () => {
    expect(validateText("A normal paragraph of text.").ok).toBe(true);
  });

  it("rejects empty input with a stable code", () => {
    const result = validateText("");
    expect(result.ok).toBe(false);
    expect(result.code).toBe("empty");
    expect(result.message).toMatch(/at least one/i);
  });

  it("rejects whitespace-only input", () => {
    const result = validateText("   \n\t  ");
    expect(result.ok).toBe(false);
    expect(result.code).toBe("empty");
  });

  it("rejects text beyond the backend character limit", () => {
    const result = validateText("x".repeat(MAX_TEXT_CHARS + 1));
    expect(result.ok).toBe(false);
    expect(result.code).toBe("too_long");
    expect(result.message).toContain("30,000");
  });

  it("accepts text exactly at the limit", () => {
    expect(validateText("x".repeat(MAX_TEXT_CHARS)).ok).toBe(true);
  });

  it("accepts a single non-whitespace character", () => {
    expect(validateText("a").ok).toBe(true);
  });
});