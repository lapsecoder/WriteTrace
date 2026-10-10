/**
 * Client-side validation and text metrics.
 *
 * These checks MIRROR the backend rules for fast feedback; they never replace
 * server validation — every submission is still validated by the API, and the
 * text is submitted verbatim (never rewritten).
 */
import { MAX_TEXT_CHARS } from "./types";

export interface TextStats {
  characters: number;
  words: number;
  /** Non-whitespace characters — a useful "is there anything here" signal. */
  nonWhitespace: number;
}

export function textStats(text: string): TextStats {
  const characters = text.length;
  const words = text.trim().length === 0 ? 0 : text.trim().split(/\s+/).length;
  const nonWhitespace = text.replace(/\s/g, "").length;
  return { characters, words, nonWhitespace };
}

export interface ValidationResult {
  ok: boolean;
  /** Stable machine code for tests and rendering. */
  code: "empty" | "too_long" | null;
  message: string | null;
}

/**
 * Match `backend/schemas.AnalyzeRequest`:
 * 1–30,000 characters and at least one non-whitespace character.
 */
export function validateText(text: string): ValidationResult {
  const stats = textStats(text);
  if (stats.nonWhitespace === 0) {
    return {
      ok: false,
      code: "empty",
      message:
        "Enter some text to analyse. WriteTrace needs at least one " +
        "non-whitespace character.",
    };
  }
  if (stats.characters > MAX_TEXT_CHARS) {
    return {
      ok: false,
      code: "too_long",
      message:
        `This text is ${stats.characters.toLocaleString()} characters. ` +
        `The backend accepts up to ${MAX_TEXT_CHARS.toLocaleString()}; ` +
        `remove ${(stats.characters - MAX_TEXT_CHARS).toLocaleString()} characters.`,
    };
  }
  return { ok: true, code: null, message: null };
}