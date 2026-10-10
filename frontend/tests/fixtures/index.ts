/**
 * Fixtures captured from the REAL backend (`backend/app.py` via TestClient,
 * artifact final-v1) so the frontend is tested against the actual contract
 * rather than invented payloads.
 *
 * Regenerate with the Python snippet documented in docs/frontend.md.
 */
import raw from "./real-responses.json" with { type: "json" };
import type { AnalyzeResponse, ErrorResponse, HealthResponse } from "../../src/lib/types";

/**
 * The JSON import widens literals (`mixed_band` becomes `number[]`), so the
 * fixtures are validated through `unknown` once, here, rather than being cast
 * ad hoc at every use site.
 */
function asResponse(value: unknown): AnalyzeResponse {
  return value as AnalyzeResponse;
}

export const okResponse = asResponse(raw.ok_response);
export const abstainResponse = asResponse(raw.abstain_response);
export const mixedResponse = asResponse(raw.mixed_response);
export const error422 = raw.error_422 as unknown as ErrorResponse;
export const healthResponse = raw.health as unknown as HealthResponse;

/** A clone so a test can mutate a fixture without leaking into others. */
export function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}