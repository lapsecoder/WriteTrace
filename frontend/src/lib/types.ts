/**
 * TypeScript mirrors of the WriteTrace backend contract.
 *
 * Source of truth: `backend/schemas.py` + `docs/api.md` (API v1).
 * These types are hand-synced with the Pydantic models; they are types only
 * and never re-implement inference.
 */

/** `Literal["machine", "human", "abstain"]` */
export type Decision = "machine" | "human" | "abstain";

/** `Literal["likely_human", "mostly_human", "mixed", "mostly_machine",
 *  "likely_machine", "insufficient_evidence"]` */
export type Label =
  | "likely_human"
  | "mostly_human"
  | "mixed"
  | "mostly_machine"
  | "likely_machine"
  | "insufficient_evidence";

export type Status = "ok" | "insufficient_evidence";

export type SignalDirection = "machine" | "human" | "neutral";

export interface TextLength {
  characters: number;
  words: number;
  sentences: number;
  paragraphs: number;
}

export interface Signal {
  signal: string;
  label: string;
  family: string;
  value: number | null;
  reference_median: number;
  contribution: number;
  direction: SignalDirection;
  interpretation: string;
  caveat: string;
}

/** Reserved for future research; the v1 API always returns `[]`. */
export interface Segment {
  status: string;
  count: number;
  note: string;
}

export interface GateInfo {
  min_words: number | null;
  mixed_band: [number, number] | null;
}

export interface AnalyzeResponse {
  status: Status;
  label: Label;
  decision: Decision;
  machine_probability: number;
  human_probability: number;
  /** `max(machine, human)` rounded to 4 dp; **null when evidence is
   *  insufficient** and absent from the model when not applicable. Never a
   *  separately trained confidence model. */
  confidence: number | null;
  word_count: number;
  text_length: TextLength;
  evidence_sufficient: boolean;
  mixed_low_confidence: boolean;
  signals: Signal[];
  segments: Segment[];
  gate: GateInfo;
  model_version: string;
  model_name: string;
  calibrator: string;
  threshold: number;
  warnings: string[];
  disclaimer: string;
}

export interface AnalyzeRequest {
  text: string;
  include_signals?: boolean;
  max_signals?: number;
}

export interface ErrorDetail {
  field: string | null;
  message: string;
  type: string | null;
}

export interface ErrorResponse {
  status: "error";
  error: {
    code: string;
    message: string;
    details: ErrorDetail[];
  };
}

export interface HealthResponse {
  status: "ok";
  api_version: string;
  model_version: string;
  model_name: string;
  loaded: boolean;
}

export interface InfoResponse {
  api_version: string;
  model_version: string;
  model_name: string;
  calibrator: string;
  threshold: number;
  n_features: number;
  syntactic_features: boolean;
  gate: GateInfo;
  pipeline_version: string;
  label_vocabulary: string[];
  data_policy: Record<string, unknown>;
  global_importance: Array<{
    feature: string;
    label: string;
    delta_pr_auc: number;
  }>;
  importance_method: string | null;
  eval_summary: Record<string, Record<string, number>>;
  disclaimer: string;
}

/** Limits mirrored from `backend/schemas.py` (MAX_TEXT_CHARS). */
export const MAX_TEXT_CHARS = 30_000;
export const MIN_TEXT_CHARS = 1;

/** Error raised by the API client for transport/HTTP failures. */
export class ApiError extends Error {
  readonly kind:
    | "validation"
    | "network"
    | "backend"
    | "malformed";
  readonly status: number | null;
  readonly details: ErrorDetail[];
  readonly code: string | null;

  constructor(
    message: string,
    kind: ApiError["kind"],
    status: number | null = null,
    details: ErrorDetail[] = [],
    code: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.details = details;
    this.code = code;
  }
}