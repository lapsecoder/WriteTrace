/**
 * WriteTrace API client.
 *
 * Responsibilities: build the request, call the configured backend, and turn
 * every failure mode into a typed `ApiError`. It performs **no inference of
 * its own** and never fabricates a result.
 *
 * Base URL resolution
 * --------------------
 * One rule, applied everywhere:
 *
 *   **endpoint paths are always the full `/api/v1/...` path, and
 *   `PUBLIC_WRITETRACE_API_BASE` is the ORIGIN they are appended to.**
 *
 * The origin defaults to empty, meaning "same origin as this page", so the
 * default request is `/api/v1/analyze`. An explicitly supplied `/api` suffix
 * is tolerated and normalised away, so `/api`, `""` and an unset variable all
 * resolve to the same URLs and the `/api` segment can never be duplicated
 * (it previously produced `/api/api/v1/...`, which 404s).
 *
 * Both supported deployments agree on this path semantics:
 *   - local dev/preview: `astro.config.mjs` proxies `/api` to the local
 *     FastAPI server WITHOUT stripping the prefix, so `/api/v1/analyze`
 *     arrives at the backend as `/api/v1/analyze`;
 *   - same-origin hosting: the request stays on the page's own origin and is
 *     routed to the API alongside the static site.
 *
 * A separate backend origin (e.g. "http://127.0.0.1:8000") is still supported
 * and simply prefixes the same paths; no deployment URL is ever hardcoded.
 */
import {
  ApiError,
  type AnalyzeRequest,
  type AnalyzeResponse,
  type ErrorDetail,
  type HealthResponse,
  type InfoResponse,
} from "./types";

/**
 * Normalise a configured base into a pure origin prefix.
 *
 * Strips trailing slashes and tolerates an explicit trailing `/api`, which is
 * what makes `/api`, `""` and an unset variable equivalent. Idempotent.
 */
export function normalizeApiBase(base: string): string {
  const trimmed = base.trim();
  // Preserve the scheme separator ("http://") while collapsing any other
  // run of slashes, so redundant input like "//api//" normalises cleanly.
  const scheme = trimmed.match(/^([a-z][a-z0-9+.-]*:\/\/)(.*)$/i);
  const prefix = scheme ? scheme[1]! : "";
  const rest = scheme ? scheme[2]! : trimmed;
  return `${prefix}${rest.replace(/\/{2,}/g, "/").replace(/\/+$/, "")}`
    .replace(/\/api$/i, "");
}

/**
 * The configured API origin. Empty by default, meaning same-origin, which
 * yields `/api/v1/...` requests.
 */
export const API_BASE: string = normalizeApiBase(
  import.meta.env?.PUBLIC_WRITETRACE_API_BASE ?? "",
);

/**
 * Compose the final request URL for an endpoint `path`.
 *
 * `path` is always a full `/api/v1/...` path. Exported so the composition rule
 * can be tested directly against every supported configuration, independently
 * of the build-time value baked into `API_BASE`.
 */
export function resolveApiUrl(path: string, base: string = API_BASE): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${normalizeApiBase(base)}${suffix}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parse the backend's uniform error envelope, tolerating unexpected shapes. */
function readErrorBody(body: unknown): {
  code: string | null;
  message: string | null;
  details: ErrorDetail[];
} {
  if (!isRecord(body) || !isRecord(body.error)) {
    return { code: null, message: null, details: [] };
  }
  const err = body.error;
  const details: ErrorDetail[] = [];
  if (Array.isArray(err.details)) {
    for (const d of err.details) {
      if (isRecord(d) && typeof d.message === "string") {
        details.push({
          field: typeof d.field === "string" ? d.field : null,
          message: d.message,
          type: typeof d.type === "string" ? d.type : null,
        });
      }
    }
  }
  return {
    code: typeof err.code === "string" ? err.code : null,
    message: typeof err.message === "string" ? err.message : null,
    details,
  };
}

/** Guard against a proxy/HTML page being returned as JSON. */
function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new ApiError(
      "The backend returned a response that was not valid JSON.",
      "malformed",
    );
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { expectStatus?: number[] } = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(resolveApiUrl(path), {
      ...init,
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        ...(init.headers ?? {}),
      },
    });
  } catch {
    // Deliberately does NOT include the submitted text in the message.
    throw new ApiError(
      "Could not reach the WriteTrace backend. Make sure it is running " +
        `(expected ${resolveApiUrl("/api/v1/health")}).`,
      "network",
    );
  }

  const text = await response.text();
  let body: unknown = null;
  if (text.length > 0) {
    body = parseJson(text);
  }

  if (!response.ok) {
    const { code, message, details } = readErrorBody(body);
    if (response.status === 422) {
      throw new ApiError(
        message ?? "The backend rejected this text as invalid.",
        "validation",
        422,
        details,
        code,
      );
    }
    throw new ApiError(
      message ??
        `The backend responded with HTTP ${response.status}. Please try again.`,
      "backend",
      response.status,
      details,
      code,
    );
  }

  return body as T;
}

/** POST /api/v1/analyze — the single inference call the UI makes. */
export function analyze(
  payload: AnalyzeRequest,
  signal?: AbortSignal,
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/api/v1/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

/** GET /api/v1/health — used for the quiet availability note in the masthead. */
export async function health(signal?: AbortSignal): Promise<HealthResponse> {
  const body = await request<unknown>("/api/v1/health", {
    method: "GET",
    signal,
  });
  if (!isRecord(body) || typeof body.status !== "string") {
    throw new ApiError("Unexpected health payload from the backend.", "malformed");
  }
  return body as unknown as HealthResponse;
}

/** GET /api/v1/info — model/gate/label metadata (no filesystem paths). */
export async function info(signal?: AbortSignal): Promise<InfoResponse> {
  const body = await request<unknown>("/api/v1/info", {
    method: "GET",
    signal,
  });
  if (
    !isRecord(body) ||
    typeof body.model_version !== "string" ||
    typeof body.model_name !== "string" ||
    typeof body.disclaimer !== "string" ||
    !Array.isArray(body.label_vocabulary)
  ) {
    throw new ApiError(
      "Unexpected info payload from the backend.",
      "malformed",
    );
  }
  return body as unknown as InfoResponse;
}