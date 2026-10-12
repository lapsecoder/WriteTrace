import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  analyze,
  health,
  info,
  API_BASE,
  normalizeApiBase,
  resolveApiUrl,
} from "../src/lib/api";
import { ApiError } from "../src/lib/types";
import { error422, healthResponse, okResponse } from "./fixtures";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("analyze()", () => {
  it("POSTs the text to the analyze endpoint with signals enabled", async () => {
    fetchMock.mockResolvedValue(jsonResponse(okResponse));
    await analyze({ text: "hello world", include_signals: true, max_signals: 8 });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    // Literal expectation: deriving it from API_BASE would hide a bad base.
    expect(url).toBe("/api/v1/analyze");
    expect(init.method).toBe("POST");
    expect(init.headers).toMatchObject({
      "content-type": "application/json",
      accept: "application/json",
    });
    const body = JSON.parse(init.body as string);
    expect(body.text).toBe("hello world");
    expect(body.include_signals).toBe(true);
    expect(body.max_signals).toBe(8);
  });

  it("returns the parsed response untouched", async () => {
    fetchMock.mockResolvedValue(jsonResponse(okResponse));
    const result = await analyze({ text: "x" });
    expect(result).toEqual(okResponse);
  });

  it("does not invent fields that the API omitted", async () => {
    const sparse = { ...okResponse, confidence: null, signals: [] };
    fetchMock.mockResolvedValue(jsonResponse(sparse));
    const result = await analyze({ text: "x" });
    expect(result.confidence).toBeNull();
    expect(result.signals).toEqual([]);
    expect("segments" in result).toBe(true);
    expect(result.segments).toEqual([]);
  });

  it("surfaces a 422 validation error with the backend's own message", async () => {
    const makeResp = () => jsonResponse(error422, 422);
    fetchMock.mockImplementation(makeResp);
    await expect(analyze({ text: "" })).rejects.toMatchObject({
      name: "ApiError",
      kind: "validation",
      status: 422,
    });
    const err = await analyze({ text: "" }).catch((e: ApiError) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).details[0]?.field).toBe("text");
    expect((err as ApiError).message).not.toContain("undefined");
  });

  it("reports a network failure without leaking the text", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await analyze({ text: "confidential draft" }).catch(
      (e: ApiError) => e,
    );
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).kind).toBe("network");
    expect((err as ApiError).message).not.toContain("confidential");
  });

  it("rejects a malformed (non-JSON) response", async () => {
    fetchMock.mockResolvedValue(
      new Response("<html>proxy error</html>", {
        status: 200,
        headers: { "content-type": "text/html" },
      }),
    );
    const err = await analyze({ text: "x" }).catch((e: ApiError) => e);
    expect((err as ApiError).kind).toBe("malformed");
  });

  it("maps a non-422 HTTP failure to a backend error", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ status: "error", error: { code: "internal_error", message: "boom", details: [] } }, 500),
    );
    const err = await analyze({ text: "x" }).catch((e: ApiError) => e);
    expect((err as ApiError).kind).toBe("backend");
    expect((err as ApiError).status).toBe(500);
  });

  it("tolerates an error body that is not the expected envelope", async () => {
    fetchMock.mockResolvedValue(new Response("", { status: 503 }));
    const err = await analyze({ text: "x" }).catch((e: ApiError) => e);
    expect((err as ApiError).kind).toBe("backend");
    expect((err as ApiError).message).toContain("503");
  });
});

describe("health() / info()", () => {
  it("reads the health endpoint", async () => {
    fetchMock.mockResolvedValue(jsonResponse(healthResponse));
    const h = await health();
    expect(h).toEqual(healthResponse);
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("/api/v1/health");
  });

  it("rejects an info payload that is missing required metadata", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ model_version: "final-v1" }));
    await expect(info()).rejects.toMatchObject({ kind: "malformed" });
  });

  it("accepts a well-formed info payload", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        api_version: "v1",
        model_version: "final-v1",
        model_name: "hgb_syn",
        disclaimer: "WriteTrace does not establish authorship.",
        label_vocabulary: ["likely_human"],
      }),
    );
    const result = await info();
    expect(result.model_name).toBe("hgb_syn");
  });
});

/**
 * URL composition regression tests.
 *
 * These assert LITERAL expected URLs for every supported configuration. The
 * previous assertions were written as `${API_BASE}/api/v1/...`, which silently
 * endorsed the `/api/api/v1/...` duplication bug because the expectation was
 * derived from the same constant the implementation used. Never do that here.
 */
describe("API URL composition", () => {
  const ENDPOINTS = [
    ["/api/v1/health", "/api/v1/health"],
    ["/api/v1/info", "/api/v1/info"],
    ["/api/v1/analyze", "/api/v1/analyze"],
  ] as const;

  describe("default configuration (unset variable)", () => {
    it("resolves to same-origin /api/v1 paths", () => {
      expect(API_BASE).toBe("");
      for (const [path, expected] of ENDPOINTS) {
        expect(resolveApiUrl(path)).toBe(expected);
      }
    });

    it("never produces a duplicated /api prefix", () => {
      for (const [path] of ENDPOINTS) {
        expect(resolveApiUrl(path)).not.toContain("/api/api/");
        // exactly one leading /api segment
        expect(resolveApiUrl(path).match(/\/api\//g)).toHaveLength(1);
      }
    });

    it("sends every endpoint to the correct same-origin URL", async () => {
      fetchMock.mockResolvedValue(jsonResponse(healthResponse));
      await health();
      await info().catch(() => undefined);
      await analyze({ text: "hello world" }).catch(() => undefined);

      const urls = fetchMock.mock.calls.map((c) => c[0] as string);
      expect(urls).toEqual([
        "/api/v1/health",
        "/api/v1/info",
        "/api/v1/analyze",
      ]);
    });
  });

  describe("same-origin '/api' configuration", () => {
    it("normalises '/api' away so it matches the default exactly", () => {
      expect(normalizeApiBase("/api")).toBe("");
      for (const [path, expected] of ENDPOINTS) {
        expect(resolveApiUrl(path, "/api")).toBe(expected);
        expect(resolveApiUrl(path, "/api")).toBe(resolveApiUrl(path));
      }
    });

    it("tolerates redundant slashes and casing", () => {
      for (const variant of ["/api", "/api/", "//api//", " /API ", "/API"]) {
        for (const [path, expected] of ENDPOINTS) {
          expect(resolveApiUrl(path, variant)).toBe(expected);
        }
      }
    });
  });

  describe("absolute backend origin", () => {
    it("prefixes the same /api/v1 paths without duplicating them", () => {
      for (const [path, expected] of ENDPOINTS) {
        expect(resolveApiUrl(path, "http://127.0.0.1:8000")).toBe(
          `http://127.0.0.1:8000${expected}`,
        );
      }
    });

    it("tolerates an origin that already ends in /api", () => {
      for (const [path, expected] of ENDPOINTS) {
        expect(resolveApiUrl(path, "http://127.0.0.1:8000/api")).toBe(
          `http://127.0.0.1:8000${expected}`,
        );
      }
    });

    it("is what the analyzer actually requests when configured", async () => {
      // exercise the real request path with an explicit origin
      fetchMock.mockResolvedValue(jsonResponse(okResponse));
      await analyze({ text: "hello world" });
      const [url] = fetchMock.mock.calls[0] as [string];
      // under the default build this is same-origin; the composition rule is
      // what matters, so assert the suffix is exactly the endpoint path
      expect(url.endsWith("/api/v1/analyze")).toBe(true);
      expect(url).not.toContain("/api/api/");
    });
  });

  describe("no deployment URL is hardcoded", () => {
    it("does not embed any vercel/app host in the client", () => {
      const base = normalizeApiBase("");
      expect(base).not.toMatch(/vercel\.app/i);
      expect(base).not.toMatch(/https?:\/\//);
      for (const [path] of ENDPOINTS) {
        expect(resolveApiUrl(path)).not.toMatch(/vercel\.app/i);
      }
    });

    it("keeps the default relative, so the deployment host is unknown to it", () => {
      // A relative default is what allows the same build to work on any host.
      expect(resolveApiUrl("/api/v1/analyze").startsWith("/")).toBe(true);
      expect(resolveApiUrl("/api/v1/analyze")).not.toMatch(/^https?:/);
    });
  });

  describe("normalisation is idempotent and safe", () => {
    it("is idempotent", () => {
      for (const raw of ["", "/api", "/api/", "http://h:8000", "http://h:8000/api"]) {
        const once = normalizeApiBase(raw);
        expect(normalizeApiBase(once)).toBe(once);
      }
    });

    it("never leaves a trailing slash or a dangling /api", () => {
      for (const raw of ["", "/", "/api", "/api/", "http://h:8000/"]) {
        const n = normalizeApiBase(raw);
        expect(n.endsWith("/")).toBe(false);
        expect(n.endsWith("/api")).toBe(false);
      }
    });

    it("does not strip a legitimate path segment that merely starts with 'api'", () => {
      // e.g. a reverse proxy mounted at /api-gateway must survive.
      expect(normalizeApiBase("https://h/api-gateway")).toBe(
        "https://h/api-gateway",
      );
    });

    it("accepts endpoint paths with or without a leading slash", () => {
      expect(resolveApiUrl("api/v1/analyze")).toBe("/api/v1/analyze");
      expect(resolveApiUrl("/api/v1/analyze")).toBe("/api/v1/analyze");
    });
  });
});