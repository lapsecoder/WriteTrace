import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { analyze, health, info, API_BASE } from "../src/lib/api";
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
    expect(url).toBe(`${API_BASE}/api/v1/analyze`);
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
    expect(url).toBe(`${API_BASE}/api/v1/health`);
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