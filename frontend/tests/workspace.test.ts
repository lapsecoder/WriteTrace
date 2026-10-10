import { describe, expect, it, vi } from "vitest";
import { AnalyzeWorkspace, EXAMPLE_TEXT } from "../src/lib/workspace";
import { ApiError, type AnalyzeResponse } from "../src/lib/types";
import { renderResult, renderResultHtml, escapeHtml } from "../src/lib/render-result";
import {
  abstainResponse,
  clone,
  mixedResponse,
  okResponse,
} from "./fixtures";

function mountWorkspace(send: typeof AnalyzeWorkspace extends never ? never : (p: unknown, s?: AbortSignal) => Promise<AnalyzeResponse>) {
  document.body.innerHTML = `
    <form data-testid="analyze-form"></form>
    <textarea data-testid="text-input"></textarea>
    <button type="submit" data-testid="submit"></button>
    <p data-counter></p>
    <p data-validation hidden></p>
    <p data-status></p>
    <div data-results hidden></div>
    <div data-error hidden><p data-error-message></p><p data-error-detail></p></div>
    <button type="button" data-testid="example"></button>
    <button type="button" data-testid="clear"></button>
  `;
  const q = <T extends Element>(sel: string) =>
    document.querySelector<T>(sel) as T;

  const elements = {
    form: q<HTMLFormElement>("form"),
    textarea: q<HTMLTextAreaElement>("textarea"),
    submit: q<HTMLButtonElement>('[data-testid="submit"]'),
    counter: q<HTMLElement>("[data-counter]"),
    validation: q<HTMLElement>("[data-validation]"),
    status: q<HTMLElement>("[data-status]"),
    results: q<HTMLElement>("[data-results]"),
    error: q<HTMLElement>("[data-error]"),
    example: q<HTMLButtonElement>('[data-testid="example"]'),
    clear: q<HTMLButtonElement>('[data-testid="clear"]'),
  };
  return {
    elements,
    send,
    submit: elements.submit,
    textarea: elements.textarea,
    results: elements.results,
    error: elements.error,
    counter: elements.counter,
    validation: elements.validation,
    status: elements.status,
  };
}

describe("AnalyzeWorkspace", () => {
  it("counts words and characters as the user types", () => {
    const send = vi.fn();
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    void ws;
    w.elements.textarea.value = "one two three";
    w.elements.textarea.dispatchEvent(new Event("input"));
    expect(w.counter.textContent).toContain("3 words");
    expect(w.counter.textContent).toContain("13 characters");
  });

  it("blocks submission of empty input and shows a validation message", async () => {
    const send = vi.fn();
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "   ";
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(state.kind).toBe("invalid");
    expect(send).not.toHaveBeenCalled();
    expect(w.validation.hidden).toBe(false);
    expect(w.validation.textContent).toMatch(/at least one/i);
  });

  it("sends the exact text with signals enabled", async () => {
    const send = vi.fn().mockResolvedValue(okResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "  Some text with  spaces  ";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await ws.submit();
    const payload = send.mock.calls[0]?.[0];
    // text must be sent verbatim, not trimmed
    expect(payload.text).toBe("  Some text with  spaces  ");
    expect(payload.include_signals).toBe(true);
  });

  it("prevents duplicate submissions while a request is pending", async () => {
    let resolveFn: (r: AnalyzeResponse) => void = () => {};
    const send = vi.fn(
      () => new Promise<AnalyzeResponse>((res) => (resolveFn = res)),
    );
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text to analyse";
    w.elements.textarea.dispatchEvent(new Event("input"));

    const first = ws.submit();
    const second = await ws.submit(); // duplicate while pending
    expect(second.kind).toBe("pending");
    expect(send).toHaveBeenCalledTimes(1);
    resolveFn(okResponse);
    await first;
    expect(ws.currentState.kind).toBe("done");
  });

  it("disables the submit button and marks aria-busy while pending", async () => {
    let resolveFn: (r: AnalyzeResponse) => void = () => {};
    const send = vi.fn(
      () => new Promise<AnalyzeResponse>((res) => (resolveFn = res)),
    );
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text to analyse";
    w.elements.textarea.dispatchEvent(new Event("input"));
    const p = ws.submit();
    expect(w.submit.disabled).toBe(true);
    expect(w.submit.getAttribute("aria-busy")).toBe("true");
    resolveFn(okResponse);
    await p;
    expect(w.submit.disabled).toBe(false);
  });

  it("preserves the text when the request fails", async () => {
    const original = "the text the user typed";
    const send = vi
      .fn()
      .mockRejectedValue(new ApiError("Backend unreachable", "network"));
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = original;
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(state.kind).toBe("failed");
    expect(w.elements.textarea.value).toBe(original);
  });

  it("preserves the text after a 422 validation error", async () => {
    const original = "text the backend will reject";
    const send = vi
      .fn()
      .mockRejectedValue(
        new ApiError("Request validation failed.", "validation", 422, [
          { field: "text", message: "String should have at least 1 character", type: "string_too_short" },
        ]),
      );
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = original;
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(state.kind).toBe("failed");
    if (state.kind === "failed") {
      expect(state.message).toContain("validation");
      expect(state.detail).toContain("text");
    }
    expect(w.elements.textarea.value).toBe(original);
  });

  it("clears results and errors when the user clears the editor", () => {
    const send = vi.fn().mockResolvedValue(okResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    void ws.submit();
    w.elements.clear.click();
    expect(w.elements.textarea.value).toBe("");
    expect(w.elements.results.hidden).toBe(true);
  });

  it("loads the example text without submitting", () => {
    const send = vi.fn();
    const w = mountWorkspace(send as never);
    new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.example.click();
    expect(w.elements.textarea.value).toBe(EXAMPLE_TEXT);
    expect(send).not.toHaveBeenCalled();
  });

  it("lets the user edit and analyse again after a result", async () => {
    const send = vi
      .fn()
      .mockResolvedValueOnce(okResponse)
      .mockResolvedValueOnce(mixedResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "first text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await ws.submit();
    w.elements.textarea.value = "second text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(send).toHaveBeenCalledTimes(2);
    if (state.kind === "done") {
      expect(state.result.label).toBe("mixed");
    }
  });
});

describe("renderResult", () => {
  it("renders a normal result with its real probabilities", () => {
    const html = renderResultHtml(okResponse);
    expect(html).toContain("Likely human-written");
    // exact API values, formatted
    expect(html).toContain("15.3%");
    expect(html).toContain("84.7%");
    expect(html).toContain(okResponse.model_version);
    // the disclaimer is rendered (escaped for safe HTML insertion)
    expect(html).toContain(escapeHtml(okResponse.disclaimer));
    expect(html).toContain("WriteTrace does not establish authorship");
  });

  it("renders abstention clearly and never as a verdict", () => {
    const html = renderResultHtml(abstainResponse);
    expect(html).toContain("Insufficient evidence");
    expect(html).toContain('data-testid="insufficient-evidence"');
    expect(html).toContain("no verdict");
    expect(html).toContain("not reported (insufficient evidence)");
    // must NOT present a definitive classification for an abstained text
    expect(html).not.toContain("likely_human");
    expect(html).not.toContain("likely_machine");
  });

  it("renders the mixed/uncertain state distinctly", () => {
    const html = renderResultHtml(mixedResponse);
    expect(html).toContain('data-testid="mixed-note"');
    expect(html).toContain("uncertain");
    expect(html).toContain("Mixed — uncertain");
  });

  it("renders null confidence as not reported, never as 0%", () => {
    const html = renderResultHtml(abstainResponse);
    expect(html).toContain("not reported");
    expect(html).not.toContain(">0.0%<");
  });

  it("renders each signal the API returned and no others", () => {
    const html = renderResultHtml(okResponse);
    for (const s of okResponse.signals) {
      expect(html).toContain(s.signal);
      expect(html).toContain(s.caveat);
    }
    const featuresInHtml = html.match(/data-signal="/g) ?? [];
    expect(featuresInHtml.length).toBe(okResponse.signals.length);
  });

  it("renders a clear empty state when there are no signals", () => {
    const noSignals = clone(okResponse);
    noSignals.signals = [];
    const html = renderResultHtml(noSignals);
    expect(html).toContain("no signals");
    expect(html).toContain('data-testid="signals-empty"');
  });

  it("never renders segment verdicts (segments are always empty)", () => {
    for (const r of [okResponse, abstainResponse, mixedResponse]) {
      const html = renderResultHtml(r);
      expect(r.segments).toEqual([]);
      expect(html).not.toMatch(/data-testid="segment-result"/);
      expect(html).toContain('data-testid="segment-note"');
    }
  });

  it("shows the warnings the backend actually returned", () => {
    const html = renderResultHtml(abstainResponse);
    expect(abstainResponse.warnings.length).toBeGreaterThan(0);
    expect(html).toContain('data-testid="warnings"');
    for (const w of abstainResponse.warnings) {
      expect(html).toContain(escapeHtml(w).slice(0, 30));
    }
  });

  it("does not fabricate a probability when the API omits signals/meta", () => {
    const minimal = clone(okResponse);
    minimal.confidence = null;
    minimal.signals = [];
    minimal.warnings = [];
    const html = renderResultHtml(minimal);
    expect(html).not.toContain('data-testid="warnings"');
    expect(html).toContain("not reported");
  });

  it("escapes untrusted strings from the API", () => {
    const nasty = clone(okResponse);
    nasty.signals = [
      {
        ...okResponse.signals[0]!,
        interpretation: '<img src=x onerror="alert(1)">',
      },
    ];
    const html = renderResultHtml(nasty);
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img");
  });

  it("builds a real DOM node mountable into the page", () => {
    const node = renderResult(okResponse);
    expect(node.nodeType).toBe(1);
    expect(node.getAttribute("data-testid")).toBe("results");
  });
});