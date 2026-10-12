import { describe, expect, it, vi } from "vitest";
import { AnalyzeWorkspace, EXAMPLE_TEXT } from "../src/lib/workspace";
import { ApiError, type AnalyzeResponse } from "../src/lib/types";
import {
  renderResult,
  renderResultHtml,
  escapeHtml,
  isAbstention,
} from "../src/lib/render-result";
import {
  abstainExtremeMachineResponse,
  abstainHighMachineResponse,
  abstainResponse,
  clone,
  error422,
  mixedResponse,
  okResponse,
} from "./fixtures";

function mountWorkspace(send: typeof AnalyzeWorkspace extends never ? never : (p: unknown, s?: AbortSignal) => Promise<AnalyzeResponse>) {
  // The error box mirrors the real static markup in index.astro: static chrome
  // plus two nodes whose TEXT the controller fills in. If a fix ever clears the
  // container's children, `stillAttached` below catches it.
  document.body.innerHTML = `
    <form data-testid="analyze-form"></form>
    <textarea data-testid="text-input"></textarea>
    <button type="submit" data-testid="submit"></button>
    <p data-counter></p>
    <p data-validation hidden></p>
    <p data-status></p>
    <div data-results hidden></div>
    <div data-error hidden role="alert">
      <p data-error-heading>analysis could not be completed</p>
      <p data-error-message></p>
      <p data-error-detail></p>
      <p data-error-reassure>Your text is still in the editor.</p>
    </div>
    <button type="button" data-testid="example"></button>
    <button type="button" data-testid="clear"></button>
  `;
  const q = <T extends Element>(sel: string) =>
    document.querySelector<T>(sel) as T;

  const elements = {
    form: q<HTMLFormElement>("form"),
    textarea: q<HTMLTextAreaElement>("[data-testid=\"text-input\"]"),
    submit: q<HTMLButtonElement>('[data-testid="submit"]'),
    counter: q<HTMLElement>("[data-counter]"),
    validation: q<HTMLElement>("[data-validation]"),
    status: q<HTMLElement>("[data-status]"),
    results: q<HTMLElement>("[data-results]"),
    error: q<HTMLElement>("[data-error]"),
    errorMessage: q<HTMLElement>("[data-error-message]"),
    errorDetail: q<HTMLElement>("[data-error-detail]"),
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
    errorMessage: elements.errorMessage,
    errorDetail: elements.errorDetail,
    counter: elements.counter,
    validation: elements.validation,
    status: elements.status,
    /** True while the error box's static children are still in the document. */
    stillAttached: () =>
      elements.error.contains(document.querySelector("[data-error-message]")) &&
      elements.error.contains(document.querySelector("[data-error-reassure]")),
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
    // and must NOT present the prominent human-vs-machine probability bar
    expect(html).not.toContain('data-testid="probability-bar"');
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

/**
 * Insufficient-evidence presentation.
 *
 * An abstention must never look like a reading. These tests pin the behaviour
 * for BOTH directions of the raw estimate, because a high raw score on a short
 * text is the case that previously rendered as a confident red "machine"
 * verdict. Fixtures are real backend responses, not invented payloads.
 */
describe("insufficient evidence is never presented as a verdict", () => {
  const abstentions = [
    ["low raw estimate", abstainResponse],
    ["high raw estimate", abstainHighMachineResponse],
    ["extreme raw estimate", abstainExtremeMachineResponse],
  ] as const;

  /** Collapse template newlines/indentation so prose assertions are stable. */
  const flat = (html: string) => html.replace(/\s+/g, " ");

  for (const [name, response] of abstentions) {
    describe(name, () => {
      const html = renderResultHtml(response);
      const text = flat(html);

      it("is detected as an abstention", () => {
        expect(isAbstention(response)).toBe(true);
      });

      it("keeps the explicit insufficient-evidence state", () => {
        expect(html).toContain('data-testid="insufficient-evidence"');
        expect(html).toContain("Insufficient evidence");
        expect(html).toContain("no verdict");
        expect(response.label).toBe("insufficient_evidence");
        expect(response.decision).toBe("abstain");
      });

      it("does NOT render the human-vs-machine probability bar", () => {
        expect(html).not.toContain('data-testid="probability-bar"');
        expect(html).not.toContain('data-segment="human"');
        expect(html).not.toContain('data-segment="machine"');
        // the large coloured verdict percentages must be gone entirely
        expect(html).not.toContain('data-testid="probability-human"');
        expect(html).not.toContain('data-testid="probability-machine"');
      });

      it("does NOT show a categorical human/machine verdict", () => {
        // The headline must stay the abstention label...
        expect(html).toContain("Insufficient evidence");
        // ...and none of the 5-level verdict headlines may appear.
        // (The API disclaimer legitimately contains the words
        // "human-written"/"machine-generated" and must be preserved verbatim,
        // so the check targets the verdict headlines, not those words.)
        for (const banned of [
          "Likely human-written",
          "Likely machine-generated",
          "Mostly human-written",
          "Mostly machine-generated",
          "Mixed — uncertain",
        ]) {
          expect(text).not.toContain(banned);
        }
        expect(html).toContain("No verdict");
      });

      it("explains the raw estimate is not reliable enough to interpret", () => {
        expect(html).toContain('data-testid="raw-estimate"');
        expect(text).toContain("not reliable enough to interpret");
        expect(text).toContain("not interpretable");
      });

      it("still reports the raw value verbatim (nothing discarded)", () => {
        const expected = `${(response.machine_probability * 100).toFixed(1)}%`;
        expect(html).toContain(expected);
        expect(html).toContain('data-testid="raw-machine-output"');
        // confidence stays null and is never invented
        expect(response.confidence).toBeNull();
        expect(html).toContain("not reported (insufficient evidence)");
        expect(html).not.toContain(">0.0%<");
      });

      it("preserves the observed word count and the threshold explanation", () => {
        expect(html).toContain(
          response.word_count.toLocaleString("en-US"),
        );
        expect(html).toContain(
          `${response.gate.min_words?.toLocaleString("en-US")} words`,
        );
        expect(text).toContain("below the");
        expect(text).toContain("evidence threshold");
      });

      it("keeps the disclaimer and provenance", () => {
        expect(html).toContain(escapeHtml(response.disclaimer));
        expect(html).toContain('data-testid="model-version"');
        expect(html).toContain(response.model_version);
      });
    });
  }

  it("frames an abstention identically regardless of raw direction", () => {
    // Compare only the framing (header/stamp/annotation/raw-output/decision/
    // provenance), not the observed counts or the signals: those legitimately
    // differ per input, and requirement 2 says the word count is PRESERVED.
    // What must not vary is the framing or the reported raw percentage.
    const framing = (h: string) =>
      h
        .slice(0, h.indexOf('aria-label="Stylistic signals"'))
        // observed counts differ by input -> not part of the framing
        .replace(/>\d+<\/dd>/g, ">N</dd>")
        .replace(/\d+\.\d%/g, "N%")
        .replace(/\d[\d,]*(\.\d+)?\s*(words|characters|sentences|paragraphs)/g,
                 "N $2")
        .replace(/\s+/g, " ");

    const low = framing(renderResultHtml(abstainResponse));
    const high = framing(renderResultHtml(abstainHighMachineResponse));
    const extreme = framing(renderResultHtml(abstainExtremeMachineResponse));

    expect(low).toEqual(high);
    expect(low).toEqual(extreme);

    // The observed word count IS still rendered per input (requirement 2).
    expect(renderResultHtml(abstainResponse)).toContain(
      'data-testid="word-count">2<',
    );
    expect(renderResultHtml(abstainHighMachineResponse)).toContain(
      'data-testid="word-count">40<',
    );
    // ...while the underlying documents genuinely differ (raw values differ).
    expect(renderResultHtml(abstainResponse)).not.toEqual(
      renderResultHtml(abstainHighMachineResponse),
    );
  });

  it("detects abstention from each contract signal independently", () => {
    const byFlag = clone(okResponse);
    byFlag.evidence_sufficient = false;
    expect(isAbstention(byFlag)).toBe(true);

    const byDecision = clone(okResponse);
    byDecision.decision = "abstain";
    expect(isAbstention(byDecision)).toBe(true);

    const byStatus = clone(okResponse);
    byStatus.status = "insufficient_evidence";
    expect(isAbstention(byStatus)).toBe(true);

    expect(isAbstention(okResponse)).toBe(false);
    expect(isAbstention(mixedResponse)).toBe(false);
  });
});

describe("evidence-sufficient results are unchanged", () => {
  it("keeps the full probability bar and percentages for a normal result", () => {
    const html = renderResultHtml(okResponse);
    expect(isAbstention(okResponse)).toBe(false);
    expect(html).toContain('data-testid="probability-bar"');
    expect(html).toContain('data-segment="human"');
    expect(html).toContain('data-segment="machine"');
    expect(html).toContain('data-testid="probability-human"');
    expect(html).toContain('data-testid="probability-machine"');
    expect(html).toContain("Likely human-written");
    expect(html).toContain("assessed");
    expect(html).toContain("human-written");
    expect(html).toContain("machine-generated");
    expect(html).not.toContain('data-testid="raw-estimate"');
    // decision row keeps the normal wording
    expect(html).toContain("Model decision: human-written");
    // real values, formatted
    expect(html).toContain("15.3%");
    expect(html).toContain("84.7%");
  });

  it("keeps the mixed/uncertain state reading as a result, not an abstention", () => {
    const html = renderResultHtml(mixedResponse);
    expect(isAbstention(mixedResponse)).toBe(false);
    expect(html).toContain('data-testid="mixed-note"');
    expect(html).toContain("uncertain");
    expect(html).toContain("Mixed — uncertain");
    // a mixed reading still gets the bar: it is a reading, just an uncertain one
    expect(html).toContain('data-testid="probability-bar"');
    expect(html).not.toContain('data-testid="insufficient-evidence"');
    expect(html).not.toContain('data-testid="raw-estimate"');
  });
});

describe("API error states remain safe", () => {
  it("does not render any result markup for a validation error payload", () => {
    // The error envelope is surfaced by the controller, not the renderer; this
    // guards against anyone "fixing" abstention by rendering error bodies.
    expect(error422.error.code).toBe("validation_error");
    expect(error422.error.message.length).toBeGreaterThan(0);
    const html = renderResultHtml(abstainResponse);
    expect(html).not.toContain(error422.error.message);
  });

  it("escapes untrusted strings even in the abstention path", () => {
    const nasty = clone(abstainExtremeMachineResponse);
    nasty.disclaimer = '<img src=x onerror="alert(1)">';
    const html = renderResultHtml(nasty);
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img");
  });
});
/**
 * Defect D1 (Phase 6C audit): analysis failures showed an EMPTY error box.
 *
 * Root cause: the controller cleared the error box's CHILDREN
 * (`error.textContent = ""` / `error.innerHTML = ""`), detaching the message
 * and detail nodes that the page then wrote into. The box stayed visible but
 * blank.
 *
 * Every test below fails against the old implementation.
 */
describe("error messages are visible (defect D1)", () => {
  it("shows a readable message for a network failure", async () => {
    const send = vi
      .fn()
      .mockRejectedValue(new ApiError("Could not reach the WriteTrace backend.",
                                       "network"));
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "some text to analyse here";
    w.elements.textarea.dispatchEvent(new Event("input"));

    const state = await new AnalyzeWorkspace(w.elements, { send: send as never })
      .submit();

    expect(state.kind).toBe("failed");
    expect(w.error.hidden).toBe(false);
    // the message is actually ON SCREEN, not just in a detached node
    expect(w.error.textContent).toContain("Could not reach");
    expect(w.errorMessage!.textContent).toContain("Could not reach");
  });

  it("keeps the error box's static children attached", async () => {
    const send = vi.fn().mockRejectedValue(new ApiError("boom", "network"));
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await new AnalyzeWorkspace(w.elements, { send: send as never }).submit();

    expect(w.stillAttached()).toBe(true);
    // the reassurance line the user needs ("your text is still in the editor")
    expect(w.error.textContent).toContain("still in the editor");
  });

  it("surfaces a backend 5xx message", async () => {
    const send = vi.fn().mockRejectedValue(
      new ApiError("The backend responded with HTTP 500.", "backend", 500));
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await new AnalyzeWorkspace(w.elements, { send: send as never }).submit();

    expect(w.error.hidden).toBe(false);
    expect(w.error.textContent).toContain("HTTP 500");
  });

  it("surfaces a 422 validation message with its detail line", async () => {
    const send = vi.fn().mockRejectedValue(
      new ApiError("Request validation failed. See details.", "validation", 422,
                   [{ field: "text", message: "at least 1 character", type: "x" }]));
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await new AnalyzeWorkspace(w.elements, { send: send as never }).submit();

    expect(w.error.textContent).toContain("validation");
    expect(w.errorDetail!.hidden).toBe(false);
    expect(w.errorDetail!.textContent).toContain("at least 1 character");
  });

  it("hides the detail line when there is no detail", async () => {
    const send = vi.fn().mockRejectedValue(new ApiError("plain", "network"));
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await new AnalyzeWorkspace(w.elements, { send: send as never }).submit();
    expect(w.errorDetail!.hidden).toBe(true);
  });

  it("never leaks stack traces, paths or the submitted text", async () => {
    const leaky = new ApiError(
      'Traceback (most recent call last): File "C:\\secret\\app.py", line 3',
      "backend", 500);
    const send = vi.fn().mockRejectedValue(leaky);
    const w = mountWorkspace(send as never);
    void new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "my confidential draft text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await new AnalyzeWorkspace(w.elements, { send: send as never }).submit();

    const shown = w.error.textContent;
    // the client must render its OWN safe copy, never the raw error text
    expect(shown).not.toContain("Traceback");
    expect(shown).not.toContain("C:\\secret");
    expect(shown).not.toContain("confidential draft");
  });

  it("clears a previous error when a new analysis starts", async () => {
    const send = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("first failure", "network"))
      .mockResolvedValueOnce(okResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text one";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await ws.submit();
    expect(w.error.hidden).toBe(false);

    // a new request clears the stale error rather than leaving it on screen
    w.elements.textarea.value = "text two";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await ws.submit();
    expect(w.error.hidden).toBe(true);
    expect(w.errorMessage!.textContent).toBe("");
    expect(w.error.textContent).not.toContain("first failure");
  });

  it("clears the error when analysis succeeds", async () => {
    const send = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("transient", "network"))
      .mockResolvedValueOnce(okResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    w.elements.textarea.value = "text one";
    w.elements.textarea.dispatchEvent(new Event("input"));
    await ws.submit();
    expect(w.error.hidden).toBe(false);

    w.elements.textarea.value = "text two";
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(state.kind).toBe("done");
    expect(w.error.hidden).toBe(true);
    expect(w.stillAttached()).toBe(true);
  });

  it("keeps the error box's live region semantics", () => {
    const w = mountWorkspace(vi.fn() as never);
    void new AnalyzeWorkspace(w.elements, { send: vi.fn() as never });
    expect(w.error.getAttribute("role")).toBe("alert");
  });

  it("survives many consecutive failures", async () => {
    const send = vi.fn().mockRejectedValue(new ApiError("still down", "network"));
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    for (let i = 0; i < 5; i++) {
      w.elements.textarea.value = `attempt ${i}`;
      w.elements.textarea.dispatchEvent(new Event("input"));
      await ws.submit();
      expect(w.error.hidden).toBe(false);
      expect(w.error.textContent).toContain("still down");
      expect(w.stillAttached()).toBe(true);
    }
  });

  it("tolerates a missing message node without throwing", async () => {
    const send = vi.fn().mockRejectedValue(new ApiError("boom", "network"));
    const w = mountWorkspace(send as never);
    const els = { ...w.elements, errorMessage: null, errorDetail: null };
    const ws = new AnalyzeWorkspace(els, { send: send as never });
    w.elements.textarea.value = "text";
    w.elements.textarea.dispatchEvent(new Event("input"));
    const state = await ws.submit();
    expect(state.kind).toBe("failed");
    expect(w.error.hidden).toBe(false);
  });
});

/**
 * The Analyse button must reflect whether a submission can actually succeed.
 *
 * Regression: whitespace-only input (spaces/tabs/newlines) left the button
 * ENABLED, because `disabled` was derived in three places that disagreed and
 * the `value.trim().length > 0` guard routed it down the "valid" branch.
 */
describe("submit button state tracks the editor content", () => {
  const type = (w: ReturnType<typeof mountWorkspace>, value: string) => {
    w.elements.textarea.value = value;
    w.elements.textarea.dispatchEvent(new Event("input"));
  };

  function mounted(send = vi.fn() as never) {
    const w = mountWorkspace(send);
    void new AnalyzeWorkspace(w.elements, { send });
    return w;
  }

  it("starts disabled on an empty editor", () => {
    const w = mounted();
    expect(w.elements.textarea.value).toBe("");
    expect(w.submit.disabled).toBe(true);
  });

  it.each([
    ["a single space", " "],
    ["several spaces", "     "],
    ["tabs", "\t\t"],
    ["newlines", "\n\n"],
    ["mixed whitespace", " \t\r\n "],
  ])("is disabled for %s", (_label, value) => {
    const w = mounted();
    type(w, value);
    expect(w.submit.disabled).toBe(true);
  });

  it("is enabled once real text is entered", () => {
    const w = mounted();
    type(w, "hello world");
    expect(w.submit.disabled).toBe(false);
  });

  it("is enabled for text that merely starts or ends with whitespace", () => {
    const w = mounted();
    type(w, "  \n hello world \n ");
    expect(w.submit.disabled).toBe(false);
  });

  it("re-enables when whitespace is replaced by text, and disables again", () => {
    const w = mounted();
    type(w, "real text");
    expect(w.submit.disabled).toBe(false);
    type(w, "   ");
    expect(w.submit.disabled).toBe(true);
    type(w, "more real text");
    expect(w.submit.disabled).toBe(false);
  });

  it("stays disabled while a request is pending", async () => {
    let resolveFn: (r: AnalyzeResponse) => void = () => {};
    const send = vi.fn(
      () => new Promise<AnalyzeResponse>((res) => (resolveFn = res)),
    );
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    type(w, "text to analyse");
    expect(w.submit.disabled).toBe(false);

    const p = ws.submit();
    expect(w.submit.disabled).toBe(true);
    expect(w.submit.getAttribute("aria-busy")).toBe("true");

    resolveFn(okResponse);
    await p;
    expect(w.submit.disabled).toBe(false);
    expect(w.submit.getAttribute("aria-busy")).toBe("false");
  });

  it("keeps the button disabled for whitespace-only text during a pending request", async () => {
    let resolveFn: (r: AnalyzeResponse) => void = () => {};
    const send = vi.fn(
      () => new Promise<AnalyzeResponse>((res) => (resolveFn = res)),
    );
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    type(w, "text to analyse");
    const p = ws.submit();
    expect(w.submit.disabled).toBe(true);
    resolveFn(okResponse);
    await p;
    // clearing the editor after a result disables it again
    type(w, "  ");
    expect(w.submit.disabled).toBe(true);
  });

  it("is enabled after a successful analysis when text remains", async () => {
    const send = vi.fn().mockResolvedValue(okResponse);
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    type(w, "a sentence worth analysing");
    const state = await ws.submit();
    expect(state.kind).toBe("done");
    expect(w.submit.disabled).toBe(false);
  });

  it("is enabled after a failed analysis when text remains", async () => {
    const send = vi.fn().mockRejectedValue(new ApiError("nope", "network"));
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    type(w, "a sentence worth analysing");
    const state = await ws.submit();
    expect(state.kind).toBe("failed");
    expect(w.submit.disabled).toBe(false); // retry must be possible
  });

  it("does not disable the button for a long-but-valid text", () => {
    const w = mounted();
    // 1,000 words is well past the 50-word gate but far under 30,000 chars
    type(w, "word ".repeat(1000).trim());
    expect(w.submit.disabled).toBe(false);
  });

  it("disables the button and warns when the text exceeds the character limit", () => {
    const w = mounted();
    type(w, "x".repeat(30_001));
    expect(w.submit.disabled).toBe(true);
    expect(w.validation.hidden).toBe(false);
    expect(w.validation.textContent).toContain("30,000");
  });

  it("enables the button again after trimming an over-long text back under the limit", () => {
    const w = mounted();
    type(w, "x".repeat(30_001));
    expect(w.submit.disabled).toBe(true);
    type(w, "x".repeat(100));
    expect(w.submit.disabled).toBe(false);
  });

  it("enables the button after 'Load an example' populates the editor", () => {
    const w = mounted();
    expect(w.submit.disabled).toBe(true);
    w.elements.example!.click();
    expect(w.elements.textarea.value.length).toBeGreaterThan(0);
    expect(w.submit.disabled).toBe(false);
  });

  it("disables the button after 'Clear' empties the editor", () => {
    const w = mounted();
    w.elements.example!.click();
    expect(w.submit.disabled).toBe(false);
    w.elements.clear!.click();
    expect(w.submit.disabled).toBe(true);
  });

  it("refuses to submit whitespace-only text without calling the API", async () => {
    const send = vi.fn();
    const w = mountWorkspace(send as never);
    const ws = new AnalyzeWorkspace(w.elements, { send: send as never });
    type(w, "   \n\t ");
    const state = await ws.submit();
    expect(send).not.toHaveBeenCalled();
    expect(state.kind).toBe("invalid");
  });
});
