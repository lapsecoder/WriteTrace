/**
 * The analysis workspace controller (the only client-side JavaScript that runs).
 *
 * Design notes
 * ------------
 * * Framework-free on purpose: the editing + results flow needs state and
 *   DOM updates, not a component runtime. A ~300-line controller is smaller and
 *   easier to test than a hydrated UI framework.
 * * Privacy: the submitted text lives in an in-memory string and in the
 *   textarea's own value. It is never written to localStorage/sessionStorage,
 *   never logged, and never placed in a URL.
 * * Honesty: results are rendered only from a real API response. Loading,
 *   error and empty states are distinct, and nothing is simulated.
 */
import { analyze } from "./api";
import type { ApiError } from "./types";
import type { AnalyzeResponse } from "./types";
import { textStats, validateText } from "./validate";

export interface WorkspaceElements {
  form: HTMLFormElement;
  textarea: HTMLTextAreaElement;
  submit: HTMLButtonElement;
  counter: HTMLElement;
  validation: HTMLElement;
  status: HTMLElement;
  results: HTMLElement;
  error: HTMLElement;
  example: HTMLButtonElement | null;
  clear: HTMLButtonElement | null;
}

export type WorkspaceState =
  | { kind: "idle" }
  | { kind: "invalid"; message: string }
  | { kind: "pending" }
  | { kind: "done"; result: AnalyzeResponse }
  | { kind: "failed"; message: string; detail: string | null };

/** Injected so tests can drive the controller without touching the network. */
export interface WorkspaceDeps {
  send?: typeof analyze;
}

const MAX_SIGNALS = 8;

function describeApiError(err: unknown): { message: string; detail: string | null } {
  const apiErr = err as Partial<ApiError>;
  if (apiErr && typeof apiErr.kind === "string") {
    const detail =
      apiErr.kind === "validation" && Array.isArray(apiErr.details)
        ? apiErr.details
            .map((d) => (d.field ? `${d.field}: ${d.message}` : d.message))
            .join(" · ")
        : null;
    const base =
      typeof apiErr.message === "string" && apiErr.message.length > 0
        ? apiErr.message
        : "The request could not be completed.";
    return { message: base, detail };
  }
  return {
    message: "Something went wrong while analysing this text.",
    detail: null,
  };
}

export class AnalyzeWorkspace {
  private readonly el: WorkspaceElements;
  private readonly send: typeof analyze;
  private state: WorkspaceState = { kind: "idle" };
  private inFlight: AbortController | null = null;

  constructor(elements: WorkspaceElements, deps: WorkspaceDeps = {}) {
    this.el = elements;
    this.send = deps.send ?? analyze;
    this.bind();
    this.syncText();
    this.render();
  }

  /** Current editor content (kept in the DOM; never persisted elsewhere). */
  get text(): string {
    return this.el.textarea.value;
  }

  get currentState(): WorkspaceState {
    return this.state;
  }

  private bind(): void {
    this.el.textarea.addEventListener("input", () => this.syncText());
    this.el.form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.submit();
    });
    this.el.example?.addEventListener("click", () => {
      this.el.textarea.value = EXAMPLE_TEXT;
      this.el.textarea.focus();
      this.syncText();
    });
    this.el.clear?.addEventListener("click", () => {
      this.el.textarea.value = "";
      this.el.textarea.focus();
      this.setState({ kind: "idle" });
      this.syncText();
    });
  }

  /** Update counters + soft validation as the user types. */
  syncText(): void {
    const stats = textStats(this.el.textarea.value);
    this.el.counter.textContent = `${stats.words.toLocaleString()} words · ${stats.characters.toLocaleString()} characters`;

    const overLimit = stats.characters > 30_000;
    this.el.counter.classList.toggle("counter--over", overLimit);

    if (this.state.kind === "pending" || this.state.kind === "done") return;

    const check = validateText(this.el.textarea.value);
    if (!check.ok && this.el.textarea.value.trim().length > 0) {
      this.el.validation.textContent = check.message ?? "";
      this.el.validation.hidden = false;
      this.el.submit.disabled = true;
      this.setState({ kind: "invalid", message: check.message ?? "" });
    } else {
      this.el.validation.textContent = "";
      this.el.validation.hidden = true;
      this.el.submit.disabled = false;
      this.setState({ kind: "idle" });
    }
  }

  /** Submit the current text; refuses duplicates while a request is pending. */
  async submit(): Promise<WorkspaceState> {
    if (this.state.kind === "pending") return this.state;

    const check = validateText(this.el.textarea.value);
    if (!check.ok) {
      this.el.validation.textContent = check.message ?? "";
      this.el.validation.hidden = false;
      this.el.submit.disabled = true;
      this.setState({ kind: "invalid", message: check.message ?? "" });
      return this.state;
    }

    const controller = new AbortController();
    this.inFlight = controller;
    // The text is captured verbatim; it is never trimmed or rewritten.
    const submitted = this.el.textarea.value;
    this.setState({ kind: "pending" });

    try {
      const result = await this.send(
        {
          text: submitted,
          include_signals: true,
          max_signals: MAX_SIGNALS,
        },
        controller.signal,
      );
      this.setState({ kind: "done", result });
      return this.state;
    } catch (err) {
      const { message, detail } = describeApiError(err);
      this.setState({ kind: "failed", message, detail });
      return this.state;
    } finally {
      this.inFlight = null;
      // The editor keeps the user's text in every outcome (success included).
      if (this.el.textarea.value !== submitted) {
        this.el.textarea.value = submitted;
      }
    }
  }

  /** Abort any in-flight request (used on teardown/navigation). */
  abort(): void {
    this.inFlight?.abort();
    this.inFlight = null;
  }

  private setState(next: WorkspaceState): void {
    this.state = next;
    this.render();
  }

  private render(): void {
    const s = this.state;
    this.el.submit.disabled = s.kind === "pending";
    this.el.submit.setAttribute("aria-busy", s.kind === "pending" ? "true" : "false");
    this.el.textarea.readOnly = s.kind === "pending";

    this.el.status.dataset.state = s.kind;
    switch (s.kind) {
      case "idle":
        this.el.status.textContent = "";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.el.error.hidden = true;
        this.el.error.innerHTML = "";
        break;
      case "invalid":
        this.el.status.textContent = "";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.el.error.hidden = true;
        this.el.error.innerHTML = "";
        break;
      case "pending":
        this.el.status.textContent = "Reading the text…";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.el.error.hidden = true;
        this.el.error.innerHTML = "";
        break;
      case "done":
        this.el.status.textContent = "";
        this.el.error.hidden = true;
        this.el.error.innerHTML = "";
        this.el.results.hidden = false;
        this.el.results.innerHTML = "";
        break;
      case "failed":
        this.el.status.textContent = "";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.el.error.hidden = false;
        this.el.error.textContent = "";
        break;
    }
  }

  /** Exposed for tests/components: inject a rendered results node. */
  mountResults(node: Node): void {
    this.el.results.innerHTML = "";
    this.el.results.appendChild(node);
    this.el.results.hidden = false;
  }
}

/** A neutral sample text (human-written register) offered in the UI. */
export const EXAMPLE_TEXT =
  "So we ended up going to that new cafe near the station, the one everyone " +
  "keeps talking about. Honestly the coffee was fine, nothing special, but " +
  "the cake was unreal - I had this lemon thing with poppy seeds and now I " +
  "want to go back every weekend. Anyway we sat there for two hours just " +
  "catching up, you know how it is. Sarah brought her dog, which was a whole " +
  "situation on its own, but whatever, it was really nice to just sit and not " +
  "think about work for a while. We should go together sometime, I think " +
  "you would like it. " +
  "The committee, meanwhile, announced that the renovation would finish in " +
  "the spring. It cost roughly 4.2 million euros, officials said, and will " +
  "stay open seven days a week from eight in the morning until ten at night. " +
  "Furthermore, the accessibility improvements include automatic doors on " +
  "every floor and a dedicated reading room for children. In conclusion, " +
  "residents will be able to borrow materials through the existing membership " +
  "system without any additional registration fees being required.";