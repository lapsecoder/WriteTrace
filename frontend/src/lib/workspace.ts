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
import { MAX_TEXT_CHARS } from "./types";

export interface WorkspaceElements {
  form: HTMLFormElement;
  textarea: HTMLTextAreaElement;
  submit: HTMLButtonElement;
  counter: HTMLElement;
  validation: HTMLElement;
  status: HTMLElement;
  results: HTMLElement;
  error: HTMLElement;
  /**
   * Message/detail nodes INSIDE `error`.
   *
   * The error box is static markup authored by the page; the controller only
   * fills in its text. The controller must never clear `error`'s children —
   * doing so detaches these nodes, after which any text written into them is
   * invisible because they are no longer in the document. That bug left users
   * looking at an empty error box (defect D1, found in the Phase 6C audit).
   */
  errorMessage: HTMLElement | null;
  errorDetail: HTMLElement | null;
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

/**
 * Fallback copy per failure kind. These are authored here so the UI never has
 * to render an untrusted string to stay useful.
 */
const SAFE_FALLBACK: Record<string, string> = {
  network:
    "Could not reach the WriteTrace backend. Check that the API is running " +
    "and try again.",
  validation: "The backend rejected this text as invalid.",
  malformed:
    "The backend returned a response that could not be read as a result.",
  backend: "The backend could not complete the analysis. Please try again.",
};

/** Longest message rendered in the error box before it is truncated. */
const MAX_MESSAGE_CHARS = 300;

/**
 * Keep untrusted error text out of the page.
 *
 * Transport failures can surface text that was never written for humans — an
 * HTML proxy page, a framework traceback, a filesystem path. Anything matching
 * these shapes is discarded in favour of safe, kind-specific copy, so no stack
 * trace, internal path, or echoed input can ever reach the screen.
 */
function isUnsafe(text: string): boolean {
  return (
    /traceback \(most recent call last\)/i.test(text) ||
    /\bfile\s+"/i.test(text) ||
    /at\s+[\w.$<>]+\s*\([^)]*:\d+:\d+\)/.test(text) ||      // JS stack frame
    /\bline\s+\d+,?\s+in\s+\w/.test(text) ||                  // Py stack frame
    /[A-Za-z]:[\\/]/.test(text) ||                           // C:\... path
    /(^|\s)\/(?:usr|home|var|etc|tmp|opt|root|app)\//.test(text) ||
    /site-packages|node_modules|\.py["'\s]?\d*\b/.test(text) ||
    /<[a-z!/][\s\S]*>/i.test(text)                            // markup/HTML
  );
}

/** Trim to a safe, bounded, human-readable string, or return null if unsafe. */
function safeText(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (text.length === 0) return null;
  if (isUnsafe(text)) return null;
  return text.length > MAX_MESSAGE_CHARS
    ? `${text.slice(0, MAX_MESSAGE_CHARS).trimEnd()}…`
    : text;
}

function describeApiError(err: unknown): { message: string; detail: string | null } {
  const apiErr = err as Partial<ApiError>;
  if (apiErr && typeof apiErr.kind === "string") {
    const detail =
      apiErr.kind === "validation" && Array.isArray(apiErr.details)
        ? apiErr.details
            .map((d) => safeText(`${d.field ? `${d.field}: ` : ""}${d.message}`))
            .filter((d): d is string => d !== null)
            .join(" · ")
        : null;
    const base =
      safeText(apiErr.message) ?? SAFE_FALLBACK[apiErr.kind] ??
      "The request could not be completed.";
    return { message: base, detail: detail || null };
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

  /**
   * Single source of truth for the submit button's enabled state.
   *
   * This used to be derived independently in `syncText`, `submit` and `render`,
   * which disagreed: the `value.trim().length > 0` guard in `syncText` sent
   * whitespace-only input down the "valid" branch and left Analyse enabled, so
   * the button looked ready but could only ever produce a rejection.
   *
   * The button is disabled whenever the request cannot succeed:
   *   - a request is already pending (no duplicate submissions),
   *   - the editor holds no non-whitespace character,
   *   - the text exceeds the backend's limit.
   */
  private updateSubmitEnabled(): void {
    const value = this.el.textarea.value;
    const hasContent = textStats(value).nonWhitespace > 0;
    const withinLimit = validateText(value).ok;
    this.el.submit.disabled =
      this.state.kind === "pending" || !hasContent || !withinLimit;
  }

  /** Update counters + soft validation as the user types. */
  syncText(): void {
    const stats = textStats(this.el.textarea.value);
    this.el.counter.textContent = `${stats.words.toLocaleString()} words · ${stats.characters.toLocaleString()} characters`;

    const overLimit = stats.characters > MAX_TEXT_CHARS;
    this.el.counter.classList.toggle("counter--over", overLimit);

    // Keep the button in step with the text even while a result is on screen
    // or a request is in flight; otherwise clearing the editor after a result
    // would leave Analyse enabled.
    this.updateSubmitEnabled();

    if (this.state.kind === "pending" || this.state.kind === "done") return;

    const check = validateText(this.el.textarea.value);
    if (!check.ok && this.el.textarea.value.trim().length > 0) {
      this.el.validation.textContent = check.message ?? "";
      this.el.validation.hidden = false;
      this.setState({ kind: "invalid", message: check.message ?? "" });
    } else {
      this.el.validation.textContent = "";
      this.el.validation.hidden = true;
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

  /**
   * Hide the error box and empty its message fields.
   *
   * Only the message/detail TEXT is reset. The box's children are static page
   * markup (heading, reassurance that the text is still in the editor) and must
   * stay attached; clearing `error.innerHTML`/`textContent` detaches them and
   * permanently breaks every later write (defect D1).
   */
  private resetError(): void {
    this.el.error.hidden = true;
    if (this.el.errorMessage) this.el.errorMessage.textContent = "";
    if (this.el.errorDetail) {
      this.el.errorDetail.textContent = "";
      this.el.errorDetail.hidden = true;
    }
  }

  /** Show the error box with a message and an optional detail line. */
  private showError(message: string, detail: string | null): void {
    this.el.error.hidden = false;
    if (this.el.errorMessage) this.el.errorMessage.textContent = message;
    if (this.el.errorDetail) {
      this.el.errorDetail.textContent = detail ?? "";
      this.el.errorDetail.hidden = !detail;
    }
  }

  private render(): void {
    const s = this.state;
    this.el.submit.setAttribute("aria-busy", s.kind === "pending" ? "true" : "false");
    this.el.textarea.readOnly = s.kind === "pending";
    this.updateSubmitEnabled();

    this.el.status.dataset.state = s.kind;
    switch (s.kind) {
      case "idle":
      case "invalid":
        this.el.status.textContent = "";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.resetError();
        break;
      case "pending":
        this.el.status.textContent = "Reading the text…";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.resetError();
        break;
      case "done":
        this.el.status.textContent = "";
        this.resetError();
        this.el.results.hidden = false;
        this.el.results.innerHTML = "";
        break;
      case "failed":
        this.el.status.textContent = "";
        this.el.results.hidden = true;
        this.el.results.innerHTML = "";
        this.showError(s.message, s.detail);
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