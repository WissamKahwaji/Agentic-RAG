"use client";

import { type ComponentProps, type Ref, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CitationContext, type CitationState, useCitations } from "@/components/citation-context";
import { EvidenceCard } from "@/components/evidence-card";
import { citationNumber, linkCitations } from "@/lib/citations";
import type { Answer, TraceStep } from "@/lib/types";

export type TurnState = {
  id: string;
  question: string;
  status: "pending" | "done" | "error";
  answer?: Answer;
  error?: string;
};

export function Turn({ turn, ref }: { turn: TurnState; ref?: Ref<HTMLElement> }) {
  // `hovered` follows the pointer/focus; `pinned` sticks after a click until clicked again.
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const answer = turn.answer;
  const citations = useMemo(() => answer?.citations ?? [], [answer]);

  const state: CitationState = useMemo(
    () => ({
      citations,
      active: hovered ?? pinned,
      hover: setHovered,
      pin: (n: number) => {
        setPinned((current) => (current === n ? null : n));
        document
          .getElementById(evidenceId(turn.id, n))
          ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      },
    }),
    [citations, hovered, pinned, turn.id],
  );

  return (
    <CitationContext value={state}>
      <article ref={ref} className="scroll-mt-6 border-b border-rule py-10 last:border-b-0">
        <h2 className="max-w-3xl font-display text-2xl leading-tight font-medium tracking-tight sm:text-[1.9rem]">
          {turn.question}
        </h2>

        <div className="mt-6 grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,22rem)] md:gap-10">
          <div className="min-w-0">
            {turn.status === "pending" && <Pending />}
            {turn.status === "error" && <ErrorNotice message={turn.error} />}
            {answer && (
              <>
                <SourceBadge source={answer.source} />
                {answer.search_query && (
                  <p className="mb-3 font-mono text-xs text-slate">
                    Searched for: {answer.search_query}
                  </p>
                )}
                <AnswerBody answer={answer} />
                <GroundingNote answer={answer} />
                <Meta answer={answer} />
                <Trace answer={answer} />
              </>
            )}
          </div>

          {citations.length > 0 && (
            <aside aria-label="Evidence" className="min-w-0">
              <div className="md:sticky md:top-6">
                <p className="mb-3 font-mono text-xs tracking-wide text-slate uppercase">
                  Evidence · {citations.length} {citations.length === 1 ? "passage" : "passages"}
                </p>
                <ol className="flex flex-col gap-3">
                  {citations.map((c) => (
                    <li key={c.number}>
                      <EvidenceCard id={evidenceId(turn.id, c.number)} citation={c} />
                    </li>
                  ))}
                </ol>
              </div>
            </aside>
          )}
        </div>
      </article>
    </CitationContext>
  );
}

function evidenceId(turnId: string, n: number) {
  return `evidence-${turnId}-${n}`;
}

function AnswerBody({ answer }: { answer: Answer }) {
  const markdown = useMemo(
    () => linkCitations(answer.answer, answer.citations.map((c) => c.number)),
    [answer],
  );
  return (
    <div className="answer-prose max-w-[68ch]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={ANSWER_COMPONENTS}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}

function AnswerLink({ href, children }: ComponentProps<"a">) {
  const n = citationNumber(href);
  if (n === null) {
    return (
      <a href={href} className="text-cobalt underline underline-offset-2">
        {children}
      </a>
    );
  }
  return <CitationMarker n={n} />;
}

function CitationMarker({ n }: { n: number }) {
  const { citations, active, hover, pin } = useCitations();
  const citation = citations.find((c) => c.number === n);
  const isActive = active === n;
  return (
    <button
      type="button"
      aria-label={`Source ${n}: ${citation?.title ?? ""}, ${citation?.location ?? ""}`}
      aria-pressed={isActive}
      onMouseEnter={() => hover(n)}
      onMouseLeave={() => hover(null)}
      onFocus={() => hover(n)}
      onBlur={() => hover(null)}
      onClick={() => pin(n)}
      className={`mx-[0.1em] inline-flex h-[1.35em] min-w-[1.35em] -translate-y-[0.12em] items-center justify-center rounded px-1 align-middle font-mono text-[0.7em] font-medium transition-colors ${
        isActive ? "bg-highlighter text-ink" : "bg-cobalt/10 text-cobalt hover:bg-cobalt/20"
      }`}
    >
      {n}
    </button>
  );
}

// Module-level so the component types are stable across renders (see citation-context.tsx).
const ANSWER_COMPONENTS: Components = {
  a: ({ href, children }) => <AnswerLink href={href}>{children}</AnswerLink>,
};

function GroundingNote({ answer }: { answer: Answer }) {
  if (answer.source === "web" && answer.grounded) {
    return (
      <p className="mt-5 max-w-[68ch] border-l-2 border-cobalt pl-3 text-sm text-slate">
        Your documents didn&apos;t answer this, so it comes from web results. Web pages can be
        outdated or wrong: check the linked sources.
      </p>
    );
  }
  if (answer.grounded) return null;
  if (answer.source === "model") {
    return (
      <p className="mt-5 max-w-[68ch] border-l-2 border-rust pl-3 text-sm text-rust">
        Your documents didn&apos;t answer this, so it comes from the model&apos;s general knowledge,
        with no sources to check. Treat it as unverified.
      </p>
    );
  }
  const check = answer.self_check;
  if (check && !check.passed) {
    return (
      <div className="mt-5 max-w-[68ch] border-l-2 border-rust pl-3 text-sm text-rust">
        <p>The self-check couldn&apos;t match these claims to their sources. Verify them before relying on them:</p>
        <ul className="mt-2 list-disc pl-5">
          {check.unsupported_claims.map((claim) => (
            <li key={claim}>{claim}</li>
          ))}
        </ul>
      </div>
    );
  }
  if (answer.refused) {
    return (
      <p className="mt-5 max-w-[68ch] border-l-2 border-rule pl-3 text-sm text-slate">
        No passage in the indexed documents answers this. Add documents that cover it and run{" "}
        <code className="font-mono">make ingest</code>.
      </p>
    );
  }
  return (
    <p className="mt-5 max-w-[68ch] border-l-2 border-rust pl-3 text-sm text-rust">
      This answer cites no passages, so it can&apos;t be checked. Treat it as unverified.
    </p>
  );
}

function Meta({ answer }: { answer: Answer }) {
  const seconds = (answer.timings.generation_ms / 1000).toFixed(1);
  return (
    <p className="mt-6 font-mono text-[11px] text-slate">
      {answer.model} · {answer.prompt} · retrieved in {answer.timings.retrieval_ms} ms · answered
      in {seconds} s
    </p>
  );
}

const SOURCE_LABELS: Record<Answer["source"], { icon: string; label: string; tone: string }> = {
  documents: { icon: "📄", label: "From your documents", tone: "text-ink" },
  web: { icon: "🌐", label: "From the web", tone: "text-cobalt" },
  model: { icon: "🧠", label: "From model knowledge · no sources", tone: "text-rust" },
  none: { icon: "∅", label: "No answer found", tone: "text-slate" },
};

function SourceBadge({ source }: { source: Answer["source"] }) {
  const s = SOURCE_LABELS[source];
  return (
    <p className={`mb-3 inline-flex items-center gap-1.5 font-mono text-xs ${s.tone}`}>
      <span aria-hidden>{s.icon}</span>
      {s.label}
    </p>
  );
}

const STEP_LABELS: Record<TraceStep["step"], string> = {
  condense: "Rewrite follow-up",
  route: "Choose another source",
  web_search: "Search the web",
  answer_direct: "Answer from model knowledge",
  retrieve: "Search",
  grade: "Grade passages",
  reformulate: "Try another query",
  generate: "Write answer",
  self_check: "Check claims",
  refuse: "Decline",
};

function Trace({ answer }: { answer: Answer }) {
  if (answer.trace.length === 0) return null;
  return (
    <details className="group mt-3 max-w-[68ch]">
      <summary className="cursor-pointer font-mono text-[11px] text-slate select-none hover:text-ink">
        How this answer was made · {answer.trace.length} steps
      </summary>
      <ol className="mt-3 border-l border-rule pl-4">
        {answer.trace.map((s, i) => (
          <li key={i} className="relative pb-3 last:pb-0">
            <span
              aria-hidden
              className="absolute top-[0.45rem] -left-[1.27rem] h-2 w-2 rounded-full border border-slate bg-fog"
            />
            <p className="text-sm">
              <span className="font-medium">{STEP_LABELS[s.step]}</span>
              <span className="ml-2 font-mono text-[11px] text-slate">{s.ms} ms</span>
            </p>
            <p className="text-sm text-slate">{s.detail}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function Pending() {
  return (
    <div role="status" className="max-w-[68ch]">
      <p className="font-mono text-xs text-slate">Finding passages and drafting an answer…</p>
      <div className="mt-4 flex flex-col gap-3" aria-hidden>
        {["w-full", "w-11/12", "w-4/6"].map((w) => (
          <div
            key={w}
            className={`h-3 animate-pulse rounded bg-rule motion-reduce:animate-none ${w}`}
          />
        ))}
      </div>
    </div>
  );
}

function ErrorNotice({ message }: { message?: string }) {
  return (
    <p role="alert" className="max-w-[68ch] border-l-2 border-rust pl-3 text-rust">
      {message ?? "Something went wrong."}
    </p>
  );
}
