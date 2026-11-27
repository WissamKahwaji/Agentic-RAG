"use client";

import { createContext, type ReactNode, use, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { useCitations } from "@/components/citation-context";
import type { Citation } from "@/lib/types";

const PREVIEW_CHARS = 320;

export function EvidenceCard({ id, citation }: { id: string; citation: Citation }) {
  const { active, hover, pin } = useCitations();
  const isActive = active === citation.number;
  const [expanded, setExpanded] = useState(false);
  const long = citation.content.length > PREVIEW_CHARS;
  const text =
    expanded || !long ? citation.content : `${citation.content.slice(0, PREVIEW_CHARS).trimEnd()}…`;

  return (
    <div
      id={id}
      onMouseEnter={() => hover(citation.number)}
      onMouseLeave={() => hover(null)}
      className={`scroll-mt-6 rounded-lg border bg-paper p-4 transition-shadow ${
        isActive ? "border-ink/30 shadow-[0_6px_20px_-10px_rgba(22,32,43,0.35)]" : "border-rule"
      }`}
    >
      <button
        type="button"
        onClick={() => pin(citation.number)}
        aria-pressed={isActive}
        className="flex w-full items-start gap-3 text-left"
      >
        <span
          className={`mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded px-1 font-mono text-xs font-medium transition-colors ${
            isActive ? "bg-highlighter text-ink" : "bg-cobalt/10 text-cobalt"
          }`}
        >
          {citation.number}
        </span>
        <span className="min-w-0">
          <span className="block text-sm leading-snug font-medium">{citation.title}</span>
          <span className="block truncate text-xs text-slate">{citation.location}</span>
        </span>
      </button>

      <PassageActive value={isActive}>
        <Passage text={text} markdown={citation.source.endsWith(".md")} active={isActive} />
      </PassageActive>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 text-xs font-medium text-cobalt hover:underline"
        >
          {expanded ? "Show less" : "Show full passage"}
        </button>
      )}

      <div className="mt-3 flex items-center gap-3 border-t border-rule pt-3">
        {citation.source.startsWith("http") ? (
          <a
            href={citation.source}
            target="_blank"
            rel="noopener noreferrer"
            className="min-w-0 flex-1 truncate font-mono text-[11px] text-cobalt hover:underline"
            title={citation.source}
          >
            {citation.source.replace(/^https?:\/\//, "")}
          </a>
        ) : (
          <span
            className="min-w-0 flex-1 truncate font-mono text-[11px] text-slate"
            title={citation.source}
          >
            {citation.source}
          </span>
        )}
        <Similarity score={citation.score} />
      </div>
    </div>
  );
}

// Whether the enclosing card is being inspected; read by the highlighter marks below.
const PassageActive = createContext(false);

function Mark({ children }: { children: ReactNode }) {
  return (
    <span className="highlight-sweep" data-active={use(PassageActive)}>
      {children}
    </span>
  );
}

// Module-level so react-markdown keeps the same component types between renders; otherwise
// every hover would remount the passage and restart the highlighter animation.
const PASSAGE_COMPONENTS: Components = {
  h1: ({ children }) => <p className="font-semibold"><Mark>{children}</Mark></p>,
  h2: ({ children }) => <p className="font-semibold"><Mark>{children}</Mark></p>,
  h3: ({ children }) => <p className="font-semibold"><Mark>{children}</Mark></p>,
  h4: ({ children }) => <p className="font-semibold"><Mark>{children}</Mark></p>,
  p: ({ children }) => <p><Mark>{children}</Mark></p>,
  li: ({ children }) => <li><Mark>{children}</Mark></li>,
  a: ({ children }) => <span className="underline decoration-rule">{children}</span>,
  img: () => null,
};

function Passage({ text, markdown, active }: { text: string; markdown: boolean; active: boolean }) {
  if (!markdown) {
    return (
      <p className="evidence-prose mt-3">
        <Mark>{text}</Mark>
      </p>
    );
  }
  return (
    <div className="evidence-prose mt-3" data-active={active}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={PASSAGE_COMPONENTS}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

function Similarity({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <span className="flex shrink-0 items-center gap-2" title="Relevance to the question, from the final ranking stage">
      <span className="h-1 w-12 overflow-hidden rounded-full bg-rule" aria-hidden>
        <span className="block h-full rounded-full bg-cobalt" style={{ width: `${pct}%` }} />
      </span>
      <span className="font-mono text-[11px] text-slate">{score.toFixed(2)}</span>
    </span>
  );
}
