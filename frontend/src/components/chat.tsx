"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Composer } from "@/components/composer";
import { Turn, type TurnState } from "@/components/turn";
import type { Answer, ApiError, DocumentSummary } from "@/lib/types";

const EXAMPLES = [
  "How do I create an HNSW index, and what do m and ef_construction control?",
  "What's the difference between RAG-Sequence and RAG-Token?",
  "When should a FastAPI path operation use async def instead of def?",
];

type Corpus = { documents: number; passages: number } | "unavailable" | null;

export function Chat() {
  const [turns, setTurns] = useState<TurnState[]>([]);
  const [corpus, setCorpus] = useState<Corpus>(null);
  const lastTurnRef = useRef<HTMLElement>(null);
  const pending = turns.some((t) => t.status === "pending");

  useEffect(() => {
    fetch("/api/documents")
      .then((res) => (res.ok ? (res.json() as Promise<DocumentSummary[]>) : Promise.reject()))
      .then((docs) =>
        setCorpus({
          documents: docs.length,
          passages: docs.reduce((sum, d) => sum + d.chunk_count, 0),
        }),
      )
      .catch(() => setCorpus("unavailable"));
  }, []);

  useEffect(() => {
    lastTurnRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [turns.length]);

  const ask = useCallback(async (question: string) => {
    const id = crypto.randomUUID();
    setTurns((prev) => [...prev, { id, question, status: "pending" }]);

    const update = (patch: Partial<TurnState>) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = (await res.json()) as Answer | ApiError;
      if (!res.ok || "error" in body) {
        update({ status: "error", error: "error" in body ? body.error : "Request failed." });
      } else {
        update({ status: "done", answer: body });
      }
    } catch {
      update({ status: "error", error: "The page couldn't reach its server. Reload and retry." });
    }
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-5 sm:px-8">
      <header className="flex items-baseline justify-between gap-4 border-b border-rule py-5">
        <span className="shrink-0 font-display text-lg font-bold tracking-tight whitespace-nowrap">
          Agentic RAG
        </span>
        <CorpusStatus corpus={corpus} />
      </header>

      <main className="flex-1 pb-10">
        {turns.length === 0 ? (
          <Intro onPick={ask} disabled={pending} />
        ) : (
          turns.map((turn, i) => (
            <Turn key={turn.id} turn={turn} ref={i === turns.length - 1 ? lastTurnRef : null} />
          ))
        )}
      </main>

      <Composer onSubmit={ask} disabled={pending} />
    </div>
  );
}

function CorpusStatus({ corpus }: { corpus: Corpus }) {
  if (corpus === null) return <span className="font-mono text-xs text-slate">Checking index…</span>;
  if (corpus === "unavailable") {
    return <span className="font-mono text-xs text-rust">API offline · run make dev</span>;
  }
  return (
    <span className="text-right font-mono text-xs text-slate">
      {corpus.documents} documents · {corpus.passages} passages
    </span>
  );
}

function Intro({ onPick, disabled }: { onPick: (q: string) => void; disabled: boolean }) {
  return (
    <section className="max-w-3xl pt-16 pb-10 sm:pt-24">
      <h1 className="font-display text-5xl leading-[1.02] font-bold tracking-tight sm:text-7xl">
        Ask your documents.
        <br />
        <span className="highlight-sweep" data-active="true">
          Check every answer.
        </span>
      </h1>
      <p className="mt-6 max-w-xl text-lg text-slate">
        Answers come only from the indexed documents. Each claim carries a numbered marker; hover
        it to see the passage it came from.
      </p>
      <div className="mt-10">
        <p className="font-mono text-xs tracking-wide text-slate uppercase">Try one</p>
        <ul className="mt-3 flex flex-col gap-2">
          {EXAMPLES.map((q) => (
            <li key={q}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => onPick(q)}
                className="group text-left text-[1.05rem] text-ink decoration-cobalt/40 underline-offset-4 hover:text-cobalt hover:underline disabled:opacity-50"
              >
                <span aria-hidden className="mr-2 text-cobalt">
                  →
                </span>
                {q}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
