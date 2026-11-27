"use client";

import { type FormEvent, type KeyboardEvent, useState } from "react";

import type { Knowledge } from "@/lib/types";

const MIN_LENGTH = 3;

const KNOWLEDGE_OPTIONS: { value: Knowledge; label: string; hint: string }[] = [
  { value: "documents", label: "Documents only", hint: "Answers only from your documents" },
  {
    value: "hybrid",
    label: "Documents + web",
    hint: "Documents first; if they can't answer, the web or the model's own knowledge (labelled)",
  },
];

export function Composer({
  onSubmit,
  disabled,
  knowledge,
  onKnowledgeChange,
}: {
  onSubmit: (question: string) => void;
  disabled: boolean;
  knowledge: Knowledge;
  onKnowledgeChange: (knowledge: Knowledge) => void;
}) {
  const [value, setValue] = useState("");
  const question = value.trim();
  const canSubmit = !disabled && question.length >= MIN_LENGTH;

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!canSubmit) return;
    onSubmit(question);
    setValue("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form
      onSubmit={submit}
      className="sticky bottom-0 -mx-5 bg-gradient-to-t from-fog from-70% to-transparent px-5 pt-6 pb-5 sm:-mx-8 sm:px-8"
    >
      <div className="flex items-end gap-3 rounded-xl border border-rule bg-paper p-2 pl-4 shadow-[0_1px_0_rgba(22,32,43,0.04),0_8px_24px_-12px_rgba(22,32,43,0.18)] focus-within:border-cobalt">
        <label htmlFor="question" className="sr-only">
          Question
        </label>
        <textarea
          id="question"
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about pgvector, FastAPI or the RAG paper…"
          className="field-sizing-content max-h-48 min-h-10 flex-1 resize-none bg-transparent py-2 text-base outline-none placeholder:text-slate/70 focus-visible:outline-none"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="h-10 rounded-lg bg-ink px-5 text-sm font-medium text-paper transition-colors hover:bg-cobalt disabled:bg-rule disabled:text-slate"
        >
          {disabled ? "Answering…" : "Ask"}
        </button>
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <div
          role="radiogroup"
          aria-label="Knowledge sources"
          className="inline-flex rounded-md border border-rule bg-paper p-0.5"
        >
          {KNOWLEDGE_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={knowledge === o.value}
              title={o.hint}
              onClick={() => onKnowledgeChange(o.value)}
              className={`rounded px-2.5 py-1 font-mono text-[11px] transition-colors ${
                knowledge === o.value ? "bg-ink text-paper" : "text-slate hover:text-ink"
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
        <p className="hidden font-mono text-[11px] text-slate sm:block">
          Enter to ask · Shift + Enter for a new line
        </p>
      </div>
    </form>
  );
}
