"use client";

import { type FormEvent, type KeyboardEvent, useState } from "react";

const MIN_LENGTH = 3;

export function Composer({
  onSubmit,
  disabled,
}: {
  onSubmit: (question: string) => void;
  disabled: boolean;
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
      <p className="mt-2 hidden font-mono text-[11px] text-slate sm:block">
        Enter to ask · Shift + Enter for a new line
      </p>
    </form>
  );
}
