"use client";

import { createContext, use } from "react";

import type { Citation } from "@/lib/types";

/**
 * Inspection state for one answer, shared by its citation markers and evidence cards.
 *
 * It lives in context rather than in closures because react-markdown treats the `components`
 * map as component *types*: building that map inside a render creates new types every time, so
 * React would unmount and remount every marker on each hover and drop in-flight clicks.
 */
export type CitationState = {
  citations: Citation[];
  active: number | null;
  hover: (n: number | null) => void;
  pin: (n: number) => void;
};

export const CitationContext = createContext<CitationState | null>(null);

export function useCitations(): CitationState {
  const value = use(CitationContext);
  if (!value) throw new Error("useCitations must be used inside <CitationContext>");
  return value;
}
