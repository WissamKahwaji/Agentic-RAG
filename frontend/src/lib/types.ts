// Mirrors the backend's Pydantic response models (backend/src/agentic_rag/answering.py).

export type Citation = {
  number: number;
  chunk_id: string;
  source: string;
  title: string;
  location: string;
  page: number | null;
  section: string | null;
  content: string;
  score: number;
};

export type Answer = {
  answer: string;
  grounded: boolean;
  refused: boolean;
  citations: Citation[];
  model: string;
  prompt: string;
  timings: { retrieval_ms: number; generation_ms: number };
};

export type DocumentSummary = {
  id: string;
  source: string;
  title: string;
  media_type: string;
  chunk_count: number;
  updated_at: string;
};

export type ApiError = { error: string };
