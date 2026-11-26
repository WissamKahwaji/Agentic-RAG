import { connection } from "next/server";

import { forward } from "@/lib/backend";

export async function GET() {
  await connection(); // always fetch fresh; the corpus changes when documents are ingested
  return forward("/documents");
}
