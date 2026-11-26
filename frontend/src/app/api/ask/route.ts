import { forward } from "@/lib/backend";

export async function POST(request: Request) {
  const body = await request.text();
  return forward("/ask", { method: "POST", body });
}
