import "server-only";

// Server-side only: the browser talks to our Route Handlers, which forward to the API. This
// keeps the backend URL private, avoids CORS, and gives one place to translate failures.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function forward(path: string, init?: RequestInit): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { error: "The API isn't reachable. Start it with `make dev` and try again." },
      { status: 502 },
    );
  }

  if (upstream.ok) {
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  }
  return Response.json({ error: await describeFailure(upstream) }, { status: upstream.status });
}

async function describeFailure(res: Response): Promise<string> {
  if (res.status === 503) {
    return "The model server isn't responding. Check that Ollama is running, then try again.";
  }
  if (res.status === 422) {
    return "Questions need between 3 and 2000 characters.";
  }
  const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
  return typeof body?.detail === "string" ? body.detail : `The API returned ${res.status}.`;
}
