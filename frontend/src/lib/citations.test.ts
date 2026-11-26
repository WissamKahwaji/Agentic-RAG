import { describe, expect, it } from "vitest";

import { citationNumber, linkCitations } from "./citations";

describe("linkCitations", () => {
  it("links markers that refer to known citations", () => {
    expect(linkCitations("HNSW is a graph [1][3].", [1, 3])).toBe(
      "HNSW is a graph [1](#cite-1)[3](#cite-3).",
    );
  });

  it("leaves unknown markers as plain text", () => {
    expect(linkCitations("See [2].", [1])).toBe("See [2].");
  });

  it("does not touch fenced code blocks", () => {
    const md = "Use this [1]:\n```python\nx = items[1]\n```\nDone [1].";
    expect(linkCitations(md, [1])).toBe(
      "Use this [1](#cite-1):\n```python\nx = items[1]\n```\nDone [1](#cite-1).",
    );
  });

  it("does not touch inline code", () => {
    expect(linkCitations("Index with `arr[1]` [1].", [1])).toBe(
      "Index with `arr[1]` [1](#cite-1).",
    );
  });

  it("handles an unterminated fence without linking inside it", () => {
    expect(linkCitations("Text [1]\n```\nlist[1]", [1])).toBe("Text [1](#cite-1)\n```\nlist[1]");
  });
});

describe("citationNumber", () => {
  it("parses citation hrefs and rejects others", () => {
    expect(citationNumber("#cite-4")).toBe(4);
    expect(citationNumber("https://example.com")).toBeNull();
    expect(citationNumber(undefined)).toBeNull();
  });
});
