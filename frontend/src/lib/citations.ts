export const CITE_PREFIX = "#cite-";

const MARKER = /\[(\d+)\]/g;
// Fenced blocks (``` or ~~~) and inline code spans; markers inside them are code, not citations.
const CODE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

/**
 * Turn `[n]` citation markers into markdown links `[n](#cite-n)` so the markdown renderer
 * hands them to our citation component. Only numbers that exist in `valid` are linked, and
 * code is left untouched (`items[1]` in a snippet is not a citation).
 */
export function linkCitations(markdown: string, valid: Iterable<number>): string {
  const allowed = new Set(valid);
  return markdown
    .split(CODE)
    .map((part, i) =>
      // split() with a capturing group puts the code matches at odd indexes.
      i % 2 === 1
        ? part
        : part.replace(MARKER, (match, n: string) =>
            allowed.has(Number(n)) ? `[${n}](${CITE_PREFIX}${n})` : match,
          ),
    )
    .join("");
}

export function citationNumber(href: string | undefined): number | null {
  if (!href?.startsWith(CITE_PREFIX)) return null;
  const n = Number(href.slice(CITE_PREFIX.length));
  return Number.isInteger(n) ? n : null;
}
