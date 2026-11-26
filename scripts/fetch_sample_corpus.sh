#!/usr/bin/env bash
# Download a small, public sample corpus into data/raw/ (git-ignored, not redistributed).
#   - pgvector README                      (PostgreSQL License)
#   - selected FastAPI documentation pages (MIT)
#   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", Lewis et al. 2020 (arXiv)
set -euo pipefail

DEST="${1:-data/raw}"
FASTAPI="https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs"

fetch() {  # fetch <url> <relative-output-path>
  local out="$DEST/$2"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then echo "exists   $2"; return; fi
  curl --fail --silent --show-error --location --max-time 60 -o "$out" "$1"
  echo "fetched  $2"
}

fetch "https://raw.githubusercontent.com/pgvector/pgvector/master/README.md" "pgvector/README.md"
for page in async.md tutorial/first-steps.md tutorial/path-params.md tutorial/query-params.md \
            tutorial/body.md tutorial/dependencies/index.md tutorial/bigger-applications.md \
            deployment/concepts.md; do
  fetch "$FASTAPI/$page" "fastapi/$page"
done
fetch "https://arxiv.org/pdf/2005.11401" "papers/rag-lewis-2020.pdf"
