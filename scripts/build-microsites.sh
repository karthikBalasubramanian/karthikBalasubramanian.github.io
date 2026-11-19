#!/usr/bin/env bash
# Build every React/Vite microsite (any top-level folder with a package.json) and copy its dist/
# into a destination folder, normally the Jekyll output (_site).
#
#   scripts/build-microsites.sh [DEST]        # DEST defaults to ./_site
#
# Used by .github/workflows/deploy.yml, Dockerfile.local and the Makefile so all three do the same thing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/_site}"
mkdir -p "$DEST"
cd "$ROOT"

found=0
while IFS= read -r pkg; do
  dir="$(dirname "$pkg" | sed 's|^\./||')"
  found=$((found + 1))
  echo "=========================================="
  echo "Building microsite: $dir"
  echo "=========================================="
  (
    cd "$dir"
    if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi
    npm run build
  )
  if [ ! -d "$dir/dist" ]; then
    echo "ERROR: no dist/ directory for $dir after build" >&2
    exit 1
  fi
  rm -rf "${DEST:?}/$dir"
  mkdir -p "$DEST/$dir"
  cp -r "$dir/dist/." "$DEST/$dir/"
  touch "$DEST/$dir/.nojekyll"
  echo "Copied $dir/dist -> $DEST/$dir"
done < <(find . -mindepth 2 -maxdepth 2 -name package.json \
           -not -path '*/node_modules/*' -not -path './shared/*' -not -path './_site/*' | sort)

echo "Built $found microsite(s)."
[ "$found" -gt 0 ]
