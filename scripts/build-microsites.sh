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

# Add built microsite URLs to Jekyll's sitemap.xml (they are excluded from Jekyll's own build).
SITEMAP="$DEST/sitemap.xml"
if [ -f "$SITEMAP" ]; then
  SITE_URL="$(grep -m1 '^url:' "$ROOT/_config.yml" | sed -E 's/^url:[[:space:]]*["'\'']?([^"'\''#]+)["'\'']?.*/\1/' | sed 's|/*$||')"
  if [ -n "$SITE_URL" ]; then
    TODAY="$(date -u +%Y-%m-%d)"
    ADDED=0
    while IFS= read -r pkg; do
      dir="$(dirname "$pkg" | sed 's|^\./||')"
      loc="${SITE_URL}/${dir}/"
      if ! grep -q "<loc>${loc}</loc>" "$SITEMAP"; then
        ENTRY="<url><loc>${loc}</loc><lastmod>${TODAY}</lastmod></url>"
        node -e '
const fs = require("node:fs");
const [p, entry] = process.argv.slice(1);
const txt = fs.readFileSync(p, "utf8");
fs.writeFileSync(p, txt.replace("</urlset>", entry + "\n</urlset>"));
' "$SITEMAP" "$ENTRY"
        ADDED=$((ADDED + 1))
      fi
    done < <(find . -mindepth 2 -maxdepth 2 -name package.json \
               -not -path '*/node_modules/*' -not -path './shared/*' -not -path './_site/*' | sort)
    echo "Added $ADDED microsite URL(s) to $SITEMAP."
  fi
fi
