Karthik Balasubramanian: personal site
=====================================

Source for <https://karthikbalasubramanian.github.io/>, a Jekyll site using the
[Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme (`jekyll-theme-chirpy` gem),
plus four React/Vite/Tailwind finance calculators served under their own paths and listed on `/tools/`.

## Layout

| Path | What |
|---|---|
| `_config.yml` | Site config. Production URL `https://karthikbalasubramanian.github.io` |
| `_config.local.yml` | Local-only override (`url: http://localhost:4000`), used by Docker / `make site` |
| `_posts/` | Blog posts. Older posts pin `permalink:` so their `/ml/YYYY/MM/DD/slug.html` URLs never change; new posts default to `/posts/:title/` |
| `_tabs/` | Sidebar pages: Blog, Projects, Tools, About, Categories, Tags |
| `_includes/home-hero.html`, `_layouts/home.html` | Home page hero (the layout is a copy of Chirpy's with one include added; re-sync on Chirpy upgrades) |
| `_data/featured.yml`, `_data/tools.yml` | Home page cards and Tools page cards |
| `assets/css/jekyll-theme-chirpy.scss` | Theme entry point plus site styles |
| `shared/site-chrome/` | Single source for the microsites' shared design tokens, header and footer |
| `scripts/build-microsites.sh` | Builds every microsite and copies its `dist/` into `_site/` |
| `_plugins/math-protect.rb` | Math in posts: write `$...$` inline and `$$...$$` display (on its own line). The plugin shields the TeX from kramdown and turns on MathJax (`math: true`) for any post that contains math |

### Microsites

`child-financial-investment-planner`, `paycheck-tax-investment-allocator`,
`rent-vs-buy-house-poor-calculator`, `rsu-hold-vs-sell-simulator`. Each is a standalone Vite app with
`base: '/<folder>/'`. Any new top-level folder with a `package.json` is picked up automatically.

They share the blog's look through `shared/site-chrome`:

- `tokens.*.css` remaps Tailwind's palette to Chirpy's greys through CSS variables, so the apps follow the blog's
  light/dark mode (localStorage key `theme`, attribute `html[data-bs-theme]`, which is what Chirpy uses).
- `SiteChrome.tsx` + `chrome.css` render the shared header (Home, Blog, Projects, Tools, About, mode toggle)
  and footer around each `<App />`.

Edit files in `shared/site-chrome/`, then run `make tokens` (`node shared/site-chrome/build.mjs`) to copy them into
each `<microsite>/src/site-chrome/`. The per-app color strategy is set in `build.mjs`.

## Build

**GitHub Actions** (`.github/workflows/deploy.yml`, runs on push to `master`/`main`): Ruby + bundler cache →
`JEKYLL_ENV=production bundle exec jekyll build` → `scripts/build-microsites.sh` → html-proofer → deploy to Pages.
In the repository settings, Pages → Source must be **GitHub Actions**.

**Local, Docker** (mirrors CI, serves via nginx):

```bash
make build   # docker compose build (Dockerfile.local)
make serve   # http://localhost:4000
```

**Local, no Docker** (Ruby 3.3 + Node 22):

```bash
make site    # Jekyll with _config.local.yml + all microsites into _site/
make check   # html-proofer (internal links)
```

## Pending settings

- `analytics.google.id` (GA4) and `webmaster_verifications.google` (Search Console) in `_config.yml`

Theme: Chirpy by Cotes Chung (MIT). The earlier Jalpc design was based on
[JiaKunUp/jalpc_jekyll_theme](https://github.com/JiaKunUp/jalpc_jekyll_theme).
