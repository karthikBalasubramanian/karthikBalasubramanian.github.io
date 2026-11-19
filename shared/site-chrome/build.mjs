#!/usr/bin/env node
// Shared "site chrome" for the React/Vite microsites.
//
//   node shared/site-chrome/build.mjs
//
// 1. Regenerates tokens.{mirror,native,fixed}.css from Tailwind's default palette (read from a microsite's node_modules)
//    so all Tailwind color utilities follow the blog's light or dark mode (Chirpy theme).
// 2. Copies the right tokens file (as tokens.css), chrome.css and SiteChrome.tsx into <microsite>/src/site-chrome/.
//
// Edit the files in shared/site-chrome/ only. The copies inside each microsite are generated.
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
// strategy:
//   'mirror' : the app was designed dark-only. Light mode mirrors every color scale (50<->950 ...).
//   'native' : the app already has Tailwind `dark:` variants. Scales are not mirrored; instead the
//              `dark:` variant is bound to html[data-bs-theme="dark"] (the blog's choice).
//   'fixed'  : the app mixes light and dark surfaces by design (no `dark:` variants), so neither
//              mirroring nor `dark:` works. Its colors stay the same in both modes (Chirpy greys);
//              only the shared header/footer follow the blog's mode.
export const MICROSITE_STRATEGY = {
  'child-financial-investment-planner': 'native',
  'paycheck-tax-investment-allocator': 'fixed',
  'rent-vs-buy-house-poor-calculator': 'mirror',
  'rsu-hold-vs-sell-simulator': 'mirror',
};
export const MICROSITES = Object.keys(MICROSITE_STRATEGY);

// --- 1. tokens -------------------------------------------------------------
const SHADES = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900', '950'];
const NEUTRALS = ['slate', 'gray', 'zinc', 'neutral', 'stone'];
const CHROMATIC = ['red', 'orange', 'amber', 'yellow', 'lime', 'green', 'emerald', 'teal', 'cyan',
  'sky', 'blue', 'indigo', 'violet', 'purple', 'fuchsia', 'pink', 'rose'];

// Chirpy 7.6 greys (see jekyll-theme-chirpy/_sass/themes/_dark.scss and _light.scss).
// Dark mode: 950 = page background ... 50 = brightest text.
const NEUTRAL_DARK = {
  50: '#ededed', 100: '#dcdcdc', 200: '#cccccc', 300: '#afb0b1', 400: '#a7a7a7', 500: '#868686',
  600: '#5c5e61', 700: '#404144', 800: '#303133', 900: '#242427', 950: '#1b1b1e',
};
// Light mode is the same scale mirrored: 950 = white page background ... 50 = darkest text.
// 'native' strategy: one ordered grey scale (light greys from Chirpy light, dark greys from Chirpy dark).
const NEUTRAL_NATIVE = {
  50: '#f6f8fa', 100: '#efefef', 200: '#e9ecef', 300: '#c5c5c5', 400: '#a7a7a7', 500: '#757575',
  600: '#585858', 700: '#404144', 800: '#303133', 900: '#242427', 950: '#1b1b1e',
};
const NEUTRAL_LIGHT = {
  50: '#1b1b1e', 100: '#22222a', 200: '#2a2a2a', 300: '#34343c', 400: '#585858', 500: '#757575',
  600: '#a0a4a8', 700: '#dee2e6', 800: '#e9ecef', 900: '#f6f8fa', 950: '#ffffff',
};

function readPalette() {
  for (const site of MICROSITES) {
    const p = join(root, site, 'node_modules', 'tailwindcss', 'theme.css');
    if (existsSync(p)) {
      const css = readFileSync(p, 'utf8');
      const out = {};
      for (const m of css.matchAll(/--color-([a-z]+)-(\d+):\s*([^;]+);/g)) {
        (out[m[1]] ??= {})[m[2]] = m[3].trim();
      }
      return out;
    }
  }
  throw new Error('Run npm install in at least one microsite first (needs node_modules/tailwindcss/theme.css).');
}

function buildTokens(strategy) {
  const pal = readPalette();
  const mirror = (s) => SHADES[SHADES.length - 1 - SHADES.indexOf(s)];
  const themeLines = [];
  const dark = [];
  const light = [];
  for (const name of NEUTRALS) {
    for (const s of SHADES) {
      themeLines.push(`  --color-${name}-${s}: var(--kb-${name}-${s});`);
      if (strategy !== 'mirror') {
        dark.push(`  --kb-${name}-${s}: ${NEUTRAL_NATIVE[s]};`);
        light.push(`  --kb-${name}-${s}: ${NEUTRAL_NATIVE[s]};`);
      } else {
        dark.push(`  --kb-${name}-${s}: ${NEUTRAL_DARK[s]};`);
        light.push(`  --kb-${name}-${s}: ${NEUTRAL_LIGHT[s]};`);
      }
    }
  }
  for (const name of CHROMATIC) {
    for (const s of SHADES) {
      themeLines.push(`  --color-${name}-${s}: var(--kb-${name}-${s});`);
      dark.push(`  --kb-${name}-${s}: ${pal[name][s]};`);
      light.push(`  --kb-${name}-${s}: ${pal[name][strategy === 'mirror' ? mirror(s) : s]};`);
    }
  }
  themeLines.push('  --color-white: var(--kb-white);', '  --color-black: var(--kb-black);');
  dark.push('  --kb-white: #ffffff;', '  --kb-black: #000000;');
  if (strategy !== 'mirror') light.push('  --kb-white: #ffffff;', '  --kb-black: #000000;');
  else light.push('  --kb-white: #1b1b1e;', '  --kb-black: #ffffff;');
  const variant = strategy === 'native'
    ? "\n/* Tailwind `dark:` follows the blog's mode instead of the OS media query. */\n@custom-variant dark (&:where([data-bs-theme='dark'], [data-bs-theme='dark'] *));\n"
    : '';

  return `/* GENERATED by shared/site-chrome/build.mjs (strategy: ${strategy}). Do not edit the copy inside a microsite.
 *
 * Remaps Tailwind's palette to CSS variables so the microsites follow the blog's light/dark mode.
 * - Dark (default): neutrals use Chirpy's dark greys; accent colors keep Tailwind's defaults.
 * - 'mirror' strategy, light (html[data-bs-theme="light"]): every scale is mirrored (50<->950, 100<->900, ...), so a
 *   dark-designed UI keeps its contrast; neutrals use Chirpy's light greys.
 * - 'native' strategy: scales are not mirrored and Tailwind's dark: variant follows data-bs-theme.
 * - 'fixed' strategy: same colors in both modes; only the shared header/footer switch.
 * The mode comes from localStorage key "theme", which Chirpy (the blog theme) also uses.
 */
${variant}
@theme {
${themeLines.join('\n')}
}

:root {
  color-scheme: dark;
${dark.join('\n')}
}

:root[data-bs-theme='light'] {
  color-scheme: light;
${light.join('\n')}
}
`;
}

for (const strategy of ['mirror', 'native', 'fixed']) writeFileSync(join(here, `tokens.${strategy}.css`), buildTokens(strategy));

// --- 2. sync ---------------------------------------------------------------
for (const site of MICROSITES) {
  const dest = join(root, site, 'src', 'site-chrome');
  mkdirSync(dest, { recursive: true });
  copyFileSync(join(here, `tokens.${MICROSITE_STRATEGY[site]}.css`), join(dest, 'tokens.css'));
  for (const f of ['chrome.css', 'SiteChrome.tsx']) copyFileSync(join(here, f), join(dest, f));
  console.log(`synced site-chrome (${MICROSITE_STRATEGY[site]}) -> ${site}/src/site-chrome/`);
}
