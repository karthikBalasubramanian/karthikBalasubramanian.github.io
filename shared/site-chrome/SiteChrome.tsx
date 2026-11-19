/*
 * Shared header and footer for the microsites under /tools/.
 * Source: shared/site-chrome/SiteChrome.tsx. Copies in <microsite>/src/site-chrome/ are generated.
 *
 * Light/dark mode uses the same localStorage key ("theme") and <html data-bs-theme> attribute
 * as the blog's Chirpy theme, so the blog and the microsites always match.
 */
import { useEffect, useState } from 'react';

type Mode = 'light' | 'dark';

const STORAGE_KEY = 'theme';

const LINKS: ReadonlyArray<{ label: string; href: string }> = [
  { label: 'Home', href: '/' },
  { label: 'Blog', href: '/blog/' },
  { label: 'Projects', href: '/projects/' },
  { label: 'Tools', href: '/tools/' },
  { label: 'About', href: '/about/' },
];

function systemMode(): Mode {
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function storedMode(): Mode | null {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch {
    return null;
  }
}

function resolveMode(): Mode {
  return storedMode() ?? systemMode();
}

function applyMode(mode: Mode): void {
  document.documentElement.setAttribute('data-bs-theme', mode);
}

export function SiteHeader() {
  const [mode, setMode] = useState<Mode>(resolveMode);

  useEffect(() => {
    applyMode(mode);
  }, [mode]);

  useEffect(() => {
    // Follow changes made in another tab (e.g. toggled on the blog) and OS changes when nothing is stored.
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY || e.key === null) setMode(resolveMode());
    };
    const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
    const onSystem = () => {
      if (!storedMode()) setMode(systemMode());
    };
    window.addEventListener('storage', onStorage);
    mq?.addEventListener('change', onSystem);
    return () => {
      window.removeEventListener('storage', onStorage);
      mq?.removeEventListener('change', onSystem);
    };
  }, []);

  const toggle = () => {
    const next: Mode = mode === 'dark' ? 'light' : 'dark';
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* storage unavailable: still switch for this page view */
    }
    setMode(next);
  };

  return (
    <header className="kb-chrome kb-site-header">
      <div className="kb-chrome-inner">
        <a className="kb-brand" href="/">
          Karthik Balasubramanian
          <small>ML Inference &amp; GPU Systems</small>
        </a>
        <nav className="kb-nav" aria-label="Site">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} aria-current={l.href === '/tools/' ? 'page' : undefined}>
              {l.label}
            </a>
          ))}
        </nav>
        <button
          type="button"
          className="kb-mode-toggle"
          onClick={toggle}
          aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {mode === 'dark' ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="kb-chrome kb-site-footer">
      <div className="kb-chrome-inner">
        <span>
          © {new Date().getFullYear()} <a href="/about/">Karthik Balasubramanian</a>. Runs entirely in your browser.
          No data is collected. Not financial advice.
        </span>
        <nav aria-label="Footer">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href}>
              {l.label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
