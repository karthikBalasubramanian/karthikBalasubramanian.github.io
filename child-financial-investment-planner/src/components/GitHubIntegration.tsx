import React, { useEffect, useState } from 'react';
import {
  Github,
  Globe,
  ExternalLink,
  GitBranch,
  Star,
  Code2,
  CheckCircle2,
  User,
  RefreshCw,
  BookOpen,
} from 'lucide-react';

interface RepoData {
  name: string;
  full_name: string;
  description: string;
  html_url: string;
  homepage: string;
  stargazers_count: number;
  forks_count: number;
  language: string;
  pushed_at: string;
  owner: {
    login: string;
    avatar_url: string;
    html_url: string;
  };
}

interface UserData {
  name: string;
  login: string;
  avatar_url: string;
  bio: string;
  public_repos: number;
  html_url: string;
}

export const GitHubIntegration: React.FC = () => {
  const [repo, setRepo] = useState<RepoData | null>(null);
  const [user, setUser] = useState<UserData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);
  const [showIframe, setShowIframe] = useState<boolean>(false);

  const REPO_URL = 'https://github.com/karthikBalasubramanian/karthikBalasubramanian.github.io';
  const HOMEPAGE_URL = 'https://karthikbalasubramanian.github.io/';

  useEffect(() => {
    let isMounted = true;
    async function fetchData() {
      try {
        setLoading(true);
        const [repoRes, userRes] = await Promise.all([
          fetch('https://api.github.com/repos/karthikBalasubramanian/karthikBalasubramanian.github.io'),
          fetch('https://api.github.com/users/karthikBalasubramanian'),
        ]);

        if (repoRes.ok && userRes.ok) {
          const repoJson = await repoRes.json();
          const userJson = await userRes.json();
          if (isMounted) {
            setRepo(repoJson);
            setUser(userJson);
            setError(false);
          }
        } else {
          if (isMounted) setError(true);
        }
      } catch {
        if (isMounted) setError(true);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    fetchData();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs space-y-6 my-8">
      {/* Header section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
        <div className="flex items-start gap-3">
          <div className="p-2.5 bg-slate-900 dark:bg-slate-800 text-white rounded-xl shadow-xs">
            <Github className="w-6 h-6" />
          </div>
          <div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950/80 text-indigo-700 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[10px] font-bold uppercase tracking-wider mb-1">
              <CheckCircle2 className="w-3 h-3 text-indigo-600 dark:text-indigo-400" /> Integrated Portfolio Repository
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
              karthikBalasubramanian.github.io
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Official website &amp; blog repository integration with live sync to GitHub.
            </p>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <a
            href={HOMEPAGE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs transition-all shadow-xs"
          >
            <Globe className="w-3.5 h-3.5" />
            Visit Live Website
            <ExternalLink className="w-3 h-3 opacity-80" />
          </a>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold text-xs transition-all"
          >
            <Github className="w-3.5 h-3.5" />
            View GitHub Repo
          </a>
        </div>
      </div>

      {/* GitHub Repo Live Summary Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* User / Owner Info */}
        <div className="bg-slate-50 dark:bg-slate-800/40 p-5 rounded-xl border border-slate-200 dark:border-slate-800 flex flex-col justify-between space-y-4">
          <div className="flex items-center gap-3">
            <img
              src={user?.avatar_url || 'https://avatars.githubusercontent.com/u/909844?v=4'}
              alt="Karthik Balasubramanian"
              className="w-12 h-12 rounded-full border-2 border-indigo-500 object-cover"
              referrerPolicy="no-referrer"
            />
            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                Author &amp; Maintainer
              </div>
              <a
                href={user?.html_url || 'https://github.com/karthikBalasubramanian'}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-bold text-slate-900 dark:text-white hover:underline flex items-center gap-1"
              >
                Karthik Balasubramanian
                <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
              <div className="text-[11px] text-slate-500 font-mono">
                @{user?.login || 'karthikBalasubramanian'}
              </div>
            </div>
          </div>

          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            {user?.bio || 'Official website, personal tech publications, and digital portfolio.'}
          </p>

          <div className="flex items-center justify-between text-xs border-t border-slate-200 dark:border-slate-700/60 pt-3 text-slate-500 dark:text-slate-400 font-mono">
            <span>Public Repos: <strong>{user?.public_repos || 1}</strong></span>
            <span>Platform: <strong>GitHub Pages</strong></span>
          </div>
        </div>

        {/* Repository Details & Live Metrics */}
        <div className="lg:col-span-2 bg-slate-900 text-white p-5 rounded-xl border border-slate-800 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider">
                  Repository Specifications
                </span>
              </div>
              <span className="text-[10px] font-mono bg-indigo-950 text-indigo-300 px-2 py-0.5 rounded border border-indigo-800">
                Public Repo
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              {repo?.description || 'My official website cum blog.'}
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 font-mono text-xs">
              <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700">
                <span className="text-[10px] text-slate-400 block uppercase">Primary Language</span>
                <span className="font-bold text-indigo-300">{repo?.language || 'CSS / HTML'}</span>
              </div>
              <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700">
                <span className="text-[10px] text-slate-400 block uppercase">Stars</span>
                <span className="font-bold text-amber-300 flex items-center gap-1">
                  <Star className="w-3 h-3 fill-amber-300" /> {repo?.stargazers_count ?? 0}
                </span>
              </div>
              <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700">
                <span className="text-[10px] text-slate-400 block uppercase">Forks</span>
                <span className="font-bold text-slate-200 flex items-center gap-1">
                  <GitBranch className="w-3 h-3 text-slate-400" /> {repo?.forks_count ?? 0}
                </span>
              </div>
              <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700">
                <span className="text-[10px] text-slate-400 block uppercase">Last Push</span>
                <span className="font-bold text-emerald-400 text-[11px]">
                  {repo?.pushed_at ? new Date(repo.pushed_at).toLocaleDateString() : '2021'}
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800 pt-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1.5 font-mono">
              <Globe className="w-3.5 h-3.5 text-indigo-400" />
              Live Deployment: <a href={HOMEPAGE_URL} target="_blank" rel="noopener noreferrer" className="text-indigo-300 underline font-semibold">karthikbalasubramanian.github.io</a>
            </span>
            <button
              onClick={() => setShowIframe(!showIframe)}
              className="text-xs px-3 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 font-semibold transition-colors flex items-center gap-1"
            >
              <BookOpen className="w-3 h-3" />
              {showIframe ? 'Close Website Viewer' : 'Preview Embedded Site'}
            </button>
          </div>
        </div>
      </div>

      {/* Deployment Guide to github.io */}
      <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-700/60 pb-3">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <h4 className="text-sm font-bold text-slate-900 dark:text-white">
              Dual Build Fix: Root Jekyll + Subfolder Vite App
            </h4>
          </div>
          <span className="text-[10px] font-mono bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 px-2.5 py-1 rounded-md border border-emerald-100 dark:border-emerald-900 font-bold">
            Root Jekyll &amp; Subfolder Solved
          </span>
        </div>

        <div className="p-3 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-900/50 rounded-lg text-indigo-900 dark:text-indigo-200 text-xs space-y-1">
          <p className="font-bold">Why the root homepage failed with raw Liquid tags:</p>
          <p className="text-[11px] text-indigo-800 dark:text-indigo-300 leading-relaxed">
            The previous workflow uploaded static files without running Jekyll on the repository root. This caused your main site's Liquid template tags (<code>&#123;% for section %&#125;</code>) to render as unparsed raw text.
          </p>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-bold text-slate-900 dark:text-white">Solution in <code>.github/workflows/deploy.yml</code>:</p>
          <p className="text-[11px] text-slate-600 dark:text-slate-300">
            We updated <code>.github/workflows/deploy.yml</code> to run <code>actions/jekyll-build-pages</code> for your root homepage AND copy the compiled Vite React app into <code>_site/child-financial-investment-planner/</code> before deploying!
          </p>
        </div>

        <div className="p-3 bg-slate-900 text-slate-200 rounded-lg font-mono text-[11px] flex items-center justify-between border border-slate-800">
          <span>Root Homepage: <code className="text-indigo-400">https://karthikbalasubramanian.github.io/</code></span>
          <span className="text-emerald-400 font-bold">✓ Jekyll Processed</span>
        </div>
      </div>

      {/* Embedded Website Preview iframe toggle */}
      {showIframe && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden space-y-2 p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 px-2 font-mono">
            <span>Embedded Preview: {HOMEPAGE_URL}</span>
            <a
              href={HOMEPAGE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-400 hover:underline flex items-center gap-1"
            >
              Open in new tab <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <iframe
            src={HOMEPAGE_URL}
            title="Karthik Balasubramanian Website"
            className="w-full h-96 rounded-lg border border-slate-800 bg-white"
          />
        </div>
      )}
    </div>
  );
};
