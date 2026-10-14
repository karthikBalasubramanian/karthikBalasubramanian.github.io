import React from 'react';
import { POPULAR_TICKERS } from '../services/stockApi';
import { Shield, TrendingUp, Gamepad2, BookOpen, Sparkles, Home } from 'lucide-react';

interface HeaderProps {
  currentSymbol: string;
  onSelectTicker: (symbol: string) => void;
  activeTab: 'dashboard' | 'game' | 'guide';
  onTabChange: (tab: 'dashboard' | 'game' | 'guide') => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentSymbol,
  onSelectTicker,
  activeTab,
  onTabChange,
}) => {
  return (
    <header className="bg-slate-900/90 backdrop-blur-xl border-b border-slate-800 sticky top-0 z-50 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {/* Top Title Bar */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-3">
            <a
              href="https://karthikbalasubramanian.github.io/"
              className="p-2.5 bg-gradient-to-br from-emerald-500 to-teal-700 rounded-xl shadow-lg shadow-emerald-950/50 text-white hover:scale-105 transition-transform"
              title="Return to Main Website Home Page"
            >
              <TrendingUp className="w-6 h-6" />
            </a>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                  RSU Hold vs. Sell <span className="text-emerald-400">Simulator & Game</span>
                </h1>
                <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <Shield className="w-3 h-3" /> Mass Wisdom Engine
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 font-medium">
                Empirical simulations & decision game for public company employees
              </p>
            </div>
          </div>

          {/* Navigation Actions & Tabs */}
          <div className="flex items-center gap-2.5 flex-wrap">
            <a
              href="https://karthikbalasubramanian.github.io/"
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 text-xs font-bold transition-all shadow-sm shrink-0"
              title="Return to Main Website Home Page"
            >
              <Home className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden sm:inline">Main Site Home</span>
              <span className="sm:hidden">Home</span>
            </a>

            {/* Navigation Tabs */}
            <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800/80">
            <button
              onClick={() => onTabChange('dashboard')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'dashboard'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>Simulator Dashboard</span>
            </button>
            <button
              onClick={() => onTabChange('game')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'game'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <Gamepad2 className="w-4 h-4" />
              <span>Vesting Time Machine</span>
            </button>
            <button
              onClick={() => onTabChange('guide')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'guide'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span>Wisdom Guide</span>
            </button>
          </div>
        </div>
      </div>

        {/* Quick Ticker Chips */}
        <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center gap-2 overflow-x-auto no-scrollbar">
          <span className="text-xs font-semibold text-slate-500 flex items-center gap-1 shrink-0">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Popular Tech RSUs:
          </span>
          {POPULAR_TICKERS.map(t => (
            <button
              key={t.symbol}
              onClick={() => onSelectTicker(t.symbol)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold shrink-0 transition-all ${
                currentSymbol.toUpperCase() === t.symbol
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'bg-slate-800/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent'
              }`}
            >
              {t.symbol}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
