import React from 'react';
import { PresetProfile, PayFrequency } from '../types';
import { FINANCIAL_PRESETS } from '../data/presets';
import { GlobalNavbar } from './GlobalNavbar';
import { Wallet, Sparkles, HelpCircle, RotateCcw } from 'lucide-react';

interface HeaderProps {
  activePresetId: string;
  onSelectPreset: (preset: PresetProfile) => void;
  payFrequency: PayFrequency;
  onChangeFrequency: (freq: PayFrequency) => void;
  onReset: () => void;
  onOpenTips: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activePresetId,
  onSelectPreset,
  payFrequency,
  onChangeFrequency,
  onReset,
  onOpenTips,
}) => {
  return (
    <>
      <GlobalNavbar currentAppName="Paycheck Tax Allocator" />
      <header className="bg-slate-900 border-b border-slate-800 text-slate-100 py-4 px-4 sm:px-8 sticky top-0 z-40 shadow-xl backdrop-blur-md bg-slate-900/95">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-[#1ab394] via-emerald-600 to-teal-400 flex items-center justify-center text-white shadow-lg shadow-[#1ab394]/20">
              <Wallet className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-black tracking-tight text-white font-sans">
                  Paycheck &amp; Tax Allocator
                </h1>
                <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-bold bg-emerald-950 text-[#1ab394] border border-[#1ab394]/40">
                  Child Wealth + ESPP
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Interactive biweekly paycheck analysis, tax dissection, &amp; investment optimizer
              </p>
            </div>
          </div>

          {/* Controls & Presets */}
          <div className="flex flex-wrap items-center gap-2.5">
            
            {/* Frequency Toggle */}
            <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex items-center gap-0.5">
              <button
                onClick={() => onChangeFrequency('biweekly')}
                className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                  payFrequency === 'biweekly'
                    ? 'bg-[#1ab394] text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Biweekly (26x)
              </button>
              <button
                onClick={() => onChangeFrequency('semimonthly')}
                className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                  payFrequency === 'semimonthly'
                    ? 'bg-[#1ab394] text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Semi-Monthly (24x)
              </button>
              <button
                onClick={() => onChangeFrequency('annual')}
                className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                  payFrequency === 'annual'
                    ? 'bg-[#1ab394] text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Annual
              </button>
            </div>

            {/* Quick Presets Dropdown / Buttons */}
            <div className="hidden lg:flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
              <span className="text-[10px] uppercase font-bold text-slate-500 px-2 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-[#1ab394]" /> Presets:
              </span>
              {FINANCIAL_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => onSelectPreset(preset)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                    activePresetId === preset.id
                      ? 'bg-slate-800 text-[#1ab394] border border-[#1ab394]/40 font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                  title={preset.description}
                >
                  {preset.badge}
                </button>
              ))}
            </div>

            {/* Tips Guide Button */}
            <button
              onClick={onOpenTips}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white border border-slate-700 transition-all"
            >
              <HelpCircle className="w-3.5 h-3.5 text-[#1ab394]" />
              <span>Guide &amp; Tax Tips</span>
            </button>

            {/* Reset Button */}
            <button
              onClick={onReset}
              title="Reset to defaults"
              className="p-2 text-slate-400 hover:text-rose-400 hover:bg-slate-800 rounded-xl transition-all border border-transparent hover:border-slate-700"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>
    </>
  );
};
