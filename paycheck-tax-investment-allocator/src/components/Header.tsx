import React from 'react';
import { PresetProfile, PayFrequency } from '../types';
import { FINANCIAL_PRESETS } from '../data/presets';
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
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 shadow-xl text-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 via-teal-600 to-indigo-600 flex items-center justify-center text-white font-black shadow-lg">
              <Wallet className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-extrabold text-white tracking-tight leading-none flex items-center gap-2">
                <span>Paycheck Tax & Investment Allocator</span>
                <span className="text-[10px] font-mono uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                  Step 1 of 3
                </span>
              </h1>
              <p className="text-xs text-slate-400 font-medium">
                Personal Wealth Operating System • Calculates liquid net take-home pay in hand
              </p>
            </div>
          </div>

          {/* Stepper Navigation (3-Step Roadmap) */}
          <div className="flex items-center gap-1 sm:gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs font-semibold overflow-x-auto">
            
            {/* Step 1 (Active) */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 text-white font-bold shadow-md shadow-emerald-950 whitespace-nowrap">
              <span className="w-4 h-4 rounded-full bg-white text-emerald-700 text-[10px] flex items-center justify-center font-extrabold">1</span>
              <span>Paycheck Allocator</span>
            </div>

            <span className="text-slate-700 font-bold">→</span>

            {/* Step 2 */}
            <a
              href="https://karthikbalasubramanian.github.io/rent-vs-buy-house-poor-calculator/"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 transition-all whitespace-nowrap"
            >
              <span className="w-4 h-4 rounded-full bg-slate-800 text-slate-300 text-[10px] flex items-center justify-center font-bold">2</span>
              <span>Rent vs Buy</span>
            </a>

            <span className="text-slate-700 font-bold">→</span>

            {/* Step 3 */}
            <a
              href="https://karthikbalasubramanian.github.io/child-financial-investment-planner/"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 transition-all whitespace-nowrap"
            >
              <span className="w-4 h-4 rounded-full bg-slate-800 text-slate-300 text-[10px] flex items-center justify-center font-bold">3</span>
              <span>Child 529 Planner</span>
            </a>
          </div>

        </div>

        {/* Action Controls & Presets Sub-bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-3 pt-2.5 border-t border-slate-800/80">
          
          {/* Pay Frequency Toggle */}
          <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex items-center gap-0.5">
            <button
              onClick={() => onChangeFrequency('biweekly')}
              className={`px-3 py-1 text-xs font-mono font-bold rounded-lg transition-all ${
                payFrequency === 'biweekly'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Biweekly (26x)
            </button>
            <button
              onClick={() => onChangeFrequency('semimonthly')}
              className={`px-3 py-1 text-xs font-mono font-bold rounded-lg transition-all ${
                payFrequency === 'semimonthly'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Semi-Monthly (24x)
            </button>
            <button
              onClick={() => onChangeFrequency('annual')}
              className={`px-3 py-1 text-xs font-mono font-bold rounded-lg transition-all ${
                payFrequency === 'annual'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Annual
            </button>
          </div>

          {/* Quick Presets */}
          <div className="hidden lg:flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <span className="text-[10px] uppercase font-bold text-slate-400 px-2 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Presets:
            </span>
            {FINANCIAL_PRESETS.map((preset) => (
              <button
                key={preset.id}
                onClick={() => onSelectPreset(preset)}
                className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                  activePresetId === preset.id
                    ? 'bg-slate-800 text-emerald-300 border border-emerald-500/40 font-semibold'
                    : 'text-slate-400 hover:text-white hover:bg-slate-900'
                }`}
                title={preset.description}
              >
                {preset.badge}
              </button>
            ))}
          </div>

          {/* Guide & Reset */}
          <div className="flex items-center gap-2">
            <button
              onClick={onOpenTips}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-slate-950 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800 transition-all"
            >
              <HelpCircle className="w-3.5 h-3.5 text-emerald-400" />
              <span>Tax Guide &amp; Tips</span>
            </button>

            <button
              onClick={onReset}
              title="Reset to defaults"
              className="p-2 text-slate-400 hover:text-rose-400 hover:bg-slate-950 rounded-xl transition-all border border-slate-800"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

        </div>
      </div>
    </header>
  );
};
