import React from 'react';
import { Home } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 shadow-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-600 via-purple-600 to-indigo-600 flex items-center justify-center text-white font-black shadow-lg">
              <Home className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-extrabold text-white tracking-tight leading-none flex items-center gap-2">
                <span>Big Purchase Affordability & Lifestyle Planner</span>
                <span className="text-[10px] font-mono uppercase bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2 py-0.5 rounded-full">
                  Step 2 of 3
                </span>
              </h1>
              <p className="text-xs text-slate-400 font-medium">
                Personal Wealth Operating System • Stress-tests Homes, 2nd Homes, Luxury Cars & Major Assets
              </p>
            </div>
          </div>

          {/* Stepper Navigation & Home Button */}
          <div className="flex items-center gap-2 flex-wrap">
            <a
              href="https://karthikbalasubramanian.github.io/"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 text-xs font-bold transition-all shadow-sm shrink-0"
              title="Return to Main Website Home Page"
            >
              <Home className="w-3.5 h-3.5 text-rose-400" />
              <span>Main Site Home</span>
            </a>

            <div className="flex items-center gap-1 sm:gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs font-semibold overflow-x-auto">
              {/* Step 1 */}
              <a
                href="https://karthikbalasubramanian.github.io/paycheck-tax-investment-allocator/"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 transition-all whitespace-nowrap"
              >
                <span className="w-4 h-4 rounded-full bg-slate-800 text-slate-300 text-[10px] flex items-center justify-center font-bold">1</span>
                <span>Paycheck Allocator</span>
              </a>

              <span className="text-slate-700 font-bold">→</span>

              {/* Step 2 (Active) */}
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600 text-white font-bold shadow-md shadow-rose-950 whitespace-nowrap">
                <span className="w-4 h-4 rounded-full bg-white text-rose-700 text-[10px] flex items-center justify-center font-extrabold">2</span>
                <span>Big Purchase Planner</span>
              </div>

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

        </div>
      </div>
    </header>
  );
};
