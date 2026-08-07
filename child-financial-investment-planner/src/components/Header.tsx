import React from 'react';
import { ParentInputs } from '../types';
import { US_STATES } from '../data/accountData';
import {
  GraduationCap,
  TrendingUp,
  Download,
  Sparkles,
  ShieldCheck,
  Building2,
  DollarSign,
  Calendar,
  Calculator,
  Baby,
  Home,
} from 'lucide-react';

interface HeaderProps {
  inputs: ParentInputs;
  onUpdateInputs: (updated: Partial<ParentInputs>) => void;
  onExportCSV: () => void;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  projectedAge18Mod: number;
  projectedAge60Mod: number;
}

export const Header: React.FC<HeaderProps> = ({
  inputs,
  onUpdateInputs,
  onExportCSV,
  activeTab,
  setActiveTab,
  projectedAge18Mod,
  projectedAge60Mod,
}) => {
  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const navTabs = [
    { id: 'optimizer', label: 'Portfolio Allocator', icon: Calculator },
    { id: 'comparison', label: 'Account Matrix', icon: Building2 },
    { id: 'projections', label: 'Growth Scenarios', icon: TrendingUp },
    { id: 'rollover', label: 'Rollover Maximizer', icon: ShieldCheck },
    { id: 'spreadsheet', label: 'Excel Grid View', icon: GraduationCap },
    { id: 'ai_advisor', label: 'AI Advisor', icon: Sparkles },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 shadow-xl text-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          
          {/* Logo & Brand Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-teal-600 to-indigo-600 flex items-center justify-center text-white font-black shadow-lg">
              <Baby className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-extrabold text-white tracking-tight leading-none flex items-center gap-2">
                <span>Child Financial Investment Planner</span>
                <span className="text-[10px] font-mono uppercase bg-teal-500/20 text-teal-300 border border-teal-500/30 px-2 py-0.5 rounded-full">
                  Step 3 of 3
                </span>
              </h1>
              <p className="text-xs text-slate-400 font-medium">
                Personal Wealth Operating System • 18-Year 529, Custodial IRA &amp; Trump Account Engine
              </p>
            </div>
          </div>

          {/* Stepper Navigation (3-Step Roadmap) & Home Button */}
          <div className="flex items-center gap-2 flex-wrap">
            <a
              href="https://karthikbalasubramanian.github.io/"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 text-xs font-bold transition-all shadow-sm shrink-0"
              title="Return to Main Website Home Page"
            >
              <Home className="w-3.5 h-3.5 text-teal-400" />
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

              {/* Step 2 */}
              <a
                href="https://karthikbalasubramanian.github.io/rent-vs-buy-house-poor-calculator/"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 transition-all whitespace-nowrap"
              >
                <span className="w-4 h-4 rounded-full bg-slate-800 text-slate-300 text-[10px] flex items-center justify-center font-bold">2</span>
                <span>Lifestyle (Rent vs Buy)</span>
              </a>

              <span className="text-slate-700 font-bold">→</span>

              {/* Step 3 (Active) */}
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 text-white font-bold shadow-md shadow-teal-950 whitespace-nowrap">
                <span className="w-4 h-4 rounded-full bg-white text-teal-700 text-[10px] flex items-center justify-center font-extrabold">3</span>
                <span>Child 529 Planner</span>
              </div>
            </div>
          </div>

        </div>

        {/* Quick Highlights Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 pt-3 border-t border-slate-800/80 text-xs">
          
          <div className="bg-slate-950 border border-slate-800 p-2.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-400 block">Monthly Budget</span>
              <span className="text-sm font-bold font-mono text-emerald-300">
                {formatCurrency(inputs.monthlyContribution)}<span className="text-[10px] font-normal text-slate-400">/mo</span>
              </span>
            </div>
            <DollarSign className="w-4 h-4 text-emerald-400" />
          </div>

          <div className="bg-slate-950 border border-slate-800 p-2.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-400 block">Child Age</span>
              <span className="text-sm font-bold font-mono text-slate-200">
                {inputs.childCurrentAge} <span className="text-[10px] font-normal text-slate-400">yrs old</span>
              </span>
            </div>
            <Calendar className="w-4 h-4 text-amber-400" />
          </div>

          <div className="bg-slate-950 border border-teal-800/60 p-2.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[10px] uppercase font-bold text-teal-400 block">Age 18 College Fund</span>
              <span className="text-sm font-bold font-mono text-teal-300">
                {formatCurrency(projectedAge18Mod)}
              </span>
            </div>
            <GraduationCap className="w-4 h-4 text-teal-400" />
          </div>

          <div className="bg-slate-950 border border-indigo-800/60 p-2.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[10px] uppercase font-bold text-indigo-400 block">Age 60 IRA Retirement</span>
              <span className="text-sm font-bold font-mono text-indigo-300">
                {formatCurrency(projectedAge60Mod)}
              </span>
            </div>
            <TrendingUp className="w-4 h-4 text-indigo-400" />
          </div>

        </div>

        {/* Navigation Tabs Sub-bar & Actions */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mt-3 pt-2.5 border-t border-slate-800/80">
          
          <nav className="flex space-x-1 overflow-x-auto py-0.5 no-scrollbar">
            {navTabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`nav-tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg whitespace-nowrap transition-all ${
                    isActive
                      ? 'bg-teal-600 text-white font-bold shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-950'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-2 shrink-0">
            {/* State Tax Rule Selector */}
            <select
              id="state-tax-rule-select"
              value={inputs.state}
              onChange={(e) => onUpdateInputs({ state: e.target.value })}
              className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-200 focus:outline-none font-mono font-bold"
            >
              {US_STATES.map((st) => (
                <option key={st.code} value={st.code}>
                  {st.name} ({st.code})
                </option>
              ))}
            </select>

            {/* Export CSV Button */}
            <button
              id="export-csv-button"
              onClick={onExportCSV}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white font-bold text-xs transition-all shadow-md"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>
          </div>

        </div>

      </div>
    </header>
  );
};
