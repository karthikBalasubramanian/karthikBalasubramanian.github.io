import React from 'react';
import { ParentInputs } from '../types';
import { US_STATES } from '../data/accountData';
import { GlobalNavbar } from './GlobalNavbar';
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
    <>
      <GlobalNavbar currentAppName="Child Financial Planner" />
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 sticky top-0 z-30 shadow-xs">
        {/* Top Banner */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            {/* App Title & Metadata */}
            <div>
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 bg-[#1ab394] rounded-lg flex items-center justify-center text-white font-bold text-xs shadow-xs">
                  KB
                </div>
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Financial Strategy Suite
                </span>
                <span className="text-slate-300 dark:text-slate-700">|</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">529 &amp; Custodial Wealth Engine</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white tracking-tight mt-1.5">
                Child Financial Investment Planner
              </h1>
              <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                Compare 529 Plans, Trump Accounts, Custodial IRAs &amp; UTMA accounts. Strategic tax-advantaged compounding.
              </p>
            </div>

            {/* Quick Actions & State Selector */}
            <div className="flex flex-wrap items-center gap-3">
              {/* State Selector for 529 tax deductions */}
              <div className="flex flex-col">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">
                  State Tax Rule
                </label>
                <select
                  id="state-tax-rule-select"
                  value={inputs.state}
                  onChange={(e) => onUpdateInputs({ state: e.target.value })}
                  className="text-xs px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-2 focus:ring-[#1ab394] font-medium"
                >
                  {US_STATES.map((st) => (
                    <option key={st.code} value={st.code}>
                      {st.name} ({st.code})
                    </option>
                  ))}
                </select>
              </div>

              {/* Export CSV Button */}
              <button
                id="export-csv-button"
                onClick={onExportCSV}
                className="mt-4 sm:mt-0 inline-flex items-center gap-1.5 px-3.5 py-2 rounded-md bg-[#1ab394] hover:bg-[#18a689] text-white font-medium text-xs transition-colors shadow-xs"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
            </div>
          </div>

          {/* Quick Highlights Stats Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-100 dark:border-slate-800/80">
            <div className="bg-slate-50 dark:bg-slate-800/40 p-3 rounded-xl border border-slate-200/80 dark:border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400 uppercase tracking-wider font-semibold">
                <DollarSign className="w-3.5 h-3.5 text-[#1ab394]" />
                Monthly Budget
              </div>
              <div className="text-lg font-bold font-mono text-slate-900 dark:text-white mt-1">
                {formatCurrency(inputs.monthlyContribution)} <span className="text-xs font-normal text-slate-400 font-sans">/mo</span>
              </div>
            </div>

            <div className="bg-slate-50 dark:bg-slate-800/40 p-3 rounded-xl border border-slate-200/80 dark:border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400 uppercase tracking-wider font-semibold">
                <Calendar className="w-3.5 h-3.5 text-slate-600 dark:text-slate-400" />
                Child Age
              </div>
              <div className="text-lg font-bold font-mono text-slate-900 dark:text-white mt-1">
                {inputs.childCurrentAge} <span className="text-xs font-normal text-slate-400 font-sans">yrs old</span>
              </div>
            </div>

            <div className="bg-[#1ab394]/10 dark:bg-[#1ab394]/20 p-3 rounded-xl border border-[#1ab394]/20">
              <div className="flex items-center gap-1.5 text-[11px] text-[#1ab394] uppercase tracking-wider font-semibold">
                <GraduationCap className="w-3.5 h-3.5" />
                Projected at Age 18
              </div>
              <div className="text-lg font-bold font-mono text-slate-900 dark:text-emerald-300 mt-1">
                {formatCurrency(projectedAge18Mod)}
              </div>
            </div>

            <div className="bg-[#2f4050] p-3 rounded-xl border border-slate-700 text-white">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-300 uppercase tracking-wider font-semibold">
                <TrendingUp className="w-3.5 h-3.5 text-[#1ab394]" />
                IRA Value at Age 60
              </div>
              <div className="text-lg font-bold font-mono text-[#1ab394] mt-1">
                {formatCurrency(projectedAge60Mod)}
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800">
          <nav className="flex space-x-1 sm:space-x-3 overflow-x-auto py-2 no-scrollbar">
            {navTabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`nav-tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg whitespace-nowrap transition-all ${
                    isActive
                      ? 'bg-[#1ab394] text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-white dark:hover:bg-slate-800'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>
    </>
  );
};
