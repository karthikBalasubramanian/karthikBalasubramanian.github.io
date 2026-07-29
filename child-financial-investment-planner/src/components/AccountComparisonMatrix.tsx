import React, { useState } from 'react';
import { ACCOUNT_DATA } from '../data/accountData';
import { AccountId } from '../types';
import {
  Check,
  X,
  ShieldCheck,
  HelpCircle,
  ExternalLink,
  Award,
  BookOpen,
  Scale,
  DollarSign,
  Info,
  Sparkles,
  Zap,
} from 'lucide-react';

export const AccountComparisonMatrix: React.FC = () => {
  const [selectedAccountId, setSelectedAccountId] = useState<AccountId>('529_plan');
  const [filterTag, setFilterTag] = useState<'all' | 'tax_free' | 'rollover_eligible' | 'high_flexibility'>('all');

  const accounts = Object.values(ACCOUNT_DATA);

  const filteredAccounts = accounts.filter((acc) => {
    if (filterTag === 'tax_free') return acc.taxTreatment.includes('100% Tax-Free');
    if (filterTag === 'rollover_eligible') return acc.iraRolloverEligible || acc.secure20Eligible;
    if (filterTag === 'high_flexibility') return acc.flexibility === 'High' || acc.flexibility === 'Moderate';
    return true;
  });

  const selectedAcc = ACCOUNT_DATA[selectedAccountId];

  return (
    <div className="space-y-8">
      {/* Overview Banner */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Strategic Account Comparison
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            529 vs. Trump Account vs. Custodial IRA Comparison
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm mt-1.5 leading-relaxed">
            Every child account has unique rules, tax advantages, and lifetime rollover options. Compare limits, tax treatments, age of control, and rollover strategies below to find your optimal mix.
          </p>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-xl">
          <button
            id="filter-all"
            onClick={() => setFilterTag('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterTag === 'all'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            All Accounts (6)
          </button>
          <button
            id="filter-rollover"
            onClick={() => setFilterTag('rollover_eligible')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterTag === 'rollover_eligible'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            ⚡ IRA Rollover Eligible
          </button>
          <button
            id="filter-tax-free"
            onClick={() => setFilterTag('tax_free')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterTag === 'tax_free'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            🛡️ 100% Tax-Free Growth
          </button>
          <button
            id="filter-high-flexibility"
            onClick={() => setFilterTag('high_flexibility')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterTag === 'high_flexibility'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            🔓 High Spending Flexibility
          </button>
        </div>
      </div>

      {/* Comparison Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAccounts.map((acc) => {
          const isSelected = selectedAccountId === acc.id;
          return (
            <div
              key={acc.id}
              id={`account-card-${acc.id}`}
              onClick={() => setSelectedAccountId(acc.id)}
              className={`cursor-pointer rounded-xl p-5 border transition-all duration-200 relative flex flex-col justify-between ${
                isSelected
                  ? 'bg-white dark:bg-slate-900 border-indigo-600 dark:border-indigo-500 shadow-sm ring-2 ring-indigo-500/20'
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 shadow-xs'
              }`}
            >
              <div>
                {/* Header Badge & Name */}
                <div className="flex items-start justify-between gap-2">
                  <span className={`px-2.5 py-1 rounded-md text-[10px] font-mono font-bold tracking-tight ${acc.bgColor} ${acc.borderColor} text-slate-800 dark:text-slate-200 uppercase`}>
                    {acc.badge}
                  </span>
                  {acc.secure20Eligible && (
                    <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border border-amber-300 dark:border-amber-800 flex items-center gap-1">
                      <Zap className="w-3 h-3 fill-amber-500" /> SECURE 2.0
                    </span>
                  )}
                </div>

                <h3 className="text-base font-bold text-slate-900 dark:text-white mt-3">
                  {acc.name}
                </h3>

                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                  {acc.description}
                </p>

                {/* Key Metrics */}
                <div className="space-y-2 mt-4 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                    <span className="text-slate-400 font-medium">Annual Cap:</span>
                    <span className="font-semibold font-mono text-slate-900 dark:text-slate-100">{acc.annualLimit}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                    <span className="text-slate-400 font-medium">Control Age:</span>
                    <span className="font-semibold font-mono text-slate-900 dark:text-slate-100">{acc.ageOfControl}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                    <span className="text-slate-400 font-medium">Flexibility:</span>
                    <span
                      className={`font-semibold ${
                        acc.flexibility === 'High'
                          ? 'text-indigo-600 dark:text-indigo-400'
                          : acc.flexibility === 'Moderate'
                          ? 'text-blue-600 dark:text-blue-400'
                          : 'text-amber-600 dark:text-amber-400'
                      }`}
                    >
                      {acc.flexibility}
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom Action */}
              <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs">
                <span className="text-slate-400">Click for details</span>
                <span className="font-semibold text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                  View Rules &rarr;
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Account Deep Dive Drawer / Panel */}
      {selectedAcc && (
        <div className="bg-slate-900 text-white rounded-xl p-6 sm:p-8 shadow-xl border border-slate-800 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-300 text-[10px] font-bold uppercase tracking-wider">
                  Deep-Dive Inspector
                </span>
                <span className="text-xs text-slate-400 font-mono">{selectedAcc.badge}</span>
              </div>
              <h3 className="text-2xl font-bold mt-1.5 text-white">{selectedAcc.name}</h3>
            </div>
            <div className="text-xs text-slate-300 bg-slate-800/80 px-4 py-2 rounded-lg border border-slate-700">
              <span className="font-bold text-indigo-400 uppercase tracking-wider text-[11px] block">Best For: </span>
              {selectedAcc.bestFor}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Column 1: Core Rules & Limits */}
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
                <BookOpen className="w-4 h-4" /> Rules &amp; Limits
              </h4>
              <div className="bg-slate-800/60 p-4 rounded-xl space-y-3 text-xs">
                <div>
                  <div className="text-slate-400 font-medium">Contribution Cap</div>
                  <div className="text-sm font-bold font-mono text-white mt-0.5">{selectedAcc.annualLimit}</div>
                </div>
                <div>
                  <div className="text-slate-400 font-medium">Tax Treatment</div>
                  <div className="text-sm font-semibold text-indigo-300 mt-0.5">{selectedAcc.taxTreatment}</div>
                </div>
                <div>
                  <div className="text-slate-400 font-medium">Age of Majority / Control</div>
                  <div className="text-sm font-semibold font-mono text-white mt-0.5">{selectedAcc.ageOfControl}</div>
                </div>
              </div>
            </div>

            {/* Column 2: Pros */}
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
                <Check className="w-4 h-4" /> Major Advantages
              </h4>
              <ul className="space-y-2 text-xs">
                {selectedAcc.pros.map((pro, idx) => (
                  <li key={idx} className="flex items-start gap-2 bg-slate-800/40 p-2.5 rounded-lg border border-slate-800">
                    <Check className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                    <span className="text-slate-200">{pro}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Column 3: Cons & Key Rules */}
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                <Scale className="w-4 h-4" /> Constraints &amp; Fine Print
              </h4>
              <ul className="space-y-2 text-xs">
                {selectedAcc.cons.map((con, idx) => (
                  <li key={idx} className="flex items-start gap-2 bg-slate-800/40 p-2.5 rounded-lg border border-slate-800">
                    <X className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <span className="text-slate-300">{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Key Guidelines Banner */}
          <div className="bg-indigo-950/40 border border-indigo-800/60 rounded-xl p-4 text-xs text-indigo-200 space-y-1">
            <div className="font-bold text-indigo-400 flex items-center gap-1.5 uppercase tracking-wider text-[11px]">
              <ShieldCheck className="w-4 h-4" /> Statutory Rule:
            </div>
            {selectedAcc.keyRules.map((rule, idx) => (
              <p key={idx} className="text-slate-300">
                • {rule}
              </p>
            ))}
          </div>

          {/* Official Government & IRS/SEC Backing Links Section */}
          {selectedAcc.officialLinks && selectedAcc.officialLinks.length > 0 && (
            <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4 text-indigo-400" /> Official IRS, SEC &amp; Government Backing Sources
                </h4>
                <span className="text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded-md">
                  Verified Official Documentation
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {selectedAcc.officialLinks.map((link, idx) => (
                  <a
                    key={idx}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-3.5 rounded-lg bg-slate-900/90 hover:bg-slate-950 border border-slate-700 hover:border-indigo-500 transition-all text-xs group flex flex-col justify-between space-y-2"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                          {link.source}
                        </span>
                        <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-indigo-400 transition-colors" />
                      </div>
                      <div className="font-semibold text-slate-200 group-hover:text-white line-clamp-2">
                        {link.title}
                      </div>
                    </div>
                    {link.description && (
                      <p className="text-[11px] text-slate-400 line-clamp-2">
                        {link.description}
                      </p>
                    )}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Side-by-Side Comparison Table */}
      <div className="bg-white dark:bg-slate-900 rounded-xl p-6 border border-slate-200 dark:border-slate-800 shadow-xs overflow-x-auto">
        <h3 className="text-base font-bold text-slate-900 dark:text-white mb-4">
          Full Side-by-Side Account Feature Table
        </h3>
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200 dark:border-slate-800 text-[11px] uppercase tracking-widest text-slate-400 font-semibold">
              <th className="px-4 py-3">Account Type</th>
              <th className="px-4 py-3">Annual Limit</th>
              <th className="px-4 py-3">Tax Growth</th>
              <th className="px-4 py-3">Age of Control</th>
              <th className="px-4 py-3">IRA Rollover Option</th>
              <th className="px-4 py-3">Flexibility</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {accounts.map((acc) => (
              <tr
                key={acc.id}
                className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
              >
                <td className="px-4 py-3.5 font-bold text-indigo-600 dark:text-indigo-400 flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${acc.bgColor} border ${acc.borderColor}`} />
                  {acc.shortName}
                </td>
                <td className="px-4 py-3.5 font-mono text-slate-700 dark:text-slate-300 font-medium">{acc.annualLimit}</td>
                <td className="px-4 py-3.5 text-slate-700 dark:text-slate-300">{acc.taxTreatment}</td>
                <td className="px-4 py-3.5 font-mono text-slate-700 dark:text-slate-300 font-medium">{acc.ageOfControl}</td>
                <td className="px-4 py-3.5 font-semibold">
                  {acc.secure20Eligible ? (
                    <span className="text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/50 px-2 py-0.5 rounded-md border border-indigo-200 dark:border-indigo-800 font-mono text-[11px]">
                      Roth IRA (Up to $35k)
                    </span>
                  ) : acc.iraRolloverEligible ? (
                    <span className="text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/50 px-2 py-0.5 rounded-md border border-amber-200 dark:border-amber-800 font-mono text-[11px]">
                      Lifetime Growth (Age 18)
                    </span>
                  ) : (
                    <span className="text-slate-400 font-mono">N/A</span>
                  )}
                </td>
                <td className="px-4 py-3.5 text-slate-700 dark:text-slate-300">{acc.flexibility}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Complete Official References Hub */}
      <div className="bg-white dark:bg-slate-900 rounded-xl p-6 border border-slate-200 dark:border-slate-800 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-600" /> Master Directory of Official Government &amp; Regulatory References
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Direct links to IRS Publications, SEC Investor Bulletins, and Congressional statutes backing every account rule and tax treatment in this app.
            </p>
          </div>
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 px-3 py-1 rounded-md border border-indigo-100 dark:border-indigo-900 shrink-0">
            IRS.gov • SEC.gov • Congress.gov
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
          {accounts.map((acc) => (
            <div
              key={acc.id}
              className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/30 space-y-3"
            >
              <div className="flex items-center justify-between border-b border-slate-200/60 dark:border-slate-700/60 pb-2">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${acc.bgColor} border ${acc.borderColor}`} />
                  <span className="font-bold text-xs text-slate-900 dark:text-white">{acc.name}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500 bg-slate-200/60 dark:bg-slate-700 px-2 py-0.5 rounded">
                  {acc.shortName}
                </span>
              </div>

              <div className="space-y-2">
                {acc.officialLinks.map((link, lIdx) => (
                  <a
                    key={lIdx}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700/80 hover:border-indigo-500 dark:hover:border-indigo-500 transition-all text-xs group"
                  >
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 font-mono">
                        {link.source}
                      </span>
                      <ExternalLink className="w-3 h-3 text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
                    </div>
                    <div className="font-semibold text-slate-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-300 line-clamp-1">
                      {link.title}
                    </div>
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
