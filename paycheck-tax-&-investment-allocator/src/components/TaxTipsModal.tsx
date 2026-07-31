import React from 'react';
import { X, HelpCircle, ShieldCheck, Sparkles, TrendingUp, Award, BookOpen } from 'lucide-react';

interface TaxTipsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TaxTipsModal: React.FC<TaxTipsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 text-slate-100 shadow-2xl space-y-5">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-400" />
            <h3 className="text-lg font-bold text-white">Financial & Tax Optimization Guide</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content sections */}
        <div className="space-y-4 text-xs text-slate-300">
          
          {/* Pre-Tax vs Post-Tax */}
          <div className="bg-slate-950 border border-purple-900/60 rounded-xl p-4 space-y-1.5">
            <h4 className="font-bold text-purple-300 text-sm flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4" /> Traditional 401(k) vs. Roth 401(k) / Roth IRA
            </h4>
            <p>
              <strong className="text-purple-200">Pre-Tax 401(k):</strong> Contributions lower your current year taxable income dollar-for-dollar. If you are in high marginal tax brackets (24%+ Federal + State), pre-tax savings save substantial cash today.
            </p>
            <p>
              <strong className="text-emerald-200">Roth (Post-Tax):</strong> Contributions are made after tax, but grow 100% tax-free for life. All compound gains and future qualified withdrawals in retirement are completely tax-free!
            </p>
          </div>

          {/* HSA Triple Tax Advantage */}
          <div className="bg-slate-950 border border-teal-900/60 rounded-xl p-4 space-y-1.5">
            <h4 className="font-bold text-teal-300 text-sm flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-yellow-400" /> The HSA "Triple Tax Advantage"
            </h4>
            <p>
              Health Savings Accounts (HSAs) via payroll deduction enjoy unmatched tax benefits:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-slate-300">
              <li>1. Tax-deductible (pre-tax deduction)</li>
              <li>2. Avoids 7.65% FICA (Social Security & Medicare) taxes when done via payroll!</li>
              <li>3. Grows tax-free and withdraws tax-free for qualified medical expenses.</li>
            </ul>
          </div>

          {/* ESPP Discount Mechanics */}
          <div className="bg-slate-950 border border-indigo-900/60 rounded-xl p-4 space-y-1.5">
            <h4 className="font-bold text-indigo-300 text-sm flex items-center gap-1.5">
              <Award className="w-4 h-4" /> Employee Stock Purchase Plan (ESPP) Instant Return
            </h4>
            <p>
              Participating in ESPP with a 15% discount off the stock price provides an immediate guaranteed return upon purchase (~17.6% instant return on invested capital!). Many financial planners recommend selling immediately at purchase to lock in profit.
            </p>
          </div>

          {/* Child Accounts & 529 */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1.5">
            <h4 className="font-bold text-teal-300 text-sm flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4" /> 529 College & Child Custodial Accounts
            </h4>
            <p>
              529 plans offer tax-free compounding for education. Furthermore, under recent SECURE 2.0 legislation, up to $35,000 of unused 529 funds can be rolled over tax-free into a Roth IRA for the child!
            </p>
          </div>

        </div>

        {/* Footer */}
        <div className="pt-3 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl transition-all"
          >
            Got it, thanks!
          </button>
        </div>

      </div>
    </div>
  );
};
