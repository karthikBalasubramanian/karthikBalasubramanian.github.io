import React from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { Home, ArrowRight, Sparkles, ShieldAlert } from 'lucide-react';

interface Step2RentVsBuyCTAProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
}

export const Step2RentVsBuyCTA: React.FC<Step2RentVsBuyCTAProps> = ({ inputs, taxResult }) => {
  const monthlyNet = Math.round(taxResult.netTakeHomePayBiweekly * (26 / 12));
  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const targetUrl = `https://karthikbalasubramanian.github.io/rent-vs-buy-house-poor-calculator/?monthlyNet=${monthlyNet}&state=${inputs.state}`;

  return (
    <div className="bg-gradient-to-r from-rose-950/80 via-slate-900 to-indigo-950 border border-rose-500/50 rounded-2xl p-6 shadow-2xl text-slate-100 space-y-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-2xl bg-rose-500/20 text-rose-300 border border-rose-500/30 shrink-0 mt-1">
            <Home className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase font-mono font-bold bg-rose-950 text-rose-300 border border-rose-800 px-2 py-0.5 rounded-full">
                Next Step in Your Financial Roadmap • Step 2
              </span>
              <Sparkles className="w-4 h-4 text-amber-400" />
            </div>

            <h3 className="text-lg font-extrabold text-white">
              Now do you want to determine what money is left after groceries, quality of life spend, & subscriptions?
            </h3>

            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              You have <strong className="text-emerald-400 font-mono">{fmt(monthlyNet)}/month</strong> in net take-home cash. Evaluate if buying a home in your target ZIP Code will leave you <strong className="text-rose-300">House Poor</strong> vs <strong className="text-amber-300">Renting</strong>!
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <a
          href={targetUrl}
          className="flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-rose-600 via-purple-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white font-extrabold text-xs shadow-xl transition-all shrink-0 font-sans border border-rose-300/30 hover:scale-105"
        >
          <span>Continue to Step 2: Rent vs Buy Stress Tester</span>
          <ArrowRight className="w-4 h-4" />
        </a>

      </div>
    </div>
  );
};
