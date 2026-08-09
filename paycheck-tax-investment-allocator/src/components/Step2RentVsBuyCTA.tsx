import React from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { Home, ArrowRight, Sparkles, ShieldAlert } from 'lucide-react';

interface Step2RentVsBuyCTAProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
}

export const Step2RentVsBuyCTA: React.FC<Step2RentVsBuyCTAProps> = ({ inputs, taxResult }) => {
  const annualNetTimeline = taxResult.schedule?.totalNetTakeHomeAnnual || (taxResult.netTakeHomePayBiweekly * 26);
  const monthlyNet = Math.round(annualNetTimeline / 12);
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
              Now evaluate how much major assets (Homes, 2nd Homes, Cars, Yachts) fit your lifestyle budget!
            </h3>

            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              You have <strong className="text-emerald-400 font-mono">{fmt(monthlyNet)}/month</strong> in net take-home cash. Stress-test your purchasing power and optimize lifestyle spend with our <strong className="text-rose-300">Big Purchase Affordability & Lifestyle Planner</strong>!
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <a
          href={targetUrl}
          className="flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-rose-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white font-extrabold text-sm shadow-xl transition-all border border-rose-400/30 group shrink-0"
        >
          <span>Continue to Step 2: Big Purchase Planner</span>
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </a>

      </div>
    </div>
  );
};
