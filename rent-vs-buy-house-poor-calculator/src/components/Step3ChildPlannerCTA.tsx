import React from 'react';
import type { UserHousingInputs } from '../types';
import { analyzeHousePoorStatus } from '../utils/calculator';
import { Baby, ArrowRight, Sparkles } from 'lucide-react';

interface Step3ChildPlannerCTAProps {
  inputs: UserHousingInputs;
}

export const Step3ChildPlannerCTA: React.FC<Step3ChildPlannerCTAProps> = ({ inputs }) => {
  const analysis = analyzeHousePoorStatus(inputs);
  const leftoverCash = Math.max(0, analysis.leftoverCashBufferBuy);

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const childPlannerUrl = `https://karthikbalasubramanian.github.io/child-financial-investment-planner/?monthlySurplus=${Math.round(leftoverCash)}`;

  return (
    <div className="bg-gradient-to-r from-purple-950/80 via-slate-900 to-indigo-950 border border-purple-500/50 rounded-2xl p-6 shadow-2xl text-slate-100 space-y-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-2xl bg-purple-500/20 text-purple-300 border border-purple-500/30 shrink-0 mt-1">
            <Baby className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase font-mono font-bold bg-purple-900/60 text-purple-200 border border-purple-700/60 px-2 py-0.5 rounded-full">
                Step 3 of 3 • Generational Wealth
              </span>
              <Sparkles className="w-4 h-4 text-amber-400" />
            </div>

            <h3 className="text-lg font-extrabold text-white">
              Now that your housing & lifestyle budget is set, are you preparing for your child's future?
            </h3>

            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Whether you rent or buy, you have <strong className="text-purple-300 font-mono">{fmt(leftoverCash)}/month</strong> in leftover cash buffer. See how allocating $200–$500/mo into a 529 College Plan & Custodial UTMA account compounds into $150,000+ for your child's 18th birthday!
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <a
          href={childPlannerUrl}
          className="flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-500 hover:from-purple-500 hover:to-teal-400 text-white font-extrabold text-xs shadow-xl transition-all shrink-0 font-sans border border-purple-300/30 hover:scale-105"
        >
          <span>Continue to Step 3: Child Financial Planner</span>
          <ArrowRight className="w-4 h-4" />
        </a>

      </div>
    </div>
  );
};
