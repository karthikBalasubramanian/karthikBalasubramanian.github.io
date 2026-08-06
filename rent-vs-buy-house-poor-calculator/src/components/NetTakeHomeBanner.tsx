import React from 'react';
import type { UserHousingInputs } from '../types';
import { Wallet } from 'lucide-react';

interface NetTakeHomeBannerProps {
  inputs: UserHousingInputs;
  onChange: (updated: Partial<UserHousingInputs>) => void;
  fromPaycheckApp?: boolean;
}

export const NetTakeHomeBanner: React.FC<NetTakeHomeBannerProps> = ({
  inputs,
  onChange,
  fromPaycheckApp,
}) => {
  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/80 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shrink-0">
            <Wallet className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase font-extrabold tracking-wider text-emerald-400">
                Step 1 Handoff Data
              </span>
              {fromPaycheckApp && (
                <span className="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded-full font-mono">
                  ✓ Auto-synced from Paycheck Allocator
                </span>
              )}
            </div>
            <h2 className="text-lg font-bold text-white">
              Monthly Liquid Net Take-Home Pay
            </h2>
            <p className="text-xs text-slate-400">
              Net cash deposited in your bank account after all taxes, 401(k), HSA, & ESPP deductions.
            </p>
          </div>
        </div>

        {/* Input Box */}
        <div className="flex items-center gap-3 bg-slate-950 border border-slate-800 p-3 rounded-xl">
          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400">
              Monthly Net Cash ($/mo)
            </label>
            <div className="flex items-center gap-1 font-mono text-xl font-extrabold text-emerald-400">
              <span>$</span>
              <input
                type="number"
                value={inputs.monthlyTakeHome}
                onChange={(e) => onChange({ monthlyTakeHome: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="bg-transparent text-emerald-400 focus:outline-none w-28 font-mono"
              />
            </div>
          </div>

          <div className="border-l border-slate-800 pl-3 text-right">
            <span className="text-[10px] text-slate-400 block font-semibold">Annualized</span>
            <span className="text-xs font-mono font-bold text-slate-300">
              {fmt(inputs.monthlyTakeHome * 12)}/yr
            </span>
          </div>
        </div>

      </div>
    </div>
  );
};
