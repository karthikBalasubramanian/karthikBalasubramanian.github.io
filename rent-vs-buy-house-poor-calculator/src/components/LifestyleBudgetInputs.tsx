import React from 'react';
import type { UserHousingInputs } from '../types';
import { ShoppingCart, Car, Wifi, Utensils, HeartPulse, Calculator } from 'lucide-react';

interface LifestyleBudgetInputsProps {
  inputs: UserHousingInputs;
  onChange: (updated: Partial<UserHousingInputs>) => void;
}

export const LifestyleBudgetInputs: React.FC<LifestyleBudgetInputsProps> = ({
  inputs,
  onChange,
}) => {
  const l = inputs.lifestyle;

  const updateField = (field: keyof typeof l, val: number) => {
    onChange({
      lifestyle: {
        ...l,
        [field]: Math.max(0, val),
      },
    });
  };

  const totalLifestyle =
    (l.groceries || 0) +
    (l.utilities || 0) +
    (l.carPaymentInsurance || 0) +
    (l.subscriptionsStreaming || 0) +
    (l.diningOutEntertainment || 0) +
    (l.healthMedical || 0) +
    (l.otherMisc || 0);

  const surplusBeforeHousing = Math.max(0, inputs.monthlyTakeHome - totalLifestyle);

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-5">
      
      {/* Title */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <ShoppingCart className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">
              Non-Housing Lifestyle & Living Expenses
            </h3>
            <p className="text-xs text-slate-400">
              Enter what you spend on groceries, car, utilities & subscriptions before housing.
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-[10px] text-slate-400 uppercase font-bold block">Total Living Cost</span>
          <span className="text-base font-mono font-bold text-indigo-400">{fmt(totalLifestyle)}/mo</span>
        </div>
      </div>

      {/* Sliders / Inputs Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
        
        {/* Groceries */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><ShoppingCart className="w-3.5 h-3.5 text-emerald-400" /> Groceries & Food</span>
            <span className="font-mono text-emerald-400">${l.groceries}</span>
          </div>
          <input
            type="range"
            min={200}
            max={3000}
            step={50}
            value={l.groceries}
            onChange={(e) => updateField('groceries', parseFloat(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>

        {/* Utilities */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><Wifi className="w-3.5 h-3.5 text-cyan-400" /> Utilities (Power, Internet)</span>
            <span className="font-mono text-cyan-400">${l.utilities}</span>
          </div>
          <input
            type="range"
            min={50}
            max={1000}
            step={25}
            value={l.utilities}
            onChange={(e) => updateField('utilities', parseFloat(e.target.value))}
            className="w-full accent-cyan-500 cursor-pointer"
          />
        </div>

        {/* Car & Insurance */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><Car className="w-3.5 h-3.5 text-amber-400" /> Car Loan & Gas/Insurance</span>
            <span className="font-mono text-amber-400">${l.carPaymentInsurance}</span>
          </div>
          <input
            type="range"
            min={0}
            max={2000}
            step={50}
            value={l.carPaymentInsurance}
            onChange={(e) => updateField('carPaymentInsurance', parseFloat(e.target.value))}
            className="w-full accent-amber-500 cursor-pointer"
          />
        </div>

        {/* Subscriptions */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><Wifi className="w-3.5 h-3.5 text-purple-400" /> Subscriptions & Streaming</span>
            <span className="font-mono text-purple-400">${l.subscriptionsStreaming}</span>
          </div>
          <input
            type="range"
            min={0}
            max={500}
            step={25}
            value={l.subscriptionsStreaming}
            onChange={(e) => updateField('subscriptionsStreaming', parseFloat(e.target.value))}
            className="w-full accent-purple-500 cursor-pointer"
          />
        </div>

        {/* Dining Out */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><Utensils className="w-3.5 h-3.5 text-rose-400" /> Dining Out & Fun</span>
            <span className="font-mono text-rose-400">${l.diningOutEntertainment}</span>
          </div>
          <input
            type="range"
            min={0}
            max={2000}
            step={50}
            value={l.diningOutEntertainment}
            onChange={(e) => updateField('diningOutEntertainment', parseFloat(e.target.value))}
            className="w-full accent-rose-500 cursor-pointer"
          />
        </div>

        {/* Health / Misc */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5">
          <div className="flex justify-between font-semibold text-slate-300">
            <span className="flex items-center gap-1.5"><HeartPulse className="w-3.5 h-3.5 text-teal-400" /> Health, Medical & Misc</span>
            <span className="font-mono text-teal-400">${l.healthMedical + l.otherMisc}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1500}
            step={50}
            value={l.healthMedical + l.otherMisc}
            onChange={(e) => updateField('healthMedical', parseFloat(e.target.value))}
            className="w-full accent-teal-500 cursor-pointer"
          />
        </div>

      </div>

      {/* Surplus Cash Banner */}
      <div className="bg-slate-950 border border-indigo-900/50 p-3.5 rounded-xl flex items-center justify-between flex-wrap gap-2 text-xs">
        <span className="text-slate-300 font-semibold flex items-center gap-2">
          <Calculator className="w-4 h-4 text-indigo-400" />
          <span>Surplus Cash Available for Housing (Net Income minus Lifestyle):</span>
        </span>
        <span className="font-mono font-extrabold text-indigo-300 text-sm">
          {fmt(surplusBeforeHousing)} / month
        </span>
      </div>

    </div>
  );
};
