import React from 'react';
import { UserFinancialInputs, FilingStatus } from '../types';
import { US_STATES } from '../data/taxRates';
import { DollarSign, MapPin, Users, Calendar, Zap, ShieldCheck, Sparkles, Building2 } from 'lucide-react';

interface IncomeAndTaxInputsProps {
  inputs: UserFinancialInputs;
  onChange: (updated: Partial<UserFinancialInputs>) => void;
  onMaximizeAll: () => void;
}

export const IncomeAndTaxInputs: React.FC<IncomeAndTaxInputsProps> = ({
  inputs,
  onChange,
  onMaximizeAll,
}) => {
  const currentStateInfo = US_STATES[inputs.state] || US_STATES.OTHER;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-indigo-950 text-indigo-400 border border-indigo-800/50">
            <DollarSign className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Paycheck & Location Inputs</h2>
            <p className="text-xs text-slate-400">Set your gross pay and state tax jurisdiction</p>
          </div>
        </div>

        {/* Maximize All Button */}
        <button
          onClick={onMaximizeAll}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-950/40 border border-emerald-400/30 transition-all transform hover:scale-[1.02] active:scale-[0.98]"
        >
          <Zap className="w-3.5 h-3.5 fill-current text-yellow-300" />
          <span>Maximize All IRS Accounts</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Salary Input */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span>Gross Salary ({inputs.payFrequency === 'annual' ? 'Annual' : 'Biweekly'})</span>
            <span className="text-[10px] text-indigo-400 font-mono">
              {inputs.payFrequency === 'biweekly'
                ? `$${(inputs.grossSalary * 26).toLocaleString()}/yr`
                : `$${Math.round(inputs.grossSalary / 26).toLocaleString()}/bw`}
            </span>
          </label>
          <div className="relative">
            <span className="absolute left-3 top-2.5 text-slate-500 font-bold">$</span>
            <input
              type="number"
              value={inputs.grossSalary || ''}
              onChange={(e) => onChange({ grossSalary: Math.max(0, parseFloat(e.target.value) || 0) })}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 pl-7 pr-3 text-sm font-mono text-white font-bold focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              placeholder="e.g. 5000"
            />
          </div>
        </div>

        {/* US State Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-300 flex items-center gap-1">
            <MapPin className="w-3.5 h-3.5 text-rose-400" />
            <span>State (Tax Rules & SDI)</span>
          </label>
          <select
            value={inputs.state}
            onChange={(e) => onChange({ state: e.target.value })}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 px-3 text-sm text-white font-medium focus:outline-none focus:border-indigo-500"
          >
            {Object.entries(US_STATES).map(([code, info]) => (
              <option key={code} value={code}>
                {info.name} ({code}) {info.hasStateTax ? '' : '- 0% State Tax'}
              </option>
            ))}
          </select>
          {currentStateInfo.notes && (
            <p className="text-[10px] text-slate-400 italic font-sans">{currentStateInfo.notes}</p>
          )}
        </div>

        {/* Filing Status */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-300 flex items-center gap-1">
            <Users className="w-3.5 h-3.5 text-purple-400" />
            <span>Filing Status</span>
          </label>
          <select
            value={inputs.filingStatus}
            onChange={(e) => onChange({ filingStatus: e.target.value as FilingStatus })}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 px-3 text-sm text-white font-medium focus:outline-none focus:border-indigo-500"
          >
            <option value="single">Single</option>
            <option value="married">Married Filing Jointly</option>
            <option value="head_of_household">Head of Household</option>
          </select>
        </div>

        {/* Dependents & Age */}
        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">Dependents</label>
            <input
              type="number"
              min={0}
              max={10}
              value={inputs.dependents}
              onChange={(e) => onChange({ dependents: parseInt(e.target.value) || 0 })}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 px-3 text-sm text-white font-mono font-medium focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">Age</label>
            <input
              type="number"
              min={18}
              max={90}
              value={inputs.age}
              onChange={(e) => onChange({ age: parseInt(e.target.value) || 30 })}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 px-3 text-sm text-white font-mono font-medium focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>

      </div>

      {/* Annual Bonus Section */}
      <div className="mt-4 pt-4 border-t border-slate-800/80 bg-gradient-to-r from-slate-950 via-amber-950/20 to-slate-950 rounded-xl p-4 border border-amber-900/30">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-amber-950 text-amber-400 border border-amber-800/50">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-xs font-bold text-amber-300 uppercase tracking-wider flex items-center gap-2">
                Annual One-Time Bonus
                <span className="text-[10px] font-semibold text-amber-200/80 bg-amber-900/50 px-2 py-0.5 rounded-full border border-amber-700/50">
                  +${Math.round(
                    (inputs.annualBonusIsPercent ?? true)
                      ? ((inputs.payFrequency === 'annual' ? inputs.grossSalary : inputs.grossSalary * 26) * (inputs.annualBonusPercent || 0)) / 100
                      : (inputs.annualBonusAmount || 0)
                  ).toLocaleString()} / year
                </span>
              </h3>
              <p className="text-[11px] text-slate-300">
                Lump-sum performance bonus paid once a year (taxed at supplemental withholding rates)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => onChange({ annualBonusPercent: 15, annualBonusIsPercent: true })}
              className={`text-[10px] font-bold px-2 py-1 rounded border transition-all ${
                (inputs.annualBonusIsPercent ?? true) && inputs.annualBonusPercent === 15
                  ? 'bg-amber-900 text-amber-200 border-amber-600'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              15% Target
            </button>
            <button
              onClick={() => onChange({ annualBonusPercent: 20, annualBonusIsPercent: true })}
              className={`text-[10px] font-bold px-2 py-1 rounded border transition-all ${
                (inputs.annualBonusIsPercent ?? true) && inputs.annualBonusPercent === 20
                  ? 'bg-amber-900 text-amber-200 border-amber-600'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              20%
            </button>
            <button
              onClick={() => onChange({ annualBonusPercent: 10, annualBonusIsPercent: true })}
              className={`text-[10px] font-bold px-2 py-1 rounded border transition-all ${
                (inputs.annualBonusIsPercent ?? true) && inputs.annualBonusPercent === 10
                  ? 'bg-amber-900 text-amber-200 border-amber-600'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              10%
            </button>
            <button
              onClick={() => onChange({ annualBonusPercent: 0, annualBonusIsPercent: true })}
              className={`text-[10px] font-bold px-2 py-1 rounded border transition-all ${
                inputs.annualBonusPercent === 0
                  ? 'bg-slate-700 text-white border-slate-500'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              0% (No Bonus)
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 mb-1">
              Bonus Amount ({inputs.annualBonusIsPercent ? '% of Gross Salary' : 'Fixed $'})
            </label>
            <div className="relative">
              <input
                type="number"
                min={0}
                max={inputs.annualBonusIsPercent ? 100 : 500000}
                step={inputs.annualBonusIsPercent ? 1 : 1000}
                value={inputs.annualBonusIsPercent ? inputs.annualBonusPercent : inputs.annualBonusAmount}
                onChange={(e) => {
                  const val = Math.max(0, parseFloat(e.target.value) || 0);
                  if (inputs.annualBonusIsPercent) {
                    onChange({ annualBonusPercent: val });
                  } else {
                    onChange({ annualBonusAmount: val });
                  }
                }}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg py-1.5 px-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-amber-500"
              />
              <span className="absolute right-3 top-1.5 text-slate-500 text-xs font-bold">
                {inputs.annualBonusIsPercent ? '%' : '$'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 pt-3 sm:pt-0">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={inputs.includeBonusIn401k ?? true}
                onChange={(e) => onChange({ includeBonusIn401k: e.target.checked })}
                className="sr-only peer"
              />
              <div className="w-8 h-4 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-amber-600"></div>
            </label>
            <span className="text-[11px] font-semibold text-slate-300">
              Deduct 401(k) % & Company Match from Bonus
            </span>
          </div>

          <div className="bg-amber-950/40 border border-amber-800/40 rounded-xl p-2.5 flex items-center justify-between">
            <div>
              <span className="text-[10px] uppercase font-bold text-amber-400 tracking-wider block">Bonus Gross Total</span>
              <span className="text-sm font-bold text-white font-mono">
                +${Math.round(
                  (inputs.annualBonusIsPercent ?? true)
                    ? ((inputs.payFrequency === 'annual' ? inputs.grossSalary : inputs.grossSalary * 26) * (inputs.annualBonusPercent || 0)) / 100
                    : (inputs.annualBonusAmount || 0)
                ).toLocaleString()} / year
              </span>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block">Est. Supplemental Tax</span>
              <span className="text-xs font-mono font-bold text-rose-400">
                ~22% Fed + State/FICA
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
