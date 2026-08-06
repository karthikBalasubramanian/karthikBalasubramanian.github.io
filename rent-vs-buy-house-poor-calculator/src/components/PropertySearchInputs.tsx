import React from 'react';
import type { UserHousingInputs } from '../types';
import { lookupZipCode } from '../data/zipDatabase';
import { MapPin, Home, Bed, Bath, Maximize, Percent, DollarSign, Key } from 'lucide-react';

interface PropertySearchInputsProps {
  inputs: UserHousingInputs;
  onChange: (updated: Partial<UserHousingInputs>) => void;
}

export const PropertySearchInputs: React.FC<PropertySearchInputsProps> = ({
  inputs,
  onChange,
}) => {
  const zipData = lookupZipCode(inputs.zipCode, inputs.state);

  const handleZipChange = (zipStr: string) => {
    const clean = zipStr.trim();
    const lookedUp = lookupZipCode(clean, inputs.state);
    onChange({
      zipCode: clean,
      cityName: lookedUp.city,
      propertyTaxRate: lookedUp.propertyTaxRate,
    });
  };

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-6">
      
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <Home className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>Housing & Location Setup</span>
              <span className="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded">
                Target Specs & Rent Comparison
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Configure your current rent and target purchase property specs in your desired ZIP code.
            </p>
          </div>
        </div>
      </div>

      {/* Inputs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        
        {/* Current Rent */}
        <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl space-y-1.5">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Key className="w-3.5 h-3.5 text-amber-400" /> Current Monthly Rent
          </label>
          <div className="flex items-center gap-1 font-mono text-base font-bold text-amber-300">
            <span>$</span>
            <input
              type="number"
              value={inputs.currentRent}
              onChange={(e) => onChange({ currentRent: Math.max(0, parseFloat(e.target.value) || 0) })}
              className="bg-transparent text-amber-300 focus:outline-none w-full font-mono"
              placeholder="3000"
            />
            <span className="text-xs text-slate-400 font-normal">/mo</span>
          </div>
          <span className="text-[10px] text-slate-400 block">Baseline rent cost to evaluate buying against</span>
        </div>

        {/* ZIP Code */}
        <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl space-y-1.5">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <MapPin className="w-3.5 h-3.5 text-rose-400" /> Target ZIP Code
          </label>
          <div className="flex items-center gap-2 font-mono text-base font-bold text-rose-300">
            <input
              type="text"
              value={inputs.zipCode}
              onChange={(e) => handleZipChange(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-rose-300 focus:outline-none w-full font-mono uppercase"
              placeholder="95113"
            />
          </div>
          <span className="text-[10px] text-slate-300 block font-semibold">
            📍 {zipData.city}, {zipData.state} ({zipData.propertyTaxRate}% Property Tax)
          </span>
        </div>

        {/* Target Home Purchase Price */}
        <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl space-y-1.5">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <DollarSign className="w-3.5 h-3.5 text-emerald-400" /> Target Home Price
          </label>
          <div className="flex items-center gap-1 font-mono text-base font-bold text-emerald-300">
            <span>$</span>
            <input
              type="number"
              value={inputs.targetHomePrice}
              onChange={(e) => onChange({ targetHomePrice: Math.max(0, parseFloat(e.target.value) || 0) })}
              className="bg-transparent text-emerald-300 focus:outline-none w-full font-mono"
              placeholder="850000"
            />
          </div>
          <span className="text-[10px] text-slate-400 block font-mono">
            {fmt(inputs.targetHomePrice)} listing price
          </span>
        </div>

        {/* Down Payment % */}
        <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl space-y-1.5">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Percent className="w-3.5 h-3.5 text-purple-400" /> Down Payment (%)
          </label>
          <div className="flex items-center gap-1 font-mono text-base font-bold text-purple-300">
            <input
              type="number"
              value={inputs.downPaymentPercent}
              onChange={(e) => onChange({ downPaymentPercent: Math.min(100, Math.max(0, parseFloat(e.target.value) || 0)) })}
              className="bg-transparent text-purple-300 focus:outline-none w-16 font-mono"
              placeholder="20"
            />
            <span className="text-xs font-normal text-slate-400">%</span>
            <span className="text-xs text-slate-400 font-mono ml-auto">
              (${Math.round(inputs.targetHomePrice * (inputs.downPaymentPercent / 100)).toLocaleString()})
            </span>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {inputs.downPaymentPercent < 20 ? '⚠️ PMI applicable (<20%)' : '✓ No PMI required'}
          </span>
        </div>

      </div>

      {/* House Specs (Beds, Baths, SqFt, Interest Rate) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs pt-2">
        
        {/* Beds */}
        <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-xl flex items-center justify-between">
          <span className="text-slate-400 font-semibold flex items-center gap-1">
            <Bed className="w-3.5 h-3.5 text-indigo-400" /> Min Beds:
          </span>
          <select
            value={inputs.targetBeds}
            onChange={(e) => onChange({ targetBeds: parseInt(e.target.value) || 3 })}
            className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-indigo-300 text-xs px-2 py-0.5 focus:outline-none"
          >
            <option value={1}>1 Bed</option>
            <option value={2}>2 Beds</option>
            <option value={3}>3 Beds</option>
            <option value={4}>4 Beds</option>
            <option value={5}>5+ Beds</option>
          </select>
        </div>

        {/* Baths */}
        <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-xl flex items-center justify-between">
          <span className="text-slate-400 font-semibold flex items-center gap-1">
            <Bath className="w-3.5 h-3.5 text-cyan-400" /> Min Baths:
          </span>
          <select
            value={inputs.targetBaths}
            onChange={(e) => onChange({ targetBaths: parseInt(e.target.value) || 2 })}
            className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-cyan-300 text-xs px-2 py-0.5 focus:outline-none"
          >
            <option value={1}>1 Bath</option>
            <option value={2}>2 Baths</option>
            <option value={3}>3 Baths</option>
            <option value={4}>4+ Baths</option>
          </select>
        </div>

        {/* Min SqFt */}
        <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-xl flex items-center justify-between">
          <span className="text-slate-400 font-semibold flex items-center gap-1">
            <Maximize className="w-3.5 h-3.5 text-teal-400" /> Min SqFt:
          </span>
          <input
            type="number"
            value={inputs.minSqFt}
            onChange={(e) => onChange({ minSqFt: Math.max(0, parseInt(e.target.value) || 0) })}
            className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-teal-300 text-xs px-2 py-0.5 w-20 text-right focus:outline-none"
          />
        </div>

        {/* Interest Rate */}
        <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-xl flex items-center justify-between">
          <span className="text-slate-400 font-semibold flex items-center gap-1">
            <Percent className="w-3.5 h-3.5 text-amber-400" /> Mortgage Rate:
          </span>
          <div className="flex items-center gap-1 font-mono text-amber-300 font-bold text-xs">
            <input
              type="number"
              step={0.1}
              value={inputs.interestRate}
              onChange={(e) => onChange({ interestRate: Math.max(0, parseFloat(e.target.value) || 0) })}
              className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-amber-300 text-xs px-1.5 py-0.5 w-14 text-right focus:outline-none"
            />
            <span>%</span>
          </div>
        </div>

      </div>

    </div>
  );
};
