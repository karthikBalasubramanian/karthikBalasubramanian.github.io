import React from 'react';
import type { UserHousingInputs } from '../types';
import { lookupZipCode } from '../data/zipDatabase';
import { MapPin, Home, Bed, Maximize, Percent, DollarSign, Key, Calendar, Wrench, Building } from 'lucide-react';

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
              <span>Housing, Mortgage & PITI Customization</span>
              <span className="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded">
                User Adjustable PITI Components
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Customize loan term length (15 vs 30 yrs), HOA fees, property taxes, insurance & maintenance reserve.
            </p>
          </div>
        </div>
      </div>

      {/* Primary Inputs Grid */}
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
          <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between gap-1 flex-wrap">
            <span>{fmt(inputs.targetHomePrice)} listing price</span>
            {zipData.medianPrice && zipData.medianPrice > 0 && inputs.targetHomePrice !== zipData.medianPrice && (
              <button
                type="button"
                onClick={() => onChange({ targetHomePrice: zipData.medianPrice! })}
                className="text-[10px] text-emerald-400 hover:text-emerald-300 font-sans underline cursor-pointer"
                title="Click to set target price to Redfin median sale price for this ZIP"
              >
                Use Redfin Median (${zipData.medianPrice.toLocaleString()})
              </button>
            )}
          </div>
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

      {/* House Specs & Loan Term Customization */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs pt-2">
        
        {/* Loan Term Length Selector (15 vs 20 vs 30 Yrs) */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5 text-amber-400" /> Loan Term Length
          </label>
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
            {[15, 20, 30].map((term) => (
              <button
                key={term}
                onClick={() => onChange({
                  loanTermYears: term,
                  interestRate: term === 15 ? 5.75 : term === 20 ? 6.15 : 6.5
                })}
                className={`flex-1 py-1 rounded text-xs font-mono font-bold transition-all ${
                  inputs.loanTermYears === term
                    ? 'bg-amber-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {term} Yrs
              </button>
            ))}
          </div>
          <span className="text-[10px] text-slate-400 block">
            {inputs.loanTermYears === 15 ? '15-Yr Fixed (Faster payoff, lower rate)' : '30-Yr Fixed (Lower monthly payment)'}
          </span>
        </div>

        {/* Mortgage Interest Rate */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Percent className="w-3.5 h-3.5 text-amber-400" /> Interest Rate (%)
          </label>
          <div className="flex items-center gap-1 font-mono text-amber-300 font-bold text-base">
            <input
              type="number"
              step={0.1}
              value={inputs.interestRate}
              onChange={(e) => onChange({ interestRate: Math.max(0, parseFloat(e.target.value) || 0) })}
              className="bg-transparent text-amber-300 focus:outline-none w-full font-mono"
            />
            <span className="text-xs font-normal text-slate-400">%</span>
          </div>
          <span className="text-[10px] text-slate-400 block">Annual fixed rate</span>
        </div>

        {/* Beds & Baths */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Bed className="w-3.5 h-3.5 text-indigo-400" /> Beds & Baths Specs
          </label>
          <div className="flex items-center gap-2">
            <select
              value={inputs.targetBeds}
              onChange={(e) => onChange({ targetBeds: parseInt(e.target.value) || 3 })}
              className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-indigo-300 text-xs px-2 py-1 focus:outline-none flex-1"
            >
              <option value={1}>1 Bed</option>
              <option value={2}>2 Beds</option>
              <option value={3}>3 Beds</option>
              <option value={4}>4 Beds</option>
              <option value={5}>5+ Beds</option>
            </select>
            <select
              value={inputs.targetBaths}
              onChange={(e) => onChange({ targetBaths: parseInt(e.target.value) || 2 })}
              className="bg-slate-900 border border-slate-800 rounded font-mono font-bold text-cyan-300 text-xs px-2 py-1 focus:outline-none flex-1"
            >
              <option value={1}>1 Bath</option>
              <option value={2}>2 Baths</option>
              <option value={3}>3 Baths</option>
              <option value={4}>4+ Baths</option>
            </select>
          </div>
          <span className="text-[10px] text-slate-400 block">Target room count</span>
        </div>

        {/* Min SqFt */}
        <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1">
          <label className="block text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
            <Maximize className="w-3.5 h-3.5 text-teal-400" /> Target Home Size
          </label>
          <div className="flex items-center gap-1 font-mono text-teal-300 font-bold text-base">
            <input
              type="number"
              value={inputs.minSqFt}
              onChange={(e) => onChange({ minSqFt: Math.max(0, parseInt(e.target.value) || 0) })}
              className="bg-transparent text-teal-300 focus:outline-none w-full font-mono"
            />
            <span className="text-xs font-normal text-slate-400">sqft</span>
          </div>
          <span className="text-[10px] text-slate-400 block">Minimum square footage</span>
        </div>

      </div>

      {/* PITI Fine-Tuning Controls (HOA, Property Tax, Insurance & Maintenance Reserve) */}
      <div className="bg-slate-950 border border-slate-800/80 p-4 rounded-xl space-y-3">
        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Building className="w-4 h-4 text-purple-400" />
          <span>User-Adjustable PITI Components (HOA, Property Tax, Maintenance Reserve)</span>
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          
          {/* HOA Fee Customization */}
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-semibold">HOA Monthly Fee</span>
              <button
                onClick={() => onChange({ hasHoa: !(inputs.hasHoa ?? true) })}
                className={`text-[10px] font-bold px-2 py-0.5 rounded transition-all ${
                  (inputs.hasHoa ?? true)
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {(inputs.hasHoa ?? true) ? 'HOA Active' : 'No HOA ($0)'}
              </button>
            </div>
            {(inputs.hasHoa ?? true) && (
              <div className="flex items-center gap-1 font-mono text-purple-300 font-bold text-sm">
                <span>$</span>
                <input
                  type="number"
                  value={inputs.hoaMonthly}
                  onChange={(e) => onChange({ hoaMonthly: Math.max(0, parseFloat(e.target.value) || 0) })}
                  className="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-purple-300 focus:outline-none w-full font-mono"
                  placeholder="150"
                />
                <span className="text-xs text-slate-400 font-normal">/mo</span>
              </div>
            )}
          </div>

          {/* Maintenance Reserve Fund Customization */}
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-semibold flex items-center gap-1">
                <Wrench className="w-3.5 h-3.5 text-teal-400" /> Maintenance Reserve
              </span>
              <button
                onClick={() => onChange({ includeMaintenanceInPiti: !(inputs.includeMaintenanceInPiti ?? true) })}
                className={`text-[10px] font-bold px-2 py-0.5 rounded transition-all ${
                  (inputs.includeMaintenanceInPiti ?? true)
                    ? 'bg-teal-600 text-white'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {(inputs.includeMaintenanceInPiti ?? true) ? 'Reserve Active' : 'Exempt ($0)'}
              </button>
            </div>
            {(inputs.includeMaintenanceInPiti ?? true) && (
              <div className="flex items-center gap-2">
                <select
                  value={inputs.maintenancePercentAnnual}
                  onChange={(e) => onChange({ maintenancePercentAnnual: parseFloat(e.target.value) || 1.0 })}
                  className="bg-slate-950 border border-slate-800 rounded text-teal-300 font-mono font-bold text-xs px-2 py-1 w-full focus:outline-none"
                >
                  <option value={0.5}>0.5% / yr (${Math.round((inputs.targetHomePrice * 0.005) / 12)}/mo)</option>
                  <option value={1.0}>1.0% / yr (${Math.round((inputs.targetHomePrice * 0.01) / 12)}/mo)</option>
                  <option value={1.5}>1.5% / yr (${Math.round((inputs.targetHomePrice * 0.015) / 12)}/mo)</option>
                  <option value={2.0}>2.0% / yr (${Math.round((inputs.targetHomePrice * 0.02) / 12)}/mo)</option>
                </select>
              </div>
            )}
            <span className="text-[9px] text-slate-400 block leading-tight">
              *Out-of-pocket repairs reserve fund (roof/HVAC). Not held in escrow!
            </span>
          </div>

          {/* Property Tax Rate */}
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl space-y-1.5">
            <span className="text-slate-300 font-semibold block">Property Tax Rate (%)</span>
            <div className="flex items-center gap-1 font-mono text-rose-300 font-bold text-sm">
              <input
                type="number"
                step={0.05}
                value={inputs.propertyTaxRate}
                onChange={(e) => onChange({ propertyTaxRate: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-rose-300 focus:outline-none w-full font-mono"
              />
              <span className="text-xs font-normal text-slate-400">%</span>
            </div>
            <span className="text-[9px] text-slate-400 block">
              ${Math.round((inputs.targetHomePrice * (inputs.propertyTaxRate / 100)) / 12)}/mo (Managed by Escrow)
            </span>
          </div>

          {/* Home Insurance Annual */}
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl space-y-1.5">
            <span className="text-slate-300 font-semibold block">Home Insurance ($/yr)</span>
            <div className="flex items-center gap-1 font-mono text-cyan-300 font-bold text-sm">
              <span>$</span>
              <input
                type="number"
                value={inputs.homeInsuranceAnnual}
                onChange={(e) => onChange({ homeInsuranceAnnual: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-cyan-300 focus:outline-none w-full font-mono"
              />
              <span className="text-xs font-normal text-slate-400">/yr</span>
            </div>
            <span className="text-[9px] text-slate-400 block">
              ${Math.round(inputs.homeInsuranceAnnual / 12)}/mo (Managed by Escrow)
            </span>
          </div>

        </div>
      </div>

    </div>
  );
};
