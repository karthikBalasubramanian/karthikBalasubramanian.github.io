import React from 'react';
import type { UserHousingInputs } from '../types';
import { analyzeHousePoorStatus, calculateMortgagePiti } from '../utils/calculator';
import { AlertTriangle, ShieldCheck, ShieldAlert, ExternalLink, Zap } from 'lucide-react';

interface HousePoorDiagnosisCardProps {
  inputs: UserHousingInputs;
}

export const HousePoorDiagnosisCard: React.FC<HousePoorDiagnosisCardProps> = ({ inputs }) => {
  const analysis = analyzeHousePoorStatus(inputs);
  const mortgage = calculateMortgagePiti(inputs);

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl text-slate-100 space-y-6">
      
      {/* Dynamic Verdict Banner Header */}
      <div
        className={`p-5 rounded-2xl border transition-all ${
          analysis.verdictStatus === 'rent_recommended'
            ? 'bg-gradient-to-r from-rose-950/90 via-slate-950 to-slate-950 border-rose-500/80 shadow-2xl shadow-rose-950'
            : analysis.verdictStatus === 'caution'
            ? 'bg-gradient-to-r from-amber-950/90 via-slate-950 to-slate-950 border-amber-500/80 shadow-2xl shadow-amber-950'
            : 'bg-gradient-to-r from-emerald-950/90 via-slate-950 to-slate-950 border-emerald-500/80 shadow-2xl shadow-emerald-950'
        }`}
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div
              className={`p-3 rounded-xl shrink-0 mt-1 ${
                analysis.verdictStatus === 'rent_recommended'
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                  : analysis.verdictStatus === 'caution'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              }`}
            >
              {analysis.verdictStatus === 'rent_recommended' ? (
                <ShieldAlert className="w-7 h-7" />
              ) : analysis.verdictStatus === 'caution' ? (
                <AlertTriangle className="w-7 h-7" />
              ) : (
                <ShieldCheck className="w-7 h-7" />
              )}
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <h2
                  className={`text-xl font-extrabold tracking-tight ${
                    analysis.verdictStatus === 'rent_recommended'
                      ? 'text-rose-400'
                      : analysis.verdictStatus === 'caution'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {analysis.verdictTitle}
                </h2>
              </div>
              <p className="text-xs sm:text-sm text-slate-200 leading-relaxed font-medium">
                {analysis.verdictMessage}
              </p>
            </div>
          </div>

          {/* Live Redfin / MLS Deep Link Button */}
          <a
            href={analysis.mlsSearchUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-rose-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg transition-all shrink-0 font-sans border border-rose-400/30"
          >
            <span>Search MLS Listings in {inputs.zipCode || '95113'}</span>
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>

      {/* 4 Core Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        
        {/* Monthly Net Take-Home */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-slate-400 uppercase font-bold text-[10px] block">Monthly Net Take-Home</span>
          <div className="text-xl font-extrabold font-mono text-emerald-400">
            {fmt(analysis.monthlyNetTakeHome)}
          </div>
          <span className="text-[10px] text-slate-400 block">Liquid cash in hand per month</span>
        </div>

        {/* Total Monthly PITI Housing Cost */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-slate-400 uppercase font-bold text-[10px] block">Buying Housing PITI Cost</span>
          <div className="text-xl font-extrabold font-mono text-rose-400">
            {fmt(analysis.monthlyBuyHousingCost)} <span className="text-xs text-slate-400 font-normal">/mo</span>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {analysis.housingNetPercentBuy.toFixed(1)}% of your Net Take-Home Pay
          </span>
        </div>

        {/* Leftover Cash Buffer (Buy) */}
        <div
          className={`border rounded-xl p-4 space-y-1 ${
            analysis.leftoverCashBufferBuy < 500
              ? 'bg-rose-950/30 border-rose-800/80 text-rose-300'
              : analysis.leftoverCashBufferBuy < 1500
              ? 'bg-amber-950/30 border-amber-800/80 text-amber-300'
              : 'bg-emerald-950/30 border-emerald-800/80 text-emerald-300'
          }`}
        >
          <span className="uppercase font-bold text-[10px] block text-slate-400">Leftover Cash Buffer (Buy)</span>
          <div className="text-xl font-extrabold font-mono">
            {fmt(analysis.leftoverCashBufferBuy)} <span className="text-xs font-normal text-slate-400">/mo</span>
          </div>
          <span className="text-[10px] block">
            {analysis.leftoverCashBufferBuy < 500
              ? '🚨 DANGER: Less than $500 buffer!'
              : '✓ Safe emergency cash buffer'}
          </span>
        </div>

        {/* Monthly Rent Comparison */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-slate-400 uppercase font-bold text-[10px] block">Renting Monthly Cost</span>
          <div className="text-xl font-extrabold font-mono text-amber-300">
            {fmt(analysis.monthlyRentHousingCost)} <span className="text-xs text-slate-400 font-normal">/mo</span>
          </div>
          <span className="text-[10px] text-slate-300 font-semibold block">
            {analysis.monthlyRentSavings > 0
              ? `Renting saves +${fmt(analysis.monthlyRentSavings)}/mo in cash flow!`
              : 'Buying is cheaper than rent!'}
          </span>
        </div>

      </div>

      {/* PITI Detailed Breakdown */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center justify-between">
          <span>PITI Mortgage & Housing Expense Breakdown (${fmt(analysis.monthlyBuyHousingCost)}/mo)</span>
          <span className="font-mono text-slate-400 text-[11px] font-normal">
            Target Price: {fmt(inputs.targetHomePrice || 850000)}
          </span>
        </h4>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono text-xs text-center">
          
          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">Principal & Interest</span>
            <span className="font-bold text-slate-200">{fmt(mortgage.monthlyPrincipalAndInterest)}</span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">Property Tax</span>
            <span className="font-bold text-rose-300">{fmt(mortgage.monthlyPropertyTax)}</span>
            <span className="text-[9px] text-slate-500 font-sans block">Escrow Paid</span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">Home Insurance</span>
            <span className="font-bold text-cyan-300">{fmt(mortgage.monthlyInsurance)}</span>
            <span className="text-[9px] text-slate-500 font-sans block">Escrow Paid</span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">PMI</span>
            <span className="font-bold text-purple-300">{fmt(mortgage.monthlyPmi)}</span>
            <span className="text-[9px] text-slate-500 font-sans block">{mortgage.monthlyPmi > 0 ? 'Escrow Paid' : 'None ($0)'}</span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">HOA Fee</span>
            <span className="font-bold text-amber-300">{fmt(mortgage.monthlyHoa)}</span>
            <span className="text-[9px] text-slate-500 font-sans block">Direct Pay</span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
            <span className="text-[10px] text-slate-400 font-sans block">Maintenance (1%)</span>
            <span className="font-bold text-teal-300">{fmt(mortgage.monthlyMaintenance)}</span>
            <span className="text-[9px] text-slate-500 font-sans block">Out-of-Pocket</span>
          </div>

        </div>

        {/* Escrow vs Out-of-Pocket Maintenance Reserve Note */}
        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between flex-wrap gap-2 font-sans">
          <span>
            🏛️ <strong>Escrow Account Notice:</strong> Escrow automatically collects <strong>Property Taxes</strong> ({fmt(mortgage.monthlyPropertyTax)}/mo) and <strong>Home Insurance</strong> ({fmt(mortgage.monthlyInsurance)}/mo).
          </span>
          <span className="text-teal-300">
            🛠️ <strong>Maintenance Fund ({fmt(mortgage.monthlyMaintenance)}/mo):</strong> Saved out-of-pocket (NOT in Escrow) for roof/HVAC repairs.
          </span>
        </div>
      </div>

      {/* Safe Max Home Purchase Price Recommendation */}
      <div className="bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950 border border-indigo-800/60 p-4 rounded-xl flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-300">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-extrabold text-indigo-400 block tracking-wider">
              Recommended Maximum Safe Home Purchase Price
            </span>
            <p className="text-xs text-slate-300">
              The highest home price in ZIP {inputs.zipCode || '95113'} you can buy while keeping a $1,500/mo cash buffer.
            </p>
          </div>
        </div>

        <div className="text-right font-mono">
          <span className="text-xl font-extrabold text-indigo-300 block">
            {fmt(analysis.maxSafeHomePrice)}
          </span>
          <span className="text-[10px] text-slate-400">
            vs {fmt(inputs.targetHomePrice)} target price
          </span>
        </div>
      </div>

      {/* Life-First Philosophy Banner */}
      <div className="bg-slate-950 border border-amber-800/40 p-4 rounded-xl space-y-2">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-xs">
          <span className="text-base">💡</span>
          <span className="uppercase tracking-wider">The Life-First Philosophy: Modest Home vs "Dream Home"</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed italic">
          "Never restrict your quality of life, retirement savings, or child's future just to buy a house. It is 100% fine to buy a modest, smaller house (or rent) in ZIP {inputs.zipCode || '95113'} if it preserves your peace of mind and keeps your 401(k) and child 529 fully funded."
        </p>
        {analysis.verdictStatus === 'rent_recommended' && (
          <div className="pt-2 text-xs font-semibold text-emerald-300 flex items-center gap-1.5 font-sans">
            <span>🏡 <strong>Right-Sized Actionable Choice:</strong> Scale down from {fmt(inputs.targetHomePrice)} to ~{fmt(analysis.maxSafeHomePrice)} (e.g. 3-bed / 1500 sqft instead of a 4-bed dream house) or rent to preserve your $1,500/mo cash cushion!</span>
          </div>
        )}
      </div>

    </div>
  );
};
