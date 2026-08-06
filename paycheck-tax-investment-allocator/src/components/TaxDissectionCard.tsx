import React from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { US_STATES } from '../data/taxRates';
import { Landmark, FileText, Percent, Shield, Info, ArrowUpRight, ChevronRight, PieChart } from 'lucide-react';

interface TaxDissectionCardProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
  onToggleDissectInSankey: () => void;
}

export const TaxDissectionCard: React.FC<TaxDissectionCardProps> = ({
  inputs,
  taxResult,
  onToggleDissectInSankey,
}) => {
  const stateInfo = US_STATES[inputs.state] || US_STATES.OTHER;
  const isBiweekly = inputs.payFrequency !== 'annual';
  const mul = isBiweekly ? 1 : 26;

  const fmt = (biweeklyVal: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(
      biweeklyVal * mul
    );

  const totalTaxes = taxResult.totalTaxesBiweekly * mul;
  const grossPay = taxResult.grossBiweekly * mul;

  const taxItems = [
    {
      id: 'federal',
      title: 'Federal Income Tax',
      amount: taxResult.federalTaxBiweekly * mul,
      percentageOfGross: taxResult.percentages.federalTax,
      percentageOfTax: totalTaxes > 0 ? ((taxResult.federalTaxBiweekly * mul) / totalTaxes) * 100 : 0,
      color: 'bg-rose-500',
      textColor: 'text-rose-400',
      description: 'Progressive IRS brackets after standard deduction',
    },
    {
      id: 'state',
      title: `${stateInfo.name} State Tax`,
      amount: taxResult.stateTaxBiweekly * mul,
      percentageOfGross: taxResult.percentages.stateTax,
      percentageOfTax: totalTaxes > 0 ? ((taxResult.stateTaxBiweekly * mul) / totalTaxes) * 100 : 0,
      color: 'bg-pink-500',
      textColor: 'text-pink-400',
      description: stateInfo.hasStateTax
        ? `${stateInfo.type === 'flat' ? 'Flat' : 'Progressive'} state bracket rates`
        : 'No state income tax!',
    },
    {
      id: 'social_security',
      title: 'Social Security (OASDI)',
      amount: taxResult.socialSecurityBiweekly * mul,
      percentageOfGross: taxResult.percentages.socialSecurity,
      percentageOfTax: totalTaxes > 0 ? ((taxResult.socialSecurityBiweekly * mul) / totalTaxes) * 100 : 0,
      color: 'bg-amber-500',
      textColor: 'text-amber-400',
      description: '6.2% FICA tax up to $176,100 annual wage cap',
    },
    {
      id: 'medicare',
      title: 'Medicare Tax',
      amount: taxResult.medicareBiweekly * mul,
      percentageOfGross: taxResult.percentages.medicare,
      percentageOfTax: totalTaxes > 0 ? ((taxResult.medicareBiweekly * mul) / totalTaxes) * 100 : 0,
      color: 'bg-orange-500',
      textColor: 'text-orange-400',
      description: '1.45% FICA + 0.9% additional for high earners',
    },
    {
      id: 'sdi',
      title: `${stateInfo.code} SDI / Paid Leave`,
      amount: taxResult.sdiBiweekly * mul,
      percentageOfGross: taxResult.percentages.sdi,
      percentageOfTax: totalTaxes > 0 ? ((taxResult.sdiBiweekly * mul) / totalTaxes) * 100 : 0,
      color: 'bg-yellow-500',
      textColor: 'text-yellow-400',
      description: stateInfo.hasSDI
        ? `State Disability / Paid Family Leave (${(stateInfo.sdiRate * 100).toFixed(2)}%)`
        : 'No employee SDI required in state',
    },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-rose-950 text-rose-400 border border-rose-800/50">
            <Landmark className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Tax Dissection Breakdown</h3>
            <p className="text-xs text-slate-400">
              Total Taxes: <span className="text-rose-400 font-bold font-mono">{fmt(taxResult.totalTaxesBiweekly)}</span> ({taxResult.percentages.taxes.toFixed(1)}% of gross)
            </p>
          </div>
        </div>

        <button
          onClick={onToggleDissectInSankey}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
        >
          <PieChart className="w-3.5 h-3.5 text-rose-400" />
          <span>{inputs.dissectTaxesInSankey ? 'In Sankey: Expanded' : 'Expand in Sankey'}</span>
        </button>
      </div>

      {/* Stacked Percentage Visual Bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Tax Distribution</span>
          <span className="font-mono text-slate-300">100% of Total Tax ({fmt(taxResult.totalTaxesBiweekly)})</span>
        </div>
        <div className="w-full h-3.5 bg-slate-950 rounded-full overflow-hidden flex p-0.5 border border-slate-800">
          {taxItems.map(
            (item) =>
              item.amount > 0 && (
                <div
                  key={item.id}
                  style={{ width: `${item.percentageOfTax}%` }}
                  className={`${item.color} h-full first:rounded-l-full last:rounded-r-full transition-all duration-300`}
                  title={`${item.title}: ${item.percentageOfTax.toFixed(1)}%`}
                />
              )
          )}
        </div>
      </div>

      {/* Tax Items Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
        {taxItems.map((item) => (
          <div
            key={item.id}
            className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3 hover:border-slate-700 transition-all space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                <span className="text-xs font-bold text-slate-200">{item.title}</span>
              </div>
              <span className={`text-xs font-bold font-mono ${item.textColor}`}>
                {fmt(item.amount / mul)}
              </span>
            </div>

            <div className="flex justify-between text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-900">
              <span>{item.percentageOfGross.toFixed(1)}% of Gross</span>
              <span className="text-slate-300">{item.percentageOfTax.toFixed(1)}% of Tax</span>
            </div>

            <p className="text-[10px] text-slate-400 leading-tight pt-0.5">{item.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
