import React from 'react';
import { SimulationOutput, StrategyKey } from '../types';
import { Layers, PieChart, Shield, CheckCircle2 } from 'lucide-react';

interface SectorComparisonProps {
  simulation: SimulationOutput;
}

export const SectorComparison: React.FC<SectorComparisonProps> = ({ simulation }) => {
  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const strategyKeys: StrategyKey[] = ['hold', 'sp500', 'nasdaq', 'sector', 'hybrid5050', 'ltcg1yr', 'cash'];

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl">
      <div className="flex items-center gap-2.5 mb-5">
        <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
          <PieChart className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-lg font-bold text-white">Sector & Strategy Matrix Comparison</h3>
          <p className="text-xs text-slate-400">
            Comparing {simulation.config.symbol} against Industry Sector ({simulation.config.sectorEtf}) and Index Benchmarks · total return, dividends reinvested · best pick ranked after tax ({simulation.config.capitalGainsTaxPct}% LTCG)
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategyKeys.map(key => {
          const strat = simulation.strategies[key];
          const isBest = simulation.bestStrategyKey === key;

          return (
            <div
              key={key}
              className={`p-4 rounded-xl border relative transition-all ${
                isBest
                  ? 'bg-slate-900 border-emerald-500/60 shadow-lg shadow-emerald-950/30'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              {isBest && (
                <div className="absolute top-3 right-3 text-[10px] font-extrabold uppercase bg-emerald-500 text-slate-950 px-2 py-0.5 rounded-full flex items-center gap-1 shadow-sm">
                  <CheckCircle2 className="w-3 h-3" /> Best After Tax
                </div>
              )}

              <div className="flex items-center gap-2 mb-2">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: strat.color }}></span>
                <h4 className="text-sm font-bold text-white">{strat.name}</h4>
              </div>

              <p className="text-[11px] text-slate-400 min-h-[32px] mb-3">{strat.description}</p>

              <div className="text-xl font-extrabold text-white tracking-tight">
                {formatCurrency(strat.finalValueToday)}
              </div>

              <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-1.5 text-xs">
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-slate-400">After-Tax if Sold Today:</span>
                  <span className="font-bold text-white">{formatCurrency(strat.afterTaxValueToday)}</span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-slate-400">Total Net Return:</span>
                  <span className={`font-bold ${strat.returnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {strat.returnPct >= 0 ? `+${strat.returnPct}%` : `${strat.returnPct}%`}
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-slate-400">Max Drawdown:</span>
                  <span className="font-semibold text-rose-400">-{strat.maxDrawdownPct}%</span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-slate-400">Concentration Risk:</span>
                  <span className={`font-bold ${
                    strat.concentrationRiskScore === 'Severe'
                      ? 'text-rose-400'
                      : strat.concentrationRiskScore === 'High'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}>
                    {strat.concentrationRiskScore}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
