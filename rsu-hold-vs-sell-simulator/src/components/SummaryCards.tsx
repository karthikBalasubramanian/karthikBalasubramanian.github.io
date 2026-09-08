import React from 'react';
import { SimulationOutput } from '../types';
import { DollarSign, TrendingUp, TrendingDown, ShieldAlert, Award, ArrowUpRight, ArrowDownRight, Users } from 'lucide-react';

interface SummaryCardsProps {
  simulation: SimulationOutput;
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({ simulation }) => {
  const hold = simulation.strategies.hold;
  const sp500 = simulation.strategies.sp500;
  const oppCost = simulation.opportunityCostOfHolding;
  const isHoldWinner = hold.finalValueToday >= sp500.finalValueToday;

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Pure Hold Card */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-sky-400 bg-sky-500/10 px-2.5 py-1 rounded-full border border-sky-500/20">
            Pure Hold Strategy
          </span>
          <span className="text-[11px] font-bold text-slate-400">100% {simulation.config.symbol}</span>
        </div>
        <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {formatCurrency(hold.finalValueToday)}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs">
          <span className="text-slate-400 font-medium">Net Return:</span>
          <span className={`font-bold flex items-center ${hold.returnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {hold.returnPct >= 0 ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5" /> : <ArrowDownRight className="w-3.5 h-3.5 mr-0.5" />}
            {hold.returnPct > 0 ? `+${hold.returnPct}%` : `${hold.returnPct}%`}
          </span>
        </div>
        <div className="mt-1 text-[11px] text-slate-400">
          Max Drawdown: <span className="text-rose-400 font-semibold">-{hold.maxDrawdownPct}%</span>
        </div>
      </div>

      {/* 2. Pure Sell & S&P 500 Card (Wisdom of Masses) */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-emerald-500/30 rounded-2xl p-5 shadow-lg shadow-emerald-950/20 relative overflow-hidden group hover:border-emerald-500/50 transition-all">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20 flex items-center gap-1">
            <Users className="w-3 h-3" /> Wisdom of Masses (SPY)
          </span>
          <span className="text-[11px] font-bold text-emerald-400">S&P 500</span>
        </div>
        <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {formatCurrency(sp500.finalValueToday)}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs">
          <span className="text-slate-400 font-medium">Net Return:</span>
          <span className="font-bold text-emerald-400 flex items-center">
            <ArrowUpRight className="w-3.5 h-3.5 mr-0.5" />
            +{sp500.returnPct}%
          </span>
        </div>
        <div className="mt-1 text-[11px] text-slate-400">
          Max Drawdown: <span className="text-amber-400 font-semibold">-{sp500.maxDrawdownPct}%</span>
        </div>
      </div>

      {/* 3. Opportunity Cost Card */}
      <div className={`bg-slate-900/80 backdrop-blur-md border rounded-2xl p-5 shadow-lg relative overflow-hidden transition-all ${
        isHoldWinner ? 'border-sky-500/30' : 'border-emerald-500/30'
      }`}>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-amber-400 bg-amber-500/10 px-2.5 py-1 rounded-full border border-amber-500/20">
            Opportunity Cost
          </span>
          <span className="text-[11px] text-slate-400 font-semibold">Divest vs Hold</span>
        </div>
        <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {formatCurrency(Math.abs(oppCost))}
        </div>
        <div className="mt-2 text-xs font-semibold">
          {isHoldWinner ? (
            <span className="text-sky-400 flex items-center gap-1">
              <TrendingUp className="w-3.5 h-3.5" /> Holding {simulation.config.symbol} produced higher returns
            </span>
          ) : (
            <span className="text-emerald-400 flex items-center gap-1">
              <TrendingUp className="w-3.5 h-3.5" /> S&P 500 Divestment produced higher returns
            </span>
          )}
        </div>
        <p className="mt-1 text-[11px] text-slate-400">
          {isHoldWinner
            ? `Holding outperformed market by +${Math.abs(simulation.opportunityCostPct)}%`
            : `Divesting to S&P 500 outperformed holding by +${Math.abs(simulation.opportunityCostPct)}%`}
        </p>
      </div>

      {/* 4. Single-Stock Concentration Risk Card */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-rose-400 bg-rose-500/10 px-2.5 py-1 rounded-full border border-rose-500/20 flex items-center gap-1">
            <ShieldAlert className="w-3 h-3" /> Concentration Risk
          </span>
          <span className={`text-xs font-extrabold px-2 py-0.5 rounded-md ${
            hold.concentrationRiskScore === 'Severe'
              ? 'bg-rose-950 text-rose-300 border border-rose-800'
              : hold.concentrationRiskScore === 'High'
              ? 'bg-amber-950 text-amber-300 border border-amber-800'
              : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
          }`}>
            {hold.concentrationRiskScore} Risk
          </span>
        </div>
        <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {hold.concentrationRiskPct}% <span className="text-xs font-normal text-slate-400">Single-Stock Exposure</span>
        </div>
        
        {/* Risk Gauge Bar */}
        <div className="w-full bg-slate-950 rounded-full h-2 mt-3 overflow-hidden border border-slate-800">
          <div
            className={`h-full rounded-full transition-all ${
              hold.concentrationRiskPct > 70
                ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                : 'bg-gradient-to-r from-emerald-500 to-amber-500'
            }`}
            style={{ width: `${hold.concentrationRiskPct}%` }}
          ></div>
        </div>

        <p className="mt-2 text-[11px] text-slate-400">
          {hold.maxDrawdownPct > 35
            ? `Single-stock experienced a severe -${hold.maxDrawdownPct}% drawdown peak-to-trough.`
            : `Single-stock max drawdown was -${hold.maxDrawdownPct}%.`}
        </p>
      </div>
    </div>
  );
};
