import React, { useState } from 'react';
import { SimulationOutput } from '../types';
import { Table, ChevronDown, ChevronUp, Download, Info } from 'lucide-react';

interface StrategyTableProps {
  simulation: SimulationOutput;
}

export const StrategyTable: React.FC<StrategyTableProps> = ({ simulation }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const displayedEvents = isExpanded ? simulation.events : simulation.events.slice(0, 5);
  const hasRefreshers = simulation.config.enableRefresherGrants;

  const exportCsv = () => {
    const headers = [
      'Quarter',
      'Date',
      'Status',
      'Initial Grant Shares',
      'Refresher Shares',
      'Total Gross Shares',
      'Tax Shares Withheld',
      'Net Shares',
      'Stock Price ($)',
      'Net Proceeds ($)',
      'S&P 500 Price ($)'
    ];
    const rows = simulation.events.map(e => {
      const vested = e.status === 'vested';
      return [
        e.quarterLabel,
        e.date,
        e.status,
        e.initialSharesVested,
        e.refresherSharesVested,
        e.grossSharesVested,
        e.taxSharesWithheld,
        e.netSharesVested,
        vested ? e.stockPriceAtVest : '',
        vested ? e.netProceedsAtVest : '',
        vested ? e.sp500PriceAtVest : '',
      ];
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${simulation.config.symbol}_RSU_Vesting_Schedule.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
            <Table className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Vesting Audit & Tax Breakdown</h3>
            <p className="text-xs text-slate-400">
              Quarterly vest events with {simulation.config.taxWithholdingPct}% income tax withholding
              {hasRefreshers ? ` (Includes +${simulation.config.annualRefresherShares} annual refresher shares/yr)` : ''}
              {simulation.futureCount > 0 ? ` · ${simulation.futureCount} upcoming vest${simulation.futureCount === 1 ? '' : 's'} excluded from results` : ''}
            </p>
          </div>
        </div>

        <button
          onClick={exportCsv}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white border border-slate-700 transition-all self-start sm:self-auto"
        >
          <Download className="w-3.5 h-3.5" /> Export Schedule CSV
        </button>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800 uppercase tracking-wider">
            <tr>
              <th className="py-3 px-3.5">Quarter</th>
              <th className="py-3 px-3.5">Vest Date</th>
              {hasRefreshers && <th className="py-3 px-3.5 text-right">Initial Grant</th>}
              {hasRefreshers && <th className="py-3 px-3.5 text-right text-emerald-400">Refresher</th>}
              <th className="py-3 px-3.5 text-right">Total Gross</th>
              <th className="py-3 px-3.5 text-right">Tax Withheld ({simulation.config.taxWithholdingPct}%)</th>
              <th className="py-3 px-3.5 text-right">Net Shares</th>
              <th className="py-3 px-3.5 text-right">{simulation.config.symbol} Price</th>
              <th className="py-3 px-3.5 text-right">Net Vest Value</th>
              <th className="py-3 px-3.5 text-right">S&P 500 Price</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80 font-medium">
            {displayedEvents.map(e => {
              const vested = e.status === 'vested';
              return (
                <tr key={e.quarterIndex} className={`hover:bg-slate-800/40 transition-colors ${vested ? '' : 'opacity-50'}`}>
                  <td className="py-3 px-3.5 font-bold text-emerald-400">{e.quarterLabel}</td>
                  <td className="py-3 px-3.5 text-slate-400">
                    {e.date}
                    {!vested && (
                      <span className="ml-1.5 text-[10px] font-bold uppercase text-amber-400 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded">
                        Upcoming
                      </span>
                    )}
                  </td>
                  {hasRefreshers && <td className="py-3 px-3.5 text-right text-slate-400">{e.initialSharesVested}</td>}
                  {hasRefreshers && <td className="py-3 px-3.5 text-right font-bold text-emerald-400">+{e.refresherSharesVested}</td>}
                  <td className="py-3 px-3.5 text-right text-slate-200">{e.grossSharesVested}</td>
                  <td className="py-3 px-3.5 text-right text-rose-400">-{e.taxSharesWithheld}</td>
                  <td className="py-3 px-3.5 text-right font-bold text-white">{e.netSharesVested}</td>
                  <td className="py-3 px-3.5 text-right text-slate-200">{vested ? `$${e.stockPriceAtVest}` : '—'}</td>
                  <td className="py-3 px-3.5 text-right font-extrabold text-emerald-400">
                    {vested ? formatCurrency(e.netProceedsAtVest) : '—'}
                  </td>
                  <td className="py-3 px-3.5 text-right text-slate-400">{vested ? `$${e.sp500PriceAtVest}` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition-colors"
        >
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          {isExpanded ? 'Show Fewer Quarters' : `Show All ${simulation.events.length} Vests`}
        </button>

        <span className="text-[11px] text-slate-400 flex items-center gap-1">
          <Info className="w-3 h-3 text-slate-400" /> Income tax is withheld on vest date at ordinary rate
        </span>
      </div>
    </div>
  );
};
