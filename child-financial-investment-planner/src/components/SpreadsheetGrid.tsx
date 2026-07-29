import React, { useState } from 'react';
import { ParentInputs, SpreadsheetCell, YearProjectionRow } from '../types';
import { buildSpreadsheetGrid } from '../utils/financialCalculators';
import { Download, Table, Edit3, Calculator, Sparkles, Filter } from 'lucide-react';

interface SpreadsheetGridProps {
  inputs: ParentInputs;
  onUpdateInputs: (updated: Partial<ParentInputs>) => void;
  projections: YearProjectionRow[];
  onExportCSV: () => void;
}

export const SpreadsheetGrid: React.FC<SpreadsheetGridProps> = ({
  inputs,
  onUpdateInputs,
  projections,
  onExportCSV,
}) => {
  const [selectedCellKey, setSelectedCellKey] = useState<string>('A1');
  const [filterQuery, setFilterQuery] = useState<string>('');

  const gridCells = buildSpreadsheetGrid(inputs, projections);
  const selectedCell = gridCells.find((c) => `${c.col}${c.row}` === selectedCellKey) || gridCells[0];

  const formatValue = (val: number | string, unit: string) => {
    if (typeof val === 'number') {
      if (unit === 'currency') {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
      }
      if (unit === 'percent') {
        return `${val}%`;
      }
      return val.toLocaleString();
    }
    return val;
  };

  const yearsTo18 = Math.max(0, 18 - inputs.childCurrentAge);

  const filteredProjections = projections.filter((p) => {
    if (!filterQuery) return true;
    return (
      p.age.toString().includes(filterQuery) ||
      p.year.toString().includes(filterQuery)
    );
  });

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
              <Table className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Interactive Financial Model
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
              Excel Sheet Cell View &amp; Formula Inspector
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Inspect underlying compound interest formulas, edit input cells live, or export directly to Microsoft Excel / Google Sheets.
            </p>
          </div>

          <button
            id="excel-download-top-button"
            onClick={onExportCSV}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs transition-colors shadow-xs shrink-0"
          >
            <Download className="w-4 h-4" /> Download (.CSV for Excel)
          </button>
        </div>

        {/* Excel Formula Bar */}
        <div className="mt-4 bg-slate-100 dark:bg-slate-800/80 p-3 rounded-xl border border-slate-200 dark:border-slate-700 flex items-center gap-3 font-mono text-xs">
          <div className="bg-white dark:bg-slate-900 px-3 py-1 rounded-md border border-slate-300 dark:border-slate-700 font-bold text-indigo-600 dark:text-indigo-400">
            {selectedCell ? `${selectedCell.col}${selectedCell.row}` : 'A1'}
          </div>
          <span className="text-slate-400 font-bold">fx =</span>
          <div className="flex-1 overflow-x-auto text-slate-900 dark:text-slate-100 font-semibold">
            {selectedCell ? selectedCell.formula : '=FV()'}
          </div>
          <div className="text-slate-500 dark:text-slate-400 text-[11px] font-sans">
            Value: <strong className="font-mono text-slate-900 dark:text-white">{selectedCell ? formatValue(selectedCell.value, selectedCell.unit) : ''}</strong>
          </div>
        </div>
      </div>

      {/* Primary KPI Excel Grid Table */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2">
          <Calculator className="w-4 h-4 text-indigo-600" /> Key Model Input &amp; Output Grid
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-200 dark:border-slate-800">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800 text-[11px] uppercase tracking-wider text-slate-400 font-semibold border-b border-slate-200 dark:border-slate-700">
                <th className="border border-slate-200 dark:border-slate-700 p-2 text-center w-12">#</th>
                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left">Col A (Variable / Metric)</th>
                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left">Col B (Formula / Projection)</th>
                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left">Col C (Target Outcome)</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2, 3, 4].map((rowNum) => {
                const rowCells = gridCells.filter((c) => c.row === rowNum);
                return (
                  <tr key={rowNum} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="border border-slate-200 dark:border-slate-800 p-2.5 text-center font-mono font-bold bg-slate-50 dark:bg-slate-800/60 text-slate-400">
                      {rowNum}
                    </td>
                    {['A', 'B', 'C'].map((colName) => {
                      const cell = rowCells.find((c) => c.col === colName);
                      if (!cell) return <td key={colName} className="border p-2" />;
                      const cellKey = `${cell.col}${cell.row}`;
                      const isSelected = selectedCellKey === cellKey;
                      return (
                        <td
                          key={colName}
                          onClick={() => setSelectedCellKey(cellKey)}
                          className={`border border-slate-200 dark:border-slate-800 p-3 cursor-pointer transition-all ${
                            isSelected
                              ? 'bg-indigo-50/80 dark:bg-indigo-950/60 ring-2 ring-indigo-500 text-slate-900 dark:text-white font-semibold'
                              : 'text-slate-800 dark:text-slate-200'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-[11px] text-slate-400 font-medium">{cell.label}:</span>
                            <span className="font-bold font-mono text-xs text-slate-900 dark:text-white">
                              {formatValue(cell.value, cell.unit)}
                            </span>
                          </div>
                          <div className="text-[10px] font-mono text-slate-400 mt-1">
                            {cell.formula}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Full Year-by-Year Schedule Spreadsheet */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Full Schedule Spreadsheet (Ages {inputs.childCurrentAge} to 60)
            </h3>
            <p className="text-xs text-slate-500">
              Detailed line-item record of contributions, market growth, and tax savings
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <input
              id="filter-schedule-input"
              type="text"
              placeholder="Search age or year..."
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800"
            />
          </div>
        </div>

        <div className="overflow-x-auto max-h-96 no-scrollbar">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800/90 text-[11px] uppercase tracking-widest text-slate-400 font-semibold border-b border-slate-200 dark:border-slate-700 z-10">
              <tr>
                <th className="p-2.5">Child Age</th>
                <th className="p-2.5">Year</th>
                <th className="p-2.5">Annual Invested</th>
                <th className="p-2.5">Total Principal</th>
                <th className="p-2.5 text-slate-600 dark:text-slate-300">Conservative (5%)</th>
                <th className="p-2.5 text-indigo-600 dark:text-indigo-400">Moderate (7.5%)</th>
                <th className="p-2.5 text-slate-600 dark:text-slate-300">Optimistic (10%)</th>
                <th className="p-2.5">Est. Tax Saved</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
              {filteredProjections.map((row) => (
                <tr
                  key={row.age}
                  className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 ${
                    row.age === 18 ? 'bg-indigo-50/70 dark:bg-indigo-950/40 font-bold' : ''
                  } ${row.age === 60 ? 'bg-indigo-50/40 dark:bg-indigo-950/20 font-bold' : ''}`}
                >
                  <td className="p-2.5 font-sans font-bold text-slate-900 dark:text-white">
                    Age {row.age} {row.age === 18 ? '🎓' : row.age === 60 ? '🏖️' : ''}
                  </td>
                  <td className="p-2.5 text-slate-500">{row.year}</td>
                  <td className="p-2.5 text-slate-700 dark:text-slate-300">${row.annualContribution.toLocaleString()}</td>
                  <td className="p-2.5 text-slate-700 dark:text-slate-300">${row.totalContributed.toLocaleString()}</td>
                  <td className="p-2.5 text-slate-600 dark:text-slate-400">${row.conservativeBalance.toLocaleString()}</td>
                  <td className="p-2.5 text-indigo-600 dark:text-indigo-400 font-bold">${row.moderateBalance.toLocaleString()}</td>
                  <td className="p-2.5 text-slate-600 dark:text-slate-400">${row.optimisticBalance.toLocaleString()}</td>
                  <td className="p-2.5 text-indigo-700 dark:text-indigo-300">${row.taxSavingsEstimate.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
