import React, { useState } from 'react';
import { RSUGrantConfig } from '../types';
import { POPULAR_TICKERS } from '../services/stockApi';
import { US_STATE_RSU_TAXES } from '../data/stateTaxes';
import { Sliders, Search, Calendar, Layers, Percent, DollarSign, ChevronDown, ChevronUp, PlusCircle, MapPin, Info } from 'lucide-react';

interface InputPanelProps {
  config: RSUGrantConfig;
  onChangeConfig: (newConfig: RSUGrantConfig) => void;
  isLoading: boolean;
}

export const InputPanel: React.FC<InputPanelProps> = ({
  config,
  onChangeConfig,
  isLoading,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchInput, setSearchInput] = useState(config.symbol);

  const handleTickerChange = (symbol: string) => {
    const upper = symbol.toUpperCase().trim();
    if (!upper) return;
    setSearchInput(upper);
    const matched = POPULAR_TICKERS.find(t => t.symbol === upper);
    onChangeConfig({
      ...config,
      symbol: upper,
      companyName: matched ? matched.name : `${upper} Inc.`,
      sectorEtf: matched ? matched.sectorEtf : 'XLK',
    });
  };

  const handleStateSelect = (stateCode: string) => {
    if (stateCode === 'Custom') {
      onChangeConfig({ ...config, statePreset: 'Custom' });
      return;
    }

    const stateInfo = US_STATE_RSU_TAXES[stateCode];
    if (stateInfo) {
      onChangeConfig({
        ...config,
        statePreset: stateCode as any,
        taxWithholdingPct: stateInfo.totalWithholdingPct,
      });
    }
  };

  const activeStateInfo = US_STATE_RSU_TAXES[config.statePreset];

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl relative overflow-hidden">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Grant Setup & State Tax Controls</h2>
            <p className="text-xs text-slate-400">Configure initial allotment, annual refreshers & state tax withholding</p>
          </div>
        </div>

        {isLoading && (
          <div className="flex items-center gap-2 text-xs text-emerald-400 font-semibold bg-emerald-950/60 px-3 py-1.5 rounded-full border border-emerald-800/50 animate-pulse">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span> Fetching Live Ticker...
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Ticker Search */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1">
            <Search className="w-3.5 h-3.5 text-emerald-400" /> Company Ticker
          </label>
          <input
            type="text"
            value={searchInput}
            onChange={e => setSearchInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleTickerChange(searchInput)}
            onBlur={() => handleTickerChange(searchInput)}
            placeholder="e.g. GOOGL, AAPL, MSFT"
            className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm font-semibold text-white focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all uppercase"
          />
        </div>

        {/* Initial Grant Shares */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1">
            <Layers className="w-3.5 h-3.5 text-emerald-400" /> Initial Grant RSU Shares
          </label>
          <input
            type="number"
            min="10"
            step="10"
            value={config.totalShares}
            onChange={e => onChangeConfig({ ...config, totalShares: Math.max(1, Number(e.target.value)) })}
            className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm font-semibold text-white focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all"
          />
        </div>

        {/* Start Year / Snapshot */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5 text-emerald-400" /> Grant Year Snapshot
          </label>
          <select
            value={config.startYear}
            onChange={e => onChangeConfig({ ...config, startYear: Number(e.target.value) })}
            className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm font-semibold text-white focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all"
          >
            {[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025].map(yr => (
              <option key={yr} value={yr}>
                {yr} Grant Snapshot
              </option>
            ))}
          </select>
        </div>

        {/* Cliff Schedule Toggle */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5">
            Initial Vesting Cliff
          </label>
          <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => onChangeConfig({ ...config, hasCliff: true })}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                config.hasCliff
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              1-Yr Cliff (25%)
            </button>
            <button
              type="button"
              onClick={() => onChangeConfig({ ...config, hasCliff: false })}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                !config.hasCliff
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              No Cliff (Quarterly)
            </button>
          </div>
        </div>
      </div>

      {/* State Tax Selection & Annual Refresher Controls */}
      <div className="mt-4 pt-4 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* State Tax Selector Dropdown & Chips */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1">
              <MapPin className="w-3.5 h-3.5 text-emerald-400" /> State Tax Withholding:
            </span>
            <span className="text-emerald-400 font-bold">{config.taxWithholdingPct}%</span>
          </label>

          <div className="flex items-center gap-2">
            <select
              value={config.statePreset}
              onChange={e => handleStateSelect(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs font-semibold text-white focus:outline-none focus:border-emerald-500 transition-all"
            >
              {Object.values(US_STATE_RSU_TAXES).map(st => (
                <option key={st.code} value={st.code}>
                  {st.name} ({st.code}) — {st.totalWithholdingPct}% Withholding
                </option>
              ))}
              <option value="Custom">Custom Rate</option>
            </select>
          </div>

          {activeStateInfo && (
            <p className="text-[11px] text-slate-400 mt-1.5 flex items-center gap-1">
              <Info className="w-3 h-3 text-slate-500 shrink-0" /> {activeStateInfo.notes}
            </p>
          )}
        </div>

        {/* Refresher Grants Controls */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1">
              <PlusCircle className="w-3.5 h-3.5 text-emerald-400" /> Stack Annual Refresher Grants:
            </span>
            <span className="text-emerald-400 font-bold">
              {config.enableRefresherGrants ? `+${config.annualRefresherShares} shares/yr` : 'Disabled'}
            </span>
          </label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onChangeConfig({ ...config, enableRefresherGrants: !config.enableRefresherGrants })}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold shrink-0 transition-all ${
                config.enableRefresherGrants
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'bg-slate-950 text-slate-400 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              {config.enableRefresherGrants ? 'Refresher Active' : '+ Add Refreshers'}
            </button>
            {config.enableRefresherGrants && (
              <input
                type="number"
                min="10"
                step="10"
                value={config.annualRefresherShares}
                onChange={e => onChangeConfig({ ...config, annualRefresherShares: Math.max(0, Number(e.target.value)) })}
                placeholder="Shares/yr (e.g. 100)"
                className="w-full bg-slate-950 border border-slate-700/80 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-emerald-500 transition-all"
              />
            )}
          </div>
        </div>
      </div>

      {/* Advanced Toggle */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs font-semibold text-slate-400 hover:text-emerald-400 flex items-center gap-1 transition-colors"
        >
          {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          {showAdvanced ? 'Hide Custom Sliders' : 'Customize Manual Tax Sliders & APY'}
        </button>

        <div className="text-xs text-slate-400 font-medium">
          Sector Benchmark ETF: <span className="text-emerald-400 font-bold">{config.sectorEtf}</span>
        </div>
      </div>

      {/* Advanced Drawer */}
      {showAdvanced && (
        <div className="mt-3 pt-3 border-t border-slate-800/60 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center justify-between">
              <span>Manual Tax Withholding Rate:</span>
              <span className="text-emerald-400 font-bold">{config.taxWithholdingPct}%</span>
            </label>
            <input
              type="range"
              min="15"
              max="50"
              value={config.taxWithholdingPct}
              onChange={e =>
                onChangeConfig({
                  ...config,
                  taxWithholdingPct: Number(e.target.value),
                  statePreset: 'Custom',
                })
              }
              className="w-full accent-emerald-500 bg-slate-950 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center justify-between">
              <span>Capital Gains Tax:</span>
              <span className="text-emerald-400 font-bold">{config.capitalGainsTaxPct}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="35"
              value={config.capitalGainsTaxPct}
              onChange={e => onChangeConfig({ ...config, capitalGainsTaxPct: Number(e.target.value) })}
              className="w-full accent-emerald-500 bg-slate-950 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center justify-between">
              <span>High Yield Cash APY:</span>
              <span className="text-emerald-400 font-bold">{config.cashApyPct}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="7"
              step="0.25"
              value={config.cashApyPct}
              onChange={e => onChangeConfig({ ...config, cashApyPct: Number(e.target.value) })}
              className="w-full accent-emerald-500 bg-slate-950 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>
  );
};
