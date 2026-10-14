import React, { useState, useEffect } from 'react';
import { RSUGrantConfig, SimulationOutput } from './types';
import { fetchSimulationData, POPULAR_TICKERS } from './services/stockApi';
import { runRsuSimulation } from './utils/rsuEngine';
import { Header } from './components/Header';
import { InputPanel } from './components/InputPanel';
import { SummaryCards } from './components/SummaryCards';
import { WealthChart } from './components/WealthChart';
import { StrategyTable } from './components/StrategyTable';
import { SectorComparison } from './components/SectorComparison';
import { GameMode } from './components/GameMode';
import { BehavioralGuide } from './components/BehavioralGuide';
import { Shield, Sparkles, Heart, AlertTriangle, Info, Calendar } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'game' | 'guide'>('dashboard');

  const [config, setConfig] = useState<RSUGrantConfig>({
    symbol: 'GOOGL',
    companyName: 'Alphabet Inc. (Google)',
    totalShares: 450,
    startYear: 2019,
    startQuarter: 1,
    hasCliff: true,
    vestingYears: 4,
    taxWithholdingPct: 35,
    statePreset: 'Custom',
    capitalGainsTaxPct: 20,
    cashApyPct: 4.5,
    sectorEtf: 'XLC',
    enableRefresherGrants: false,
    annualRefresherShares: 100,
  });

  const [simulation, setSimulation] = useState<SimulationOutput | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load and calculate simulation whenever config changes
  useEffect(() => {
    let isSubscribed = true;

    async function loadData() {
      setIsLoading(true);
      try {
        const datasets = await fetchSimulationData(config.symbol, config.sectorEtf);
        if (isSubscribed) {
          const result = runRsuSimulation(
            config,
            datasets.stock,
            datasets.sp500,
            datasets.nasdaq,
            datasets.sector
          );
          setSimulation(result);
        }
      } catch (err) {
        console.error('Error running RSU simulation:', err);
      } finally {
        if (isSubscribed) setIsLoading(false);
      }
    }

    loadData();

    return () => {
      isSubscribed = false;
    };
  }, [
    config.symbol,
    config.startYear,
    config.startQuarter,
    config.totalShares,
    config.hasCliff,
    config.vestingYears,
    config.taxWithholdingPct,
    config.statePreset,
    config.capitalGainsTaxPct,
    config.cashApyPct,
    config.sectorEtf,
    config.enableRefresherGrants,
    config.annualRefresherShares,
  ]);

  const handleSelectTicker = (symbol: string) => {
    const matched = POPULAR_TICKERS.find(t => t.symbol === symbol.toUpperCase());
    setConfig(prev => ({
      ...prev,
      symbol: symbol.toUpperCase(),
      companyName: matched ? matched.name : `${symbol.toUpperCase()} Inc.`,
      sectorEtf: matched ? matched.sectorEtf : 'XLK',
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <Header
        currentSymbol={config.symbol}
        onSelectTicker={handleSelectTicker}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8 w-full">
        {/* Config Input Controls */}
        <InputPanel
          config={config}
          onChangeConfig={setConfig}
          isLoading={isLoading}
        />

        {/* Tab 1: Dashboard */}
        {activeTab === 'dashboard' && simulation && (
          <div className="space-y-6 sm:space-y-8">
            {/* Synthetic Data Alert Banner */}
            {simulation.dataSources.stock === 'synthetic' && (
              <div className="bg-amber-950/30 border border-amber-800/50 rounded-xl px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs text-amber-200">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>
                    <strong>Modeled Data Mode:</strong> Displaying illustrative price history for {simulation.config.symbol} (live Yahoo Finance request timed out or was blocked by browser CORS).
                  </span>
                </div>
                <span className="self-start sm:self-auto text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-amber-900/60 text-amber-200 border border-amber-700/50">
                  Synthetic Curve
                </span>
              </div>
            )}

            {/* Future Vests Pending Notice */}
            {simulation.futureCount > 0 && simulation.vestedCount > 0 && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-2.5 flex items-center justify-between text-xs text-slate-400">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>
                    <strong className="text-slate-200">{simulation.vestedCount} of {simulation.events.length}</strong> quarterly vests have occurred. The remaining <strong className="text-slate-200">{simulation.futureCount}</strong> upcoming vests are pending and excluded from realized strategy totals.
                  </span>
                </div>
              </div>
            )}

            {/* Zero Vests State */}
            {simulation.vestedCount === 0 ? (
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 text-center space-y-3">
                <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto" />
                <h3 className="text-base font-bold text-white">No Vests Have Occurred Yet</h3>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  For a grant snapshot starting in {simulation.config.startYear} Q{simulation.config.startQuarter} {simulation.config.hasCliff ? 'with a 1-year cliff' : ''}, the first vest is scheduled for {simulation.events[0]?.date || 'a future date'}. Select an earlier grant year to analyze historical performance.
                </p>
                <div className="pt-2">
                  <StrategyTable simulation={simulation} />
                </div>
              </div>
            ) : (
              <>
                {/* KPI Cards */}
                <SummaryCards simulation={simulation} />

                {/* Wealth Trajectory Chart */}
                <WealthChart simulation={simulation} />

                {/* Sector & Strategy Matrix */}
                <SectorComparison simulation={simulation} />

                {/* Vesting Audit & Tax Table */}
                <StrategyTable simulation={simulation} />
              </>
            )}
          </div>
        )}

        {/* Tab 2: Interactive Game */}
        {activeTab === 'game' && simulation && (
          <GameMode simulation={simulation} />
        )}

        {/* Tab 3: Behavioral Guide */}
        {activeTab === 'guide' && (
          <BehavioralGuide />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-slate-900/80 border-t border-slate-800 py-8 text-center text-xs text-slate-400 mt-12">
        <div className="max-w-7xl mx-auto px-4 space-y-2">
          <div className="flex items-center justify-center gap-2 text-slate-300 font-semibold">
            <Shield className="w-4 h-4 text-emerald-400" /> Personal Lifestyle & Wealth Planner Series
          </div>
          <p className="max-w-2xl mx-auto text-slate-400">
            Designed for public company employees to make informed, evidence-based equity divestment decisions using historical market data. We do not collect, store, or sell your data.
          </p>
          <p className="text-slate-400 pt-2">
            Built with React 19, TypeScript, Tailwind CSS v4, and Recharts.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
