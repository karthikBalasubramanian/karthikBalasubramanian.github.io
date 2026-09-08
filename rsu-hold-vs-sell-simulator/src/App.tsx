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
import { Shield, Sparkles, Heart } from 'lucide-react';

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
    config.totalShares,
    config.hasCliff,
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
            {/* KPI Cards */}
            <SummaryCards simulation={simulation} />

            {/* Wealth Trajectory Chart */}
            <WealthChart simulation={simulation} />

            {/* Sector & Strategy Matrix */}
            <SectorComparison simulation={simulation} />

            {/* Vesting Audit & Tax Table */}
            <StrategyTable simulation={simulation} />
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
