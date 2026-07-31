/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import { UserFinancialInputs, PresetProfile, PayFrequency } from './types';
import { FINANCIAL_PRESETS } from './data/presets';
import { calculatePaycheckTaxBreakdown, getMaximizedInputs } from './utils/taxCalculator';
import { Header } from './components/Header';
import { IncomeAndTaxInputs } from './components/IncomeAndTaxInputs';
import { SankeyDiagram } from './components/SankeyDiagram';
import { TaxDissectionCard } from './components/TaxDissectionCard';
import { InvestmentControls } from './components/InvestmentControls';
import { PaycheckSummary } from './components/PaycheckSummary';
import { ChildWealthProjection } from './components/ChildWealthProjection';
import { TaxTipsModal } from './components/TaxTipsModal';
import { Sparkles, PieChart, Landmark, TrendingUp, Shield, Wallet, Baby } from 'lucide-react';

export default function App() {
  // Default to High Earner / Balanced preset ($5,200 biweekly = $135.2k/yr in CA)
  const [inputs, setInputs] = useState<UserFinancialInputs>({
    grossSalary: 5200,
    payFrequency: 'biweekly',
    filingStatus: 'single',
    state: 'CA',
    dependents: 1,
    age: 32,

    traditional401k: 750,
    traditional401kIsPercent: false,
    hsa: 165.38,
    hsaCoverage: 'single',
    fsa: 0,

    roth401k: 0,
    roth401kIsPercent: false,
    ira: 0,
    rothIra: 269.23, // $7,000 / 26
    plan529: 200,
    custodialAccount: 100,
    trumpAccount: 100,
    custodialIra: 0,
    esppPercent: 10,
    esppDiscountPercent: 15,

    companyMatchPercent: 100,
    companyMatchUpToPercent: 6,

    // Annual Bonus
    annualBonusPercent: 15,
    annualBonusIsPercent: true,
    annualBonusAmount: 0,
    includeBonusIn401k: true,

    dissectTaxesInSankey: false,
  });

  const [activePresetId, setActivePresetId] = useState<string>('tech_high_earner');
  const [isTipsOpen, setIsTipsOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'sankey' | 'taxes' | 'investments' | 'child_wealth'>('sankey');

  // Compute live tax and paycheck results
  const taxResult = useMemo(() => {
    return calculatePaycheckTaxBreakdown(inputs);
  }, [inputs]);

  // Input change handler
  const handleInputChange = (updated: Partial<UserFinancialInputs>) => {
    setInputs((prev) => ({ ...prev, ...updated }));
  };

  // Maximize all IRS allowable investments
  const handleMaximizeAll = () => {
    setInputs((prev) => getMaximizedInputs(prev));
  };

  // Preset selector
  const handleSelectPreset = (preset: PresetProfile) => {
    setActivePresetId(preset.id);
    setInputs((prev) => ({ ...prev, ...preset.inputs }));
  };

  // Pay frequency switcher
  const handleChangeFrequency = (freq: PayFrequency) => {
    setInputs((prev) => ({ ...prev, payFrequency: freq }));
  };

  // Reset to default
  const handleReset = () => {
    const defaultPreset = FINANCIAL_PRESETS[0];
    setActivePresetId(defaultPreset.id);
    setInputs((prev) => ({ ...prev, ...defaultPreset.inputs }));
  };

  // Toggle tax dissection in Sankey
  const handleToggleDissectTaxes = () => {
    setInputs((prev) => ({
      ...prev,
      dissectTaxesInSankey: !prev.dissectTaxesInSankey,
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased pb-16 selection:bg-indigo-500 selection:text-white">
      {/* App Header */}
      <Header
        activePresetId={activePresetId}
        onSelectPreset={handleSelectPreset}
        payFrequency={inputs.payFrequency}
        onChangeFrequency={handleChangeFrequency}
        onReset={handleReset}
        onOpenTips={() => setIsTipsOpen(true)}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        
        {/* Income & Tax Setup Section */}
        <IncomeAndTaxInputs
          inputs={inputs}
          onChange={handleInputChange}
          onMaximizeAll={handleMaximizeAll}
        />

        {/* Primary Interactive Sankey Diagram */}
        <SankeyDiagram
          inputs={inputs}
          taxResult={taxResult}
          onToggleDissectTaxes={handleToggleDissectTaxes}
        />

        {/* Navigation / Feature Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 pb-1 overflow-x-auto">
          <button
            onClick={() => setActiveTab('sankey')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl transition-all ${
              activeTab === 'sankey'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <PieChart className="w-4 h-4" />
            <span>Paycheck & Taxes Overview</span>
          </button>

          <button
            onClick={() => setActiveTab('taxes')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl transition-all ${
              activeTab === 'taxes'
                ? 'bg-rose-600 text-white shadow-lg shadow-rose-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Landmark className="w-4 h-4" />
            <span>Tax Dissection (Fed, State, SDI)</span>
          </button>

          <button
            onClick={() => setActiveTab('investments')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl transition-all ${
              activeTab === 'investments'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            <span>401k, HSA, Roth & ESPP</span>
          </button>

          <button
            onClick={() => setActiveTab('child_wealth')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl transition-all ${
              activeTab === 'child_wealth'
                ? 'bg-teal-600 text-white shadow-lg shadow-teal-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Baby className="w-4 h-4" />
            <span>Child 529 & Wealth Growth</span>
          </button>
        </div>

        {/* Tab Content Panels */}
        {activeTab === 'sankey' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <PaycheckSummary inputs={inputs} taxResult={taxResult} />
              <TaxDissectionCard
                inputs={inputs}
                taxResult={taxResult}
                onToggleDissectInSankey={handleToggleDissectTaxes}
              />
            </div>
            <div className="space-y-6">
              <InvestmentControls inputs={inputs} onChange={handleInputChange} />
            </div>
          </div>
        )}

        {activeTab === 'taxes' && (
          <div className="space-y-6">
            <TaxDissectionCard
              inputs={inputs}
              taxResult={taxResult}
              onToggleDissectInSankey={handleToggleDissectTaxes}
            />
            <PaycheckSummary inputs={inputs} taxResult={taxResult} />
          </div>
        )}

        {activeTab === 'investments' && (
          <div className="space-y-6">
            <InvestmentControls inputs={inputs} onChange={handleInputChange} />
            <PaycheckSummary inputs={inputs} taxResult={taxResult} />
          </div>
        )}

        {activeTab === 'child_wealth' && (
          <div className="space-y-6">
            <ChildWealthProjection inputs={inputs} taxResult={taxResult} />
            <InvestmentControls inputs={inputs} onChange={handleInputChange} />
          </div>
        )}

      </main>

      {/* Tax Tips Modal */}
      <TaxTipsModal isOpen={isTipsOpen} onClose={() => setIsTipsOpen(false)} />
    </div>
  );
}
