/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useRef } from 'react';
import { UserFinancialInputs, PresetProfile, PayFrequency } from './types';
import { FINANCIAL_PRESETS } from './data/presets';
import { calculatePaycheckTaxBreakdown, getMaximizedInputs } from './utils/taxCalculator';
import { Header } from './components/Header';
import { IncomeAndTaxInputs } from './components/IncomeAndTaxInputs';
import { SankeyDiagram } from './components/SankeyDiagram';
import { TaxDissectionCard } from './components/TaxDissectionCard';
import { PreTaxDeductionsCard, PostTaxAllocationsCard } from './components/InvestmentControls';
import { PaycheckSummary } from './components/PaycheckSummary';
import { ChildWealthProjection } from './components/ChildWealthProjection';
import { PaycheckScheduleTable } from './components/PaycheckScheduleTable';
import { Step2RentVsBuyCTA } from './components/Step2RentVsBuyCTA';
import { TaxTipsModal } from './components/TaxTipsModal';
import { Sparkles, PieChart, Landmark, TrendingUp, Shield, Wallet, Baby, Calendar } from 'lucide-react';

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
    bonusPayPeriodNumber: 4,

    dissectTaxesInSankey: false,
  });

  const [activePresetId, setActivePresetId] = useState<string>('tech_high_earner');
  const [isTipsOpen, setIsTipsOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'sankey' | 'schedule' | 'taxes' | 'investments' | 'child_wealth'>('sankey');
  
  const [isInputsCollapsed, setIsInputsCollapsed] = useState<boolean>(false);
  const [isFocusMode, setIsFocusMode] = useState<boolean>(false);
  const sankeyRef = useRef<HTMLDivElement | null>(null);

  // Compute live tax and paycheck results
  const taxResult = useMemo(() => {
    return calculatePaycheckTaxBreakdown(inputs);
  }, [inputs]);

  // Input change handler
  const handleInputChange = (updated: Partial<UserFinancialInputs>) => {
    setInputs((prev) => {
      const next = { ...prev, ...updated };
      // Auto-sync HSA coverage with filing status unless user explicitly provided hsaCoverage
      if (updated.filingStatus !== undefined && updated.hsaCoverage === undefined) {
        next.hsaCoverage = updated.filingStatus === 'married' ? 'family' : 'single';
      }
      return next;
    });
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
    setIsInputsCollapsed(false);
    setIsFocusMode(false);
  };

  // Toggle tax dissection in Sankey
  const handleToggleDissectTaxes = () => {
    setInputs((prev) => ({
      ...prev,
      dissectTaxesInSankey: !prev.dissectTaxesInSankey,
    }));
  };

  // Toggle optional post-tax allocations in Sankey
  const handleToggleIncludePostTax = () => {
    setInputs((prev) => ({
      ...prev,
      includePostTaxInSankey: !prev.includePostTaxInSankey,
    }));
  };

  // Action: Visualize flow (collapses inputs + scrolls to Sankey chart)
  const handleVisualizeFlow = () => {
    setIsInputsCollapsed(true);
    setIsFocusMode(true);
    setTimeout(() => {
      sankeyRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
  };

  // Action: Toggle focus mode
  const handleToggleFocusMode = () => {
    setIsFocusMode(!isFocusMode);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased pb-16">
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
          isCollapsed={isInputsCollapsed}
          onToggleCollapse={() => setIsInputsCollapsed(!isInputsCollapsed)}
          onVisualizeFlow={handleVisualizeFlow}
        />

        {/* Primary Interactive Sankey Diagram */}
        <SankeyDiagram
          inputs={inputs}
          taxResult={taxResult}
          onToggleDissectTaxes={handleToggleDissectTaxes}
          onToggleIncludePostTax={handleToggleIncludePostTax}
          isFocusMode={isFocusMode}
          onToggleFocusMode={handleToggleFocusMode}
          chartRef={sankeyRef}
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
            onClick={() => setActiveTab('schedule')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl transition-all ${
              activeTab === 'schedule'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Calendar className="w-4 h-4" />
            <span>26-Paycheck Sequencing Schedule</span>
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
          <div className="space-y-6">
            <PaycheckSummary inputs={inputs} taxResult={taxResult} />

            {/* Top Grid: Pre-Tax Deductions (Left) and Post-Tax Accounts (Right) Side-by-Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
              <PreTaxDeductionsCard inputs={inputs} onChange={handleInputChange} />
              <PostTaxAllocationsCard inputs={inputs} onChange={handleInputChange} />
            </div>

            {/* Middle Section: Full-Width Horizontal Tax Dissection Breakdown */}
            <TaxDissectionCard
              inputs={inputs}
              taxResult={taxResult}
              onToggleDissectInSankey={handleToggleDissectTaxes}
            />

            {/* Bottom Section: 26-Paycheck Chronological Schedule Table */}
            <PaycheckScheduleTable inputs={inputs} taxResult={taxResult} onChange={handleInputChange} />

            <Step2RentVsBuyCTA inputs={inputs} taxResult={taxResult} />
          </div>
        )}

        {activeTab === 'schedule' && (
          <div className="space-y-6">
            <PaycheckScheduleTable inputs={inputs} taxResult={taxResult} onChange={handleInputChange} />
            <PaycheckSummary inputs={inputs} taxResult={taxResult} />
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
