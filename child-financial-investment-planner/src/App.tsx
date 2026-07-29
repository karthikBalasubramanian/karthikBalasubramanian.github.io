import React, { useState, useMemo } from 'react';
import { ParentInputs, AccountId } from './types';
import { calculateOptimalAllocation, generateYearlyProjections } from './utils/financialCalculators';
import { Header } from './components/Header';
import { AccountComparisonMatrix } from './components/AccountComparisonMatrix';
import { PortfolioOptimizer } from './components/PortfolioOptimizer';
import { GrowthCharts } from './components/GrowthCharts';
import { RolloverMaximizer } from './components/RolloverMaximizer';
import { SpreadsheetGrid } from './components/SpreadsheetGrid';
import { AIAdvisorTab } from './components/AIAdvisorTab';
import { GitHubIntegration } from './components/GitHubIntegration';
import { Github, ExternalLink, Globe } from 'lucide-react';

export default function App() {
  const [inputs, setInputs] = useState<ParentInputs>({
    childCurrentAge: 0,
    targetAge1: 18,
    targetAge2: 60,
    monthlyContribution: 300,
    yearlyLumpSum: 0,
    primaryGoal: 'balanced_growth',
    parentMarginalTaxRate: 0.24,
    childEarnedIncome: 0,
    state: 'NY',
    customAllocations: {
      '529_plan': 45,
      trump_account: 30,
      custodial_roth_ira: 15,
      utma_ugma: 10,
      coverdell_esa: 0,
      taxable_brokerage: 0,
    },
  });

  const [activeTab, setActiveTab] = useState<string>('optimizer');

  const handleUpdateInputs = (updated: Partial<ParentInputs>) => {
    setInputs((prev) => ({ ...prev, ...updated }));
  };

  // Compute live financial metrics
  const allocationResults = useMemo(() => calculateOptimalAllocation(inputs), [inputs]);
  const projections = useMemo(() => generateYearlyProjections(inputs), [inputs]);

  const age18Row = projections.find((p) => p.age === 18) || projections[projections.length - 1];
  const age60Row = projections.find((p) => p.age === 60) || projections[projections.length - 1];

  // CSV Export Handler
  const handleExportCSV = () => {
    const headers = [
      'Child Age',
      'Year',
      'Annual Contribution ($)',
      'Total Contributed ($)',
      'Conservative Balance (5%)',
      'Moderate Balance (7.5%)',
      'Optimistic Balance (10%)',
      'Estimated Tax Saved ($)',
      '529 SECURE 2.0 Roth Accumulation at Age 60',
      'Trump Account IRA Rollover Accumulation at Age 60',
    ];

    const rows = projections.map((p) => [
      p.age,
      p.year,
      p.annualContribution,
      p.totalContributed,
      p.conservativeBalance,
      p.moderateBalance,
      p.optimisticBalance,
      p.taxSavingsEstimate,
      p.secure20RothRolloverAccumulation || 0,
      p.trumpAccountRolloverAccumulation || 0,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `Child_Financial_Investment_Plan_Age_${inputs.childCurrentAge}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans pb-16">
      {/* Header Bar */}
      <Header
        inputs={inputs}
        onUpdateInputs={handleUpdateInputs}
        onExportCSV={handleExportCSV}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        projectedAge18Mod={age18Row.moderateBalance}
        projectedAge60Mod={age60Row.moderateBalance}
      />

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeTab === 'optimizer' && (
          <PortfolioOptimizer
            inputs={inputs}
            onUpdateInputs={handleUpdateInputs}
            allocationResults={allocationResults}
          />
        )}

        {activeTab === 'comparison' && <AccountComparisonMatrix />}

        {activeTab === 'projections' && (
          <GrowthCharts projections={projections} inputs={inputs} />
        )}

        {activeTab === 'rollover' && <RolloverMaximizer inputs={inputs} />}

        {activeTab === 'spreadsheet' && (
          <SpreadsheetGrid
            inputs={inputs}
            onUpdateInputs={handleUpdateInputs}
            projections={projections}
            onExportCSV={handleExportCSV}
          />
        )}

        {activeTab === 'ai_advisor' && <AIAdvisorTab inputs={inputs} />}

        {activeTab === 'author_site' && <GitHubIntegration />}
      </main>

      {/* Global Application Footer with Integrated Website Link */}
      <footer className="mt-12 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 py-8 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-indigo-600 rounded flex items-center justify-center text-white font-bold text-[10px]">
              J
            </div>
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              Child Financial Investment Planner
            </span>
            <span className="text-slate-400">•</span>
            <span className="text-slate-400">529 &amp; Trump Account Optimization Engine</span>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <a
              href="https://karthikbalasubramanian.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 font-semibold text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              <Globe className="w-3.5 h-3.5" />
              karthikbalasubramanian.github.io
              <ExternalLink className="w-3 h-3 opacity-70" />
            </a>
            <a
              href="https://github.com/karthikBalasubramanian/karthikBalasubramanian.github.io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
            >
              <Github className="w-3.5 h-3.5" />
              GitHub Repository
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
