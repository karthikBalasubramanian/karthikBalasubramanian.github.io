import { useState, useEffect } from 'react';
import type { UserHousingInputs } from './types';
import { analyzeHousePoorStatus } from './utils/calculator';
import { Header } from './components/Header';
import { NetTakeHomeBanner } from './components/NetTakeHomeBanner';
import { AssetTypeSelector } from './components/AssetTypeSelector';
import { LifestyleBudgetInputs } from './components/LifestyleBudgetInputs';
import { PropertySearchInputs } from './components/PropertySearchInputs';
import { HousePoorDiagnosisCard } from './components/HousePoorDiagnosisCard';
import { BuyOptimizationEngine } from './components/BuyOptimizationEngine';
import { InstitutionalDecisionEngine } from './components/InstitutionalDecisionEngine';
import { HomeReadinessTimelineChart } from './components/HomeReadinessTimelineChart';
import { Step3ChildPlannerCTA } from './components/Step3ChildPlannerCTA';

export default function App() {
  const [fromPaycheckApp, setFromPaycheckApp] = useState(false);
  const [inputs, setInputs] = useState<UserHousingInputs>({
    monthlyTakeHome: 9048, // Default $9,048/mo net cash
    payFrequency: 'biweekly',
    state: 'CA',

    assetType: 'primary_home',

    isRenter: true,
    currentRent: 3000,

    zipCode: '95113',
    cityName: 'San Jose',
    targetBeds: 3,
    targetBaths: 2,
    minSqFt: 1800,
    minLotSqFt: 5000,

    targetHomePrice: 850000,
    downPaymentPercent: 20,
    interestRate: 6.5,
    loanTermYears: 30,
    propertyTaxRate: 1.25,
    homeInsuranceAnnual: 3800,
    hasHoa: false,
    hoaMonthly: 0,
    includeMaintenanceInPiti: true,
    maintenancePercentAnnual: 1.0,

    rainyDayBufferTarget: 500,
    annualSalaryRaisePercent: 3.0,

    lifestyle: {
      groceries: 1000,
      utilities: 350,
      carPaymentInsurance: 600,
      subscriptionsStreaming: 150,
      diningOutEntertainment: 500,
      healthMedical: 200,
      otherMisc: 200,
    },
  });

  // Auto-read URL query parameters passed from Microsite 1 (Paycheck Allocator)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const monthlyNetParam = params.get('monthlyNet') || params.get('net');
    const stateParam = params.get('state');

    if (monthlyNetParam) {
      const parsedNet = parseFloat(monthlyNetParam);
      if (!isNaN(parsedNet) && parsedNet > 0) {
        setInputs((prev) => ({
          ...prev,
          monthlyTakeHome: Math.round(parsedNet),
          state: stateParam ? stateParam.toUpperCase() : prev.state,
        }));
        setFromPaycheckApp(true);
      }
    }
  }, []);

  const handleInputChange = (updated: Partial<UserHousingInputs>) => {
    setInputs((prev) => ({ ...prev, ...updated }));
  };

  const analysis = analyzeHousePoorStatus(inputs);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased pb-20">
      {/* App Header with 3-Step Navigation */}
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        
        {/* Step 1 Handoff Data Banner */}
        <NetTakeHomeBanner
          inputs={inputs}
          onChange={handleInputChange}
          fromPaycheckApp={fromPaycheckApp}
        />

        {/* Big Purchase Asset Category Selector (Homes, 2nd Homes, Cars, Yachts) */}
        <AssetTypeSelector
          inputs={inputs}
          onChange={handleInputChange}
        />

        {/* Non-Housing Lifestyle Budget Inputs */}
        <LifestyleBudgetInputs
          inputs={inputs}
          onChange={handleInputChange}
        />

        {/* Property & Location Search Inputs */}
        <PropertySearchInputs
          inputs={inputs}
          onChange={handleInputChange}
        />

        {/* Primary Decision Engine: House Poor Diagnosis Card */}
        <HousePoorDiagnosisCard
          inputs={inputs}
          onChange={handleInputChange}
        />

        {/* 4-Stage Buy-Optimization & Sensitivity Engine */}
        <BuyOptimizationEngine
          inputs={inputs}
          analysis={analysis}
          onChange={handleInputChange}
        />

        {/* 5 Wall-Street Institutional Decision Engine */}
        <InstitutionalDecisionEngine
          inputs={inputs}
          analysis={analysis}
        />

        {/* Homeownership Readiness & Cashflow Roadmap Chart */}
        <HomeReadinessTimelineChart
          inputs={inputs}
          analysis={analysis}
        />

        {/* Step 3 Cue Banner -> Child Financial Investment Planner */}
        <Step3ChildPlannerCTA inputs={inputs} />

      </main>
    </div>
  );
}
