export interface LifestyleExpenses {
  groceries: number;
  utilities: number;
  carPaymentInsurance: number;
  subscriptionsStreaming: number;
  diningOutEntertainment: number;
  healthMedical: number;
  otherMisc: number;
}

export type AssetCategory = 'primary_home' | 'second_home' | 'luxury_car' | 'yacht' | 'custom';

export interface UserHousingInputs {
  monthlyTakeHome: number; // Liquid net cash in hand per month
  payFrequency: 'monthly' | 'biweekly';
  state: string;

  // Selected Asset Category
  assetType: AssetCategory;

  // Rent Info
  isRenter: boolean;
  currentRent: number;

  // Target Buy Specs
  zipCode: string;
  cityName?: string;
  targetBeds: number;
  targetBaths: number;
  minSqFt: number;
  minLotSqFt: number;
  
  targetHomePrice: number;
  downPaymentPercent: number; // e.g. 20
  interestRate: number; // e.g. 6.5
  loanTermYears: number; // 15, 20, or 30 (default 30)
  
  propertyTaxRate: number; // e.g. 1.25 for 1.25%
  homeInsuranceAnnual: number;
  hasHoa: boolean; // default true/false
  hoaMonthly: number;
  includeMaintenanceInPiti: boolean; // default true
  maintenancePercentAnnual: number; // e.g. 1.0%
  customPmiPercent?: number; // optional custom PMI % override

  // Non-Housing Lifestyle Expenses
  lifestyle: LifestyleExpenses;

  // Homeownership Readiness Levers
  rainyDayBufferTarget?: number; // default $500 or $1000
  annualSalaryRaisePercent?: number; // default 3%
  monthlyDownPaymentSavings?: number; // monthly down payment savings while renting
}

export interface HomeReadinessPoint {
  year: number;
  calendarYear: number;
  monthlyTakeHome: number;
  totalDownPaymentSaved: number;
  loanAmount: number;
  monthlyPiti: number;
  lifestyleExpenses: number;
  monthlyCashflowSurplus: number; // Take Home - Lifestyle - PITI
  isReadyToBuy: boolean;
}

export interface MortgageBreakdown {
  loanAmount: number;
  downPaymentAmount: number;
  monthlyPrincipalAndInterest: number;
  monthlyPropertyTax: number;
  monthlyInsurance: number;
  monthlyPmi: number;
  monthlyHoa: number;
  monthlyMaintenance: number;
  totalMonthlyPiti: number;
}

export interface HousePoorAnalysis {
  monthlyNetTakeHome: number;
  totalLifestyleExpenses: number;
  surplusCashBeforeHousing: number;

  // Rainy Day Buffer Target
  rainyDayBufferTarget: number;

  // Buying metrics
  monthlyBuyHousingCost: number;
  leftoverCashBufferBuy: number;
  housingNetPercentBuy: number;
  isHousePoorBuy: boolean;

  // Renting metrics
  monthlyRentHousingCost: number;
  leftoverCashBufferRent: number;
  housingNetPercentRent: number;

  // Monthly savings of renting vs buying
  monthlyRentSavings: number; // buy cost - rent cost

  // Recommendation status
  verdictStatus: 'buy' | 'caution' | 'rent_recommended';
  verdictTitle: string;
  verdictMessage: string;

  // Readiness Roadmap
  isReadyToBuyToday: boolean;
  readinessYear: number; // 0 = today, 1 = year 1, etc.
  readinessTimeline: HomeReadinessPoint[];

  // 4-Stage Multi-Variable Model Metrics
  hedonicSpecMapping: HedonicSpecMapping;
  expenseOptimization: ExpenseOptimizationRecommendation;
  stressTestMetrics: StressTestMetrics;

  // 5 Institutional-Grade Layers
  institutional: InstitutionalAnalysis;

  // Safe Max Home Price
  maxSafeHomePrice: number;

  // Live MLS URL
  mlsSearchUrl: string;
}

export interface InstitutionalAnalysis {
  // Layer 1: Unrecoverable Costs
  unrecoverableBuyMonthly: number;
  unrecoverableRentMonthly: number;
  unrecoverableDeltaMonthly: number; // Buy Unrecoverable - Rent Unrecoverable
  capitalOpportunityCostMonthly: number;
  principalEquityPaydownMonthly: number; // Forced Savings

  // Layer 2: Dynamic Tax Shield
  taxShieldMonthlyRefund: number;
  afterTaxMonthlyPiti: number;

  // Layer 3: Crossover Horizon (T*)
  crossoverBreakEvenYear: number;

  // Layer 4: Terminal Net Worth NPV Differential (10-Yr)
  terminalNetWorthBuy10Yr: number;
  terminalNetWorthRent10Yr: number;
  terminalNetWorthDelta10Yr: number;

  // Layer 5: Monte Carlo Simulation
  monteCarloConfidenceScore: number; // e.g. 86 (%)
  monteCarloIterations: number; // 1000
}

export interface HedonicSpecMapping {
  affordableSqFt: number;
  estimatedBeds: number;
  estimatedBaths: number;
  pricePerSqFt: number;
  zipCode: string;
  cityName: string;
}

export interface CategoryTrimRecommendation {
  category: string;
  label: string;
  currentAmount: number;
  recommendedTrim: number;
  newAmount: number;
}

export interface ExpenseOptimizationRecommendation {
  monthlyPaymentGap: number;
  categoryTrims: CategoryTrimRecommendation[];
  totalTrimmed: number;
  canBridgeGap100Percent: boolean;
}

export interface StressTestMetrics {
  reserveBufferMonths: number;
  housingExpenseRatio: number; // HER = PITI / Gross
  riskLevel: 'low' | 'moderate' | 'house_poor';
  riskLabel: string;
}

export interface NetWorthProjectionPoint {
  year: number;
  buyNetWorth: number;      // Home Equity + Home Value Appreciation
  rentAndInvestNetWorth: number; // Invested Savings in S&P500 at 7% return
}
