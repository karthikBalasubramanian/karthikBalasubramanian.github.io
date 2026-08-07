export interface LifestyleExpenses {
  groceries: number;
  utilities: number;
  carPaymentInsurance: number;
  subscriptionsStreaming: number;
  diningOutEntertainment: number;
  healthMedical: number;
  otherMisc: number;
}

export interface UserHousingInputs {
  monthlyTakeHome: number; // Liquid net cash in hand per month
  payFrequency: 'monthly' | 'biweekly';
  state: string;

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

  // Safe Max Home Price
  maxSafeHomePrice: number;

  // Live MLS URL
  mlsSearchUrl: string;
}

export interface NetWorthProjectionPoint {
  year: number;
  buyNetWorth: number;      // Home Equity + Home Value Appreciation
  rentAndInvestNetWorth: number; // Invested Savings in S&P500 at 7% return
}
