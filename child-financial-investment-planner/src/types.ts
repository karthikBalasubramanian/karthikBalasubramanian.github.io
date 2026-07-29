export type AccountId =
  | '529_plan'
  | 'trump_account'
  | 'custodial_roth_ira'
  | 'utma_ugma'
  | 'coverdell_esa'
  | 'taxable_brokerage';

export type InvestmentGoal =
  | 'education_focused'
  | 'balanced_growth'
  | 'long_term_wealth'
  | 'maximum_flexibility';

export interface ReferenceLink {
  title: string;
  url: string;
  source: string;
  description?: string;
}

export interface AccountInfo {
  id: AccountId;
  name: string;
  shortName: string;
  badge: string;
  color: string;
  bgColor: string;
  borderColor: string;
  annualLimit: string;
  taxTreatment: string;
  ageOfControl: string;
  flexibility: 'High' | 'Moderate' | 'Restricted' | 'Education Only';
  secure20Eligible: boolean;
  iraRolloverEligible: boolean;
  description: string;
  bestFor: string;
  pros: string[];
  cons: string[];
  keyRules: string[];
  officialLinks: ReferenceLink[];
  defaultAllocationPct: Record<InvestmentGoal, number>;
}

export interface ScenarioRate {
  name: 'Conservative' | 'Moderate' | 'Optimistic';
  cagr: number; // e.g. 0.05 for 5%
  color: string;
  description: string;
}

export interface ParentInputs {
  childCurrentAge: number; // 0 to 17
  targetAge1: number; // 18 or 22 (Education/Adulthood)
  targetAge2: number; // 60 (Retirement compounding horizon)
  monthlyContribution: number; // $ per month
  yearlyLumpSum: number; // $ per year extra
  primaryGoal: InvestmentGoal;
  parentMarginalTaxRate: number; // e.g., 0.24 for 24%
  childEarnedIncome: number; // $ earned income per year for Roth eligibility
  state: string; // U.S. State for tax deduction estimates
  customAllocations: Record<AccountId, number>; // percentages sum to 100
}

export interface YearProjectionRow {
  age: number;
  year: number;
  annualContribution: number;
  totalContributed: number;
  conservativeBalance: number;
  moderateBalance: number;
  optimisticBalance: number;
  conservativeGrowth: number;
  moderateGrowth: number;
  optimisticGrowth: number;
  accountBalances: Record<AccountId, { conservative: number; moderate: number; optimistic: number }>;
  taxSavingsEstimate: number;
  secure20RothRolloverAccumulation?: number;
  trumpAccountRolloverAccumulation?: number;
}

export interface AllocationResult {
  accountId: AccountId;
  accountName: string;
  percentage: number;
  monthlyAmount: number;
  annualAmount: number;
  projectedAge18Moderate: number;
  projectedAge60Moderate: number;
  recommendationReason: string;
}

export interface SpreadsheetCell {
  row: number;
  col: string;
  label: string;
  formula: string;
  value: number | string;
  unit: 'currency' | 'percent' | 'number' | 'text';
  editable: boolean;
  accountId?: AccountId;
}
