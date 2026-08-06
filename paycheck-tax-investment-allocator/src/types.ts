export type PayFrequency = 'biweekly' | 'semimonthly' | 'monthly' | 'annual';

export type FilingStatus = 'single' | 'married' | 'head_of_household';

export interface StateTaxInfo {
  code: string;
  name: string;
  hasStateTax: boolean;
  hasSDI: boolean;
  sdiRate: number; // e.g. 0.011 for 1.1%
  sdiMaxWage?: number; // Cap if any
  type: 'flat' | 'progressive' | 'none';
  flatRate?: number;
  brackets?: {
    single: { min: number; max: number; rate: number }[];
    married: { min: number; max: number; rate: number }[];
  };
  notes?: string;
}

export interface UserFinancialInputs {
  grossSalary: number; // Amount based on payFrequency
  payFrequency: PayFrequency;
  filingStatus: FilingStatus;
  state: string; // State code e.g. "CA", "NY", "TX"
  dependents: number;
  age: number;
  taxYear?: number; // e.g. 2024, 2025, 2026, 2027

  // Pre-tax account elections
  traditional401k: number;
  traditional401kIsPercent: boolean;
  hsa: number; // biweekly dollar
  hsaCoverage: 'single' | 'family';
  employerHsaAnnual?: number; // annual employer HSA contribution/seed (e.g. $850 single, $1,700 family)
  fsa: number; // biweekly dollar

  // Post-tax account elections
  roth401k: number;
  roth401kIsPercent: boolean;
  ira: number; // Traditional post-tax IRA
  rothIra: number;
  plan529: number; // 529 College Savings
  custodialAccount: number; // UTMA/UGMA / Child account
  trumpAccount: number; // Special child trust / savings
  custodialIra: number; // Child Custodial IRA
  esppPercent: number; // 0% to 25% (IRS $25,000 annual limit)
  esppDiscountPercent: number; // Typically 15%

  // 401(k) Employer Match
  companyMatchPercent: number; // e.g. 100 for 100%
  companyMatchUpToPercent: number; // e.g. 6 for 6% of salary

  // Annual Bonus
  annualBonusPercent: number; // e.g. 15 for 15%
  annualBonusIsPercent: boolean; // default true
  annualBonusAmount: number; // custom fixed amount if not percentage
  includeBonusIn401k: boolean; // default true
  includeBonusInHsa?: boolean; // default false
  includeBonusInEspp?: boolean; // default false
  bonusPayPeriodNumber?: number; // 1 to 26 (default 4 for mid-February)

  // Sankey controls
  dissectTaxesInSankey: boolean;
  includePostTaxInSankey?: boolean;
}

export interface PayPeriodDetail {
  periodNumber: number; // 1 to 26
  label: string; // e.g. "Paycheck #4 (Feb)"
  isBonusPeriod: boolean;

  grossSalary: number;
  grossBonus: number;
  totalGross: number;

  employee401k: number;
  employerMatch: number;
  hsa: number;
  fsa: number;
  totalPreTax: number;

  is401kCapHit: boolean;
  isHsaCapHit: boolean;
  isEsppCapHit: boolean;
  isSocialSecurityCapHit: boolean;

  taxableGross: number;
  federalTax: number;
  stateTax: number;
  socialSecurity: number;
  medicare: number;
  sdi: number;
  totalTaxes: number;

  rothIra: number;
  plan529: number;
  childAccounts: number;
  esppContribution: number;
  totalPostTax: number;

  netTakeHomePay: number;
  netTakeHomeAfterPostTax: number;

  // Cumulative YTD trackers
  ytdGross: number;
  ytd401kEmployee: number;
  ytdHsaTotal: number;
  ytdEsppPayroll: number;
  ytdSocialSecurityWages: number;
}

export interface PaycheckScheduleResult {
  periods: PayPeriodDetail[];
  bonusPeriodNumber: number;

  // Phase summary metrics
  earlyPhaseNetBiweekly: number; // Paychecks before maxing out 401k/HSA
  latePhaseNetBiweekly: number;  // Paychecks after 401k/HSA maxed out
  maxOutPayPeriod401k: number | null; // Pay period when 401k reaches $24,500
  maxOutPayPeriodHsa: number | null;  // Pay period when HSA reaches statutory cap
  maxOutPayPeriodSS: number | null;   // Pay period when SS cap ($176,100) reached
}

export interface TaxBreakdownResult {
  payPeriodsPerYear: number;
  grossAnnual: number;
  grossBiweekly: number;

  // Employer Company Match
  companyMatchBiweekly: number;
  companyMatchAnnual: number;
  employee401kPercent: number;
  total401kAccumulationBiweekly: number;
  total401kAccumulationAnnual: number;

  // Pre-tax
  preTax401kBiweekly: number;
  hsaBiweekly: number;
  employerHsaBiweekly: number;
  employerHsaAnnual: number;
  fsaBiweekly: number;
  preTaxDeductionsBiweekly: number;
  preTaxDeductionsAnnual: number;

  taxableGrossBiweekly: number;
  taxableGrossAnnual: number;

  // Taxes
  federalTaxBiweekly: number;
  federalTaxAnnual: number;
  stateTaxBiweekly: number;
  stateTaxAnnual: number;
  socialSecurityBiweekly: number;
  socialSecurityAnnual: number;
  medicareBiweekly: number;
  medicareAnnual: number;
  sdiBiweekly: number;
  sdiAnnual: number;
  totalTaxesBiweekly: number;
  totalTaxesAnnual: number;

  // Post-tax
  roth401kBiweekly: number;
  iraBiweekly: number;
  rothIraBiweekly: number;
  plan529Biweekly: number;
  custodialAccountBiweekly: number;
  trumpAccountBiweekly: number;
  custodialIraBiweekly: number;
  esppContributionBiweekly: number;
  esppDiscountGainBiweekly: number;
  postTaxContributionsBiweekly: number;
  postTaxContributionsAnnual: number;

  // Take home (Default Net Income = Gross - PreTax 401k/HSA - Taxes)
  netTakeHomePayBiweekly: number;
  netTakeHomePayAnnual: number;
  netTakeHomeAfterPostTaxBiweekly: number;
  netTakeHomeAfterPostTaxAnnual: number;

  // Annual Bonus Breakdown
  grossAnnualBonus: number;
  grossTotalAnnualWithBonus: number;
  bonus401kContribution: number;
  bonusHsaContribution: number;
  bonusEsppContribution: number;
  bonusCompanyMatch: number;
  bonusTaxableGross: number;
  bonusFederalTax: number;
  bonusStateTax: number;
  bonusFicaTax: number;
  bonusTotalTaxes: number;
  bonusNetTakeHome: number;
  totalCombinedNetAnnual: number; // Base net + bonus net
  totalCombinedWealthInvestedAnnual: number; // Regular investments + bonus 401k + match

  // 26-Paycheck Chronological Timeline Schedule
  schedule: PaycheckScheduleResult;

  // Percentages of gross
  percentages: {
    preTax: number;
    taxes: number;
    postTax: number;
    takeHome: number;

    federalTax: number;
    stateTax: number;
    socialSecurity: number;
    medicare: number;
    sdi: number;
  };
}

export interface SankeyNodeData {
  id: string;
  name: string;
  category: 'gross' | 'preTax' | 'taxes' | 'taxChild' | 'postTax' | 'takeHome';
  valueBiweekly: number;
  valueAnnual: number;
  percentageOfGross: number;
  color?: string;
}

export interface SankeyLinkData {
  source: string;
  target: string;
  value: number;
  formattedValue: string;
  percentage: number;
  color?: string;
}

export interface PresetProfile {
  id: string;
  name: string;
  description: string;
  badge: string;
  inputs: Partial<UserFinancialInputs>;
}
