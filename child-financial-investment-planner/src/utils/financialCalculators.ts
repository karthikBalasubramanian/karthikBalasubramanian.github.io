import { ParentInputs, YearProjectionRow, AccountId, AllocationResult, SpreadsheetCell } from '../types';
import { ACCOUNT_DATA, SCENARIO_RATES } from '../data/accountData';

/**
 * Calculates compound interest given initial principal, monthly contribution, annual rate, and years
 */
export function calculateFV(principal: number, monthlyContrib: number, annualRate: number, years: number): number {
  if (years <= 0) return principal;
  const i = annualRate / 12;
  const n = years * 12;

  if (i === 0) {
    return principal + monthlyContrib * n;
  }

  const fvPrincipal = principal * Math.pow(1 + i, n);
  const fvContrib = monthlyContrib * ((Math.pow(1 + i, n) - 1) / i);
  return fvPrincipal + fvContrib;
}

/**
 * Calculates optimal account distribution percentages and dollar amounts
 */
export function calculateOptimalAllocation(inputs: ParentInputs): AllocationResult[] {
  const { monthlyContribution, yearlyLumpSum, primaryGoal, customAllocations, childEarnedIncome } = inputs;
  const totalMonthlyEquivalent = monthlyContribution + yearlyLumpSum / 12;

  // Check if user is overriding with custom percentages
  const customSum = Object.values(customAllocations).reduce((a: number, b: number) => a + b, 0);
  const useCustom = Math.abs(customSum - 100) < 0.1;

  const results: AllocationResult[] = [];

  (Object.keys(ACCOUNT_DATA) as AccountId[]).forEach((id) => {
    const acc = ACCOUNT_DATA[id];
    let pct = useCustom ? customAllocations[id] || 0 : acc.defaultAllocationPct[primaryGoal] || 0;

    // Adjust if child has no earned income for Custodial Roth IRA
    if (id === 'custodial_roth_ira' && childEarnedIncome <= 0 && !useCustom) {
      // Reallocate Roth IRA percentage to Trump account or 529
      pct = 0;
    }

    const monthlyAmount = Math.round((totalMonthlyEquivalent * pct) / 100);
    const annualAmount = monthlyAmount * 12;

    // Calculate projected growth at age 18 and 60
    const yearsTo18 = Math.max(0, 18 - inputs.childCurrentAge);
    const projectedAge18Moderate = calculateFV(0, monthlyAmount, SCENARIO_RATES.moderate.cagr, yearsTo18);

    // After 18, assume contributions stop or roll over to compound till 60
    const yearsTo60 = Math.max(0, 60 - inputs.childCurrentAge);
    let projectedAge60Moderate = 0;

    if (acc.iraRolloverEligible || id === 'custodial_roth_ira' || id === '529_plan') {
      // Compounding from age 18 to 60
      const balanceAt18 = projectedAge18Moderate;
      const yearsFrom18To60 = 60 - 18;
      projectedAge60Moderate = balanceAt18 * Math.pow(1 + SCENARIO_RATES.moderate.cagr, yearsFrom18To60);
    } else {
      projectedAge60Moderate = projectedAge18Moderate * Math.pow(1 + SCENARIO_RATES.moderate.cagr, 60 - 18);
    }

    let recommendationReason = '';
    if (id === '529_plan') {
      recommendationReason = 'Provides 100% tax-free growth for college/school + up to $35k SECURE 2.0 Roth rollover capability.';
    } else if (id === 'trump_account') {
      recommendationReason = 'Structured $5,000/yr cap for childhood growth with seamless rollover to IRA at age 18.';
    } else if (id === 'custodial_roth_ira') {
      recommendationReason = childEarnedIncome > 0
        ? `Ideal for child's $${childEarnedIncome}/yr earned income. Tax-free compounding for life!`
        : 'Requires child earned income (e.g. babysitting, modeling, family job) to unlock.';
    } else if (id === 'utma_ugma') {
      recommendationReason = 'Flexible spending for non-education expenses (car, sports, computer) with first $1,350 tax-free.';
    } else if (id === 'coverdell_esa') {
      recommendationReason = 'Tax-free for K-12 tuition with individual stock selection capability.';
    } else {
      recommendationReason = 'Allows parent to retain full control over funds past age 18/21.';
    }

    results.push({
      accountId: id,
      accountName: acc.name,
      percentage: Math.round(pct),
      monthlyAmount,
      annualAmount,
      projectedAge18Moderate: Math.round(projectedAge18Moderate),
      projectedAge60Moderate: Math.round(projectedAge60Moderate),
      recommendationReason,
    });
  });

  return results;
}

/**
 * Generates full year-by-year projections from current child age up to age 60
 */
export function generateYearlyProjections(inputs: ParentInputs): YearProjectionRow[] {
  const { childCurrentAge, monthlyContribution, yearlyLumpSum, customAllocations, primaryGoal } = inputs;
  const totalMonthly = monthlyContribution + yearlyLumpSum / 12;

  const customSum = Object.values(customAllocations).reduce((a: number, b: number) => a + b, 0);
  const useCustom = Math.abs(customSum - 100) < 0.1;

  const rows: YearProjectionRow[] = [];

  let consBalance = 0;
  let modBalance = 0;
  let optBalance = 0;
  let totalContributed = 0;

  const accountBalances: Record<AccountId, { conservative: number; moderate: number; optimistic: number }> = {
    '529_plan': { conservative: 0, moderate: 0, optimistic: 0 },
    'trump_account': { conservative: 0, moderate: 0, optimistic: 0 },
    'custodial_roth_ira': { conservative: 0, moderate: 0, optimistic: 0 },
    'utma_ugma': { conservative: 0, moderate: 0, optimistic: 0 },
    'coverdell_esa': { conservative: 0, moderate: 0, optimistic: 0 },
    'taxable_brokerage': { conservative: 0, moderate: 0, optimistic: 0 },
  };

  const currentYear = new Date().getFullYear();

  for (let age = childCurrentAge; age <= 60; age++) {
    const yearsElapsed = age - childCurrentAge;
    const isContributingYears = age < 18; // Contributions active from birth to 18

    const annualContribution = isContributingYears ? totalMonthly * 12 : 0;
    totalContributed += annualContribution;

    // Update each account
    (Object.keys(ACCOUNT_DATA) as AccountId[]).forEach((id) => {
      const acc = ACCOUNT_DATA[id];
      const pct = useCustom ? customAllocations[id] || 0 : acc.defaultAllocationPct[primaryGoal] || 0;
      const accMonthly = isContributingYears ? (totalMonthly * pct) / 100 : 0;

      // Calculate end of year balance for this account
      accountBalances[id].conservative = calculateFV(
        accountBalances[id].conservative,
        accMonthly,
        SCENARIO_RATES.conservative.cagr,
        1
      );
      accountBalances[id].moderate = calculateFV(
        accountBalances[id].moderate,
        accMonthly,
        SCENARIO_RATES.moderate.cagr,
        1
      );
      accountBalances[id].optimistic = calculateFV(
        accountBalances[id].optimistic,
        accMonthly,
        SCENARIO_RATES.optimistic.cagr,
        1
      );
    });

    // Sum overall portfolio balances
    consBalance = Object.values(accountBalances).reduce((sum: number, b) => sum + b.conservative, 0);
    modBalance = Object.values(accountBalances).reduce((sum: number, b) => sum + b.moderate, 0);
    optBalance = Object.values(accountBalances).reduce((sum: number, b) => sum + b.optimistic, 0);

    const conservativeGrowth = Math.max(0, consBalance - totalContributed);
    const moderateGrowth = Math.max(0, modBalance - totalContributed);
    const optimisticGrowth = Math.max(0, optBalance - totalContributed);

    // Tax savings estimate (assuming 20% average capital gains / income tax saved on growth)
    const taxSavingsEstimate = Math.round(moderateGrowth * inputs.parentMarginalTaxRate);

    // Rollover accumulations
    const trumpBalanceAt18Mod = accountBalances['trump_account'].moderate;
    const yearsSince18 = Math.max(0, age - 18);
    const trumpAccountRolloverAccumulation = trumpBalanceAt18Mod * Math.pow(1 + SCENARIO_RATES.moderate.cagr, yearsSince18);

    const secure20RothRolloverAccumulation = 35000 * Math.pow(1 + SCENARIO_RATES.moderate.cagr, yearsSince18);

    rows.push({
      age,
      year: currentYear + yearsElapsed,
      annualContribution: Math.round(annualContribution),
      totalContributed: Math.round(totalContributed),
      conservativeBalance: Math.round(consBalance),
      moderateBalance: Math.round(modBalance),
      optimisticBalance: Math.round(optBalance),
      conservativeGrowth: Math.round(conservativeGrowth),
      moderateGrowth: Math.round(moderateGrowth),
      optimisticGrowth: Math.round(optimisticGrowth),
      accountBalances: JSON.parse(JSON.stringify(accountBalances)),
      taxSavingsEstimate,
      secure20RothRolloverAccumulation: Math.round(secure20RothRolloverAccumulation),
      trumpAccountRolloverAccumulation: Math.round(trumpAccountRolloverAccumulation),
    });
  }

  return rows;
}

/**
 * Builds the Excel Sheet cells structure for the formula grid view
 */
export function buildSpreadsheetGrid(inputs: ParentInputs, projections: YearProjectionRow[]): SpreadsheetCell[] {
  const cells: SpreadsheetCell[] = [];
  const age18Row = projections.find((p) => p.age === 18) || projections[projections.length - 1];
  const age60Row = projections.find((p) => p.age === 60) || projections[projections.length - 1];

  cells.push({
    row: 1,
    col: 'A',
    label: 'Child Current Age',
    formula: 'USER_INPUT',
    value: inputs.childCurrentAge,
    unit: 'number',
    editable: true,
  });

  cells.push({
    row: 1,
    col: 'B',
    label: 'Monthly Contribution ($)',
    formula: 'USER_INPUT',
    value: inputs.monthlyContribution,
    unit: 'currency',
    editable: true,
  });

  cells.push({
    row: 1,
    col: 'C',
    label: 'Annual Contribution ($)',
    formula: '=B1*12',
    value: inputs.monthlyContribution * 12,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 2,
    col: 'A',
    label: 'Total Principal Invested (to Age 18)',
    formula: '=C1*(18-A1)',
    value: age18Row.totalContributed,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 2,
    col: 'B',
    label: 'Portfolio Value at Age 18 (Moderate 7.5%)',
    formula: '=FV(7.5%/12, (18-A1)*12, -B1, 0)',
    value: age18Row.moderateBalance,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 2,
    col: 'C',
    label: 'Growth at Age 18 (Moderate 7.5%)',
    formula: '=B2-A2',
    value: age18Row.moderateGrowth,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 3,
    col: 'A',
    label: 'Portfolio Value at Age 60 (Moderate 7.5%)',
    formula: '=B2*(1.075^42)',
    value: age60Row.moderateBalance,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 3,
    col: 'B',
    label: '529 SECURE 2.0 Roth IRA Rollover at Age 60',
    formula: '=35000*(1.075^42)',
    value: Math.round(35000 * Math.pow(1.075, 42)),
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 3,
    col: 'C',
    label: 'Trump Account IRA Rollover Value at Age 60',
    formula: '=TRUMP_AGE18_BAL*(1.075^42)',
    value: age60Row.trumpAccountRolloverAccumulation || 0,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 4,
    col: 'A',
    label: 'Conservative Portfolio (5.0% CAGR) at 18',
    formula: '=FV(5%/12, (18-A1)*12, -B1, 0)',
    value: age18Row.conservativeBalance,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 4,
    col: 'B',
    label: 'Optimistic Portfolio (10.0% CAGR) at 18',
    formula: '=FV(10%/12, (18-A1)*12, -B1, 0)',
    value: age18Row.optimisticBalance,
    unit: 'currency',
    editable: false,
  });

  cells.push({
    row: 4,
    col: 'C',
    label: 'Estimated Tax Dollars Saved (to Age 18)',
    formula: '=C2 * TAX_RATE',
    value: age18Row.taxSavingsEstimate,
    unit: 'currency',
    editable: false,
  });

  return cells;
}
