import { UserFinancialInputs, TaxBreakdownResult, StateTaxInfo, SankeyNodeData, SankeyLinkData } from '../types';
import { getTaxLimitsForYear, US_STATES } from '../data/taxRates';

export function getPayPeriodsCount(frequency: string): number {
  switch (frequency) {
    case 'biweekly':
      return 26;
    case 'semimonthly':
      return 24;
    case 'monthly':
      return 12;
    case 'annual':
      return 1;
    default:
      return 26;
  }
}

export function calculatePaycheckTaxBreakdown(inputs: UserFinancialInputs): TaxBreakdownResult {
  const taxLimits = getTaxLimitsForYear(inputs.taxYear || 2026);
  const payPeriods = getPayPeriodsCount(inputs.payFrequency);

  // Normalize gross to Annual and Biweekly equivalent for consistent calculations
  let grossAnnual = 0;
  if (inputs.payFrequency === 'annual') {
    grossAnnual = inputs.grossSalary;
  } else {
    grossAnnual = inputs.grossSalary * payPeriods;
  }
  const grossBiweekly = grossAnnual / 26; // Always compute standard biweekly (26 per year)

  // 1. Pre-Tax Deductions
  let preTax401kBiweekly = 0;
  if (inputs.traditional401kIsPercent) {
    preTax401kBiweekly = (grossBiweekly * (inputs.traditional401k || 0)) / 100;
  } else {
    preTax401kBiweekly = inputs.traditional401k || 0;
  }
  // Cap 401k at IRS limit
  const age = inputs.age || 30;
  const max401kAnnual = age >= 50
    ? taxLimits.TRADITIONAL_401K_MAX + taxLimits.TRADITIONAL_401K_CATCHUP
    : taxLimits.TRADITIONAL_401K_MAX;
  
  if (preTax401kBiweekly * 26 > max401kAnnual) {
    preTax401kBiweekly = max401kAnnual / 26;
  }

  // Post-tax Roth 401k check for match calculation
  let roth401kBiweekly = 0;
  if (inputs.roth401kIsPercent) {
    roth401kBiweekly = (grossBiweekly * (inputs.roth401k || 0)) / 100;
  } else {
    roth401kBiweekly = inputs.roth401k || 0;
  }

  // 401(k) Employer Match Calculation with IRS §401(a)(17) Annual Compensation Limit
  const companyMatchPercent = inputs.companyMatchPercent ?? 100;
  const companyMatchUpToPercent = inputs.companyMatchUpToPercent ?? 6;

  const employee401kBiweekly = preTax401kBiweekly + roth401kBiweekly;
  const employee401kPercent = grossBiweekly > 0 ? (employee401kBiweekly / grossBiweekly) * 100 : 0;

  const eligibleMatchPercent = Math.min(employee401kPercent, companyMatchUpToPercent);

  // IRS §401(a)(17) caps compensation used for employer match at limit
  const eligibleSalaryForMatchAnnual = Math.min(grossAnnual, taxLimits.COMPENSATION_LIMIT_401K);

  const companyMatchAnnual = eligibleSalaryForMatchAnnual * (eligibleMatchPercent / 100) * (companyMatchPercent / 100);
  const companyMatchBiweekly = companyMatchAnnual / 26;

  const total401kAccumulationBiweekly = employee401kBiweekly + companyMatchBiweekly;
  const total401kAccumulationAnnual = total401kAccumulationBiweekly * 26;

  const employerHsaAnnual = inputs.employerHsaAnnual || 0;
  const employerHsaBiweekly = employerHsaAnnual / 26;

  const maxHsaStatutoryAnnual = inputs.hsaCoverage === 'family'
    ? taxLimits.HSA_FAMILY_MAX + (age >= 55 ? taxLimits.HSA_CATCHUP : 0)
    : taxLimits.HSA_SINGLE_MAX + (age >= 55 ? taxLimits.HSA_CATCHUP : 0);
  const maxEmployeeHsaAnnual = Math.max(0, maxHsaStatutoryAnnual - employerHsaAnnual);

  let hsaBiweekly = inputs.hsa || 0;
  if (hsaBiweekly * 26 > maxEmployeeHsaAnnual) {
    hsaBiweekly = maxEmployeeHsaAnnual / 26;
  }

  const fsaBiweekly = inputs.fsa || 0;

  const preTaxDeductionsBiweekly = preTax401kBiweekly + hsaBiweekly + fsaBiweekly;
  const preTaxDeductionsAnnual = preTaxDeductionsBiweekly * 26;

  // Taxable Gross Income (Federal/State income tax base)
  const taxableGrossAnnual = Math.max(0, grossAnnual - preTaxDeductionsAnnual);
  const taxableGrossBiweekly = taxableGrossAnnual / 26;

  // FICA Base (401k does NOT reduce FICA, but HSA & FSA do reduce FICA!)
  const ficaDeductionsAnnual = (hsaBiweekly + fsaBiweekly) * 26;
  const ficaGrossAnnual = Math.max(0, grossAnnual - ficaDeductionsAnnual);

  // 2. Federal Income Tax Calculation
  const stdDeduction = taxLimits.FEDERAL_STANDARD_DEDUCTION[inputs.filingStatus] || 15000;
  const federalTaxableBase = Math.max(0, taxableGrossAnnual - stdDeduction);

  const federalBrackets = taxLimits.FEDERAL_BRACKETS[inputs.filingStatus] || taxLimits.FEDERAL_BRACKETS.single;
  let federalIncomeTaxAnnual = 0;

  for (const b of federalBrackets) {
    if (federalTaxableBase > b.min) {
      const taxableInBracket = Math.min(federalTaxableBase, b.max) - b.min;
      federalIncomeTaxAnnual += taxableInBracket * b.rate;
    }
  }
  const federalIncomeTaxBiweekly = federalIncomeTaxAnnual / 26;

  // 3. State Income Tax & SDI Calculation
  const stateInfo = US_STATES[inputs.state] || US_STATES.OTHER;
  let stateIncomeTaxAnnual = 0;
  let sdiTaxAnnual = 0;

  if (stateInfo.hasStateTax) {
    if (stateInfo.type === 'flat' && stateInfo.flatRate) {
      stateIncomeTaxAnnual = taxableGrossAnnual * stateInfo.flatRate;
    } else if (stateInfo.brackets) {
      const brackets = stateInfo.brackets[inputs.filingStatus === 'married' ? 'married' : 'single'] || stateInfo.brackets.single;
      for (const b of brackets) {
        if (taxableGrossAnnual > b.min) {
          const taxableInBracket = Math.min(taxableGrossAnnual, b.max) - b.min;
          stateIncomeTaxAnnual += taxableInBracket * b.rate;
        }
      }
    }
  }

  // SDI Calculation
  if (stateInfo.hasSDI && stateInfo.sdiRate) {
    const sdiSubjectGross = stateInfo.sdiMaxWage
      ? Math.min(ficaGrossAnnual, stateInfo.sdiMaxWage)
      : ficaGrossAnnual;
    sdiTaxAnnual = sdiSubjectGross * stateInfo.sdiRate;
  }

  const federalTaxAnnual = federalIncomeTaxAnnual;
  const federalTaxBiweekly = federalIncomeTaxBiweekly;
  const stateTaxAnnual = stateIncomeTaxAnnual;
  const stateTaxBiweekly = stateIncomeTaxAnnual / 26;
  const sdiAnnual = sdiTaxAnnual;
  const sdiBiweekly = sdiTaxAnnual / 26;

  // FICA Taxes (Social Security 6.2% + Medicare 1.45% + Add'l Medicare 0.9%)
  const ssSubjectAnnual = Math.min(ficaGrossAnnual, taxLimits.SOCIAL_SECURITY_WAGE_CAP);
  const socialSecurityAnnual = ssSubjectAnnual * taxLimits.SOCIAL_SECURITY_RATE;
  const socialSecurityBiweekly = socialSecurityAnnual / 26;

  let medicareAnnual = ficaGrossAnnual * taxLimits.MEDICARE_RATE;
  const addMedicareThreshold = inputs.filingStatus === 'married'
    ? taxLimits.ADDITIONAL_MEDICARE_THRESHOLD_MARRIED
    : taxLimits.ADDITIONAL_MEDICARE_THRESHOLD_SINGLE;

  if (ficaGrossAnnual > addMedicareThreshold) {
    medicareAnnual += (ficaGrossAnnual - addMedicareThreshold) * taxLimits.ADDITIONAL_MEDICARE_RATE;
  }
  const medicareBiweekly = medicareAnnual / 26;

  // Total Taxes
  const totalTaxesBiweekly = federalTaxBiweekly + stateTaxBiweekly + socialSecurityBiweekly + medicareBiweekly + sdiBiweekly;
  const totalTaxesAnnual = totalTaxesBiweekly * 26;

  // 7. Post-Tax Contributions (roth401kBiweekly calculated above for match)
  const iraBiweekly = inputs.ira || 0;
  const rothIraBiweekly = inputs.rothIra || 0;
  const plan529Biweekly = inputs.plan529 || 0;
  const custodialAccountBiweekly = inputs.custodialAccount || 0;
  const trumpAccountBiweekly = inputs.trumpAccount || 0;
  const custodialIraBiweekly = inputs.custodialIra || 0;

  // ESPP (Capped at 25% of paycheck and $25,000 annual IRS FMV stock purchase limit = $21,250 payroll contribution with 15% discount)
  const esppPct = Math.min(Math.max(0, inputs.esppPercent || 0), 25);
  const uncappedEsppBiweekly = (grossBiweekly * esppPct) / 100;
  const discountFrac = (inputs.esppDiscountPercent || 15) / 100;
  const maxEsppAnnualPayroll = 25000 * (1 - discountFrac); // $21,250 for 15% discount
  const maxEsppBiweekly = maxEsppAnnualPayroll / 26; // ~$817.31 biweekly
  const esppContributionBiweekly = Math.min(uncappedEsppBiweekly, maxEsppBiweekly);

  // ESPP Discount Gain benefit (e.g. 15% discount gives ~17.6% instant return on contribution)
  const esppDiscountGainBiweekly = esppContributionBiweekly > 0
    ? esppContributionBiweekly * (discountFrac / (1 - discountFrac))
    : 0;

  const postTaxContributionsBiweekly =
    roth401kBiweekly +
    iraBiweekly +
    rothIraBiweekly +
    plan529Biweekly +
    custodialAccountBiweekly +
    trumpAccountBiweekly +
    custodialIraBiweekly +
    esppContributionBiweekly;

  const postTaxContributionsAnnual = postTaxContributionsBiweekly * 26;

  // Default Net Income = Gross Income - Pre-Tax Deductions (401k, HSA, FSA) - Taxes
  const netTakeHomePayBiweekly = Math.max(0, grossBiweekly - preTaxDeductionsBiweekly - totalTaxesBiweekly);
  const netTakeHomePayAnnual = netTakeHomePayBiweekly * 26;

  // Net Cash after optional Post-Tax Allocations (Roth 401k/IRA, 529, Child Accounts, ESPP)
  const netTakeHomeAfterPostTaxBiweekly = Math.max(0, netTakeHomePayBiweekly - postTaxContributionsBiweekly);
  const netTakeHomeAfterPostTaxAnnual = netTakeHomeAfterPostTaxBiweekly * 26;

  // 8. Annual Bonus Calculations
  const annualBonusIsPercent = inputs.annualBonusIsPercent ?? true;
  const annualBonusPercent = inputs.annualBonusPercent ?? 0;
  const grossAnnualBonus = annualBonusIsPercent
    ? grossAnnual * (annualBonusPercent / 100)
    : (inputs.annualBonusAmount || 0);
  const grossTotalAnnualWithBonus = grossAnnual + grossAnnualBonus;

  const includeBonusIn401k = inputs.includeBonusIn401k ?? true;
  let bonus401kContribution = 0;
  let bonusCompanyMatch = 0;

  if (includeBonusIn401k && grossAnnualBonus > 0) {
    const remaining401kCapacity = Math.max(0, max401kAnnual - (preTax401kBiweekly * 26));
    const desiredBonus401k = grossAnnualBonus * (employee401kPercent / 100);
    bonus401kContribution = Math.min(desiredBonus401k, remaining401kCapacity);

    const eligibleMatchPercent = Math.min(employee401kPercent, companyMatchUpToPercent);

    // Total compensation eligible for matching across salary + bonus is capped at IRS limit
    const remainingMatchCompCapacity = Math.max(0, taxLimits.COMPENSATION_LIMIT_401K - eligibleSalaryForMatchAnnual);
    const eligibleBonusForMatch = Math.min(grossAnnualBonus, remainingMatchCompCapacity);

    bonusCompanyMatch = eligibleBonusForMatch * (eligibleMatchPercent / 100) * (companyMatchPercent / 100);
  }

  const bonusTaxableGross = Math.max(0, grossAnnualBonus - bonus401kContribution);

  // Bonus Federal Supplemental Tax (22% standard IRS supplemental rate)
  const bonusFederalTax = bonusTaxableGross * 0.22;

  // Bonus State Tax (Supplemental state rate or flat rate, e.g., CA ~10.23% or flat rate)
  let bonusStateTaxRate = 0.05; // default ~5% estimate
  if (stateInfo.hasStateTax) {
    if (stateInfo.type === 'flat' && stateInfo.flatRate) {
      bonusStateTaxRate = stateInfo.flatRate;
    } else if (stateInfo.code === 'CA') {
      bonusStateTaxRate = 0.1023; // CA supplemental bonus rate
    } else if (stateInfo.brackets) {
      const bList = stateInfo.brackets[inputs.filingStatus === 'married' ? 'married' : 'single'];
      bonusStateTaxRate = bList[bList.length - 1]?.rate || 0.06;
    }
  } else {
    bonusStateTaxRate = 0;
  }
  const bonusStateTax = bonusTaxableGross * bonusStateTaxRate;

  // Bonus FICA Tax (Social Security 6.2% up to cap, Medicare 1.45% + 0.9%)
  const ssUsedSoFar = Math.min(ficaGrossAnnual, taxLimits.SOCIAL_SECURITY_WAGE_CAP);
  const ssRemainingCap = Math.max(0, taxLimits.SOCIAL_SECURITY_WAGE_CAP - ssUsedSoFar);
  const bonusSubjectSS = Math.min(bonusTaxableGross, ssRemainingCap);
  const bonusSS = bonusSubjectSS * taxLimits.SOCIAL_SECURITY_RATE;

  let bonusMedicare = bonusTaxableGross * taxLimits.MEDICARE_RATE;
  if (ficaGrossAnnual + bonusTaxableGross > addMedicareThreshold) {
    const amountInAddMedicare = Math.max(
      0,
      (ficaGrossAnnual + bonusTaxableGross) - Math.max(ficaGrossAnnual, addMedicareThreshold)
    );
    bonusMedicare += amountInAddMedicare * taxLimits.ADDITIONAL_MEDICARE_RATE;
  }
  const bonusFicaTax = bonusSS + bonusMedicare;

  const bonusTotalTaxes = bonusFederalTax + bonusStateTax + bonusFicaTax;
  const bonusNetTakeHome = Math.max(0, grossAnnualBonus - bonus401kContribution - bonusTotalTaxes);

  const totalCombinedNetAnnual = netTakeHomePayAnnual + bonusNetTakeHome;
  const totalCombinedWealthInvestedAnnual =
    (preTaxDeductionsAnnual) +
    companyMatchAnnual +
    postTaxContributionsAnnual +
    bonus401kContribution +
    bonusCompanyMatch;

  // Percentages relative to Gross Biweekly
  const pct = (val: number) => (grossBiweekly > 0 ? (val / grossBiweekly) * 100 : 0);

  return {
    payPeriodsPerYear: 26,
    grossAnnual,
    grossBiweekly,

    companyMatchBiweekly,
    companyMatchAnnual,
    employee401kPercent,
    total401kAccumulationBiweekly,
    total401kAccumulationAnnual,

    preTax401kBiweekly,
    hsaBiweekly,
    employerHsaBiweekly,
    employerHsaAnnual,
    fsaBiweekly,
    preTaxDeductionsBiweekly,
    preTaxDeductionsAnnual,

    taxableGrossBiweekly,
    taxableGrossAnnual,

    federalTaxBiweekly,
    federalTaxAnnual,
    stateTaxBiweekly,
    stateTaxAnnual,
    socialSecurityBiweekly,
    socialSecurityAnnual,
    medicareBiweekly,
    medicareAnnual,
    sdiBiweekly,
    sdiAnnual,
    totalTaxesBiweekly,
    totalTaxesAnnual,

    roth401kBiweekly,
    iraBiweekly,
    rothIraBiweekly,
    plan529Biweekly,
    custodialAccountBiweekly,
    trumpAccountBiweekly,
    custodialIraBiweekly,
    esppContributionBiweekly,
    esppDiscountGainBiweekly,
    postTaxContributionsBiweekly,
    postTaxContributionsAnnual,

    netTakeHomePayBiweekly,
    netTakeHomePayAnnual,
    netTakeHomeAfterPostTaxBiweekly,
    netTakeHomeAfterPostTaxAnnual,

    // Annual Bonus Breakdown
    grossAnnualBonus,
    grossTotalAnnualWithBonus,
    bonus401kContribution,
    bonusCompanyMatch,
    bonusTaxableGross,
    bonusFederalTax,
    bonusStateTax,
    bonusFicaTax,
    bonusTotalTaxes,
    bonusNetTakeHome,
    totalCombinedNetAnnual,
    totalCombinedWealthInvestedAnnual,

    percentages: {
      preTax: pct(preTaxDeductionsBiweekly),
      taxes: pct(totalTaxesBiweekly),
      postTax: pct(postTaxContributionsBiweekly),
      takeHome: pct(netTakeHomePayBiweekly),

      federalTax: pct(federalTaxBiweekly),
      stateTax: pct(stateTaxBiweekly),
      socialSecurity: pct(socialSecurityBiweekly),
      medicare: pct(medicareBiweekly),
      sdi: pct(sdiBiweekly),
    },
  };
}

export function generateSankeyData(
  inputs: UserFinancialInputs,
  res: TaxBreakdownResult
): { nodes: SankeyNodeData[]; links: SankeyLinkData[] } {
  const isBiweekly = inputs.payFrequency !== 'annual';
  const mul = isBiweekly ? 1 : 26;
  const val = (b: number) => Math.max(0, b * mul);

  const formatCurrency = (amt: number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(amt);
  };

  const grossVal = val(res.grossBiweekly);

  const nodes: SankeyNodeData[] = [
    {
      id: 'gross',
      name: 'Gross Paycheck',
      category: 'gross',
      valueBiweekly: res.grossBiweekly,
      valueAnnual: res.grossAnnual,
      percentageOfGross: 100,
      color: '#3b82f6', // blue
    },
  ];

  const links: SankeyLinkData[] = [];

  // Pre-tax Node
  const preTaxVal = val(res.preTaxDeductionsBiweekly);
  if (preTaxVal > 0) {
    nodes.push({
      id: 'preTax',
      name: 'Pre-Tax Deductions',
      category: 'preTax',
      valueBiweekly: res.preTaxDeductionsBiweekly,
      valueAnnual: res.preTaxDeductionsAnnual,
      percentageOfGross: res.percentages.preTax,
      color: '#8b5cf6', // purple
    });

    links.push({
      source: 'gross',
      target: 'preTax',
      value: preTaxVal,
      formattedValue: formatCurrency(preTaxVal),
      percentage: res.percentages.preTax,
      color: '#c4b5fd',
    });

    // Sub-nodes for Pre-tax
    if (res.preTax401kBiweekly > 0) {
      nodes.push({
        id: 'trad401k',
        name: '401(k) Employee',
        category: 'preTax',
        valueBiweekly: res.preTax401kBiweekly,
        valueAnnual: res.preTax401kBiweekly * 26,
        percentageOfGross: (res.preTax401kBiweekly / res.grossBiweekly) * 100,
        color: '#7c3aed',
      });
      links.push({
        source: 'preTax',
        target: 'trad401k',
        value: val(res.preTax401kBiweekly),
        formattedValue: formatCurrency(val(res.preTax401kBiweekly)),
        percentage: (res.preTax401kBiweekly / res.grossBiweekly) * 100,
      });
    }

    if (res.companyMatchBiweekly > 0) {
      const matchP = inputs.companyMatchPercent ?? 100;
      const matchCap = inputs.companyMatchUpToPercent ?? 6;
      nodes.push({
        id: 'coMatch',
        name: `Employer Match (${matchP}% up to ${matchCap}%)`,
        category: 'preTax',
        valueBiweekly: res.companyMatchBiweekly,
        valueAnnual: res.companyMatchAnnual,
        percentageOfGross: (res.companyMatchBiweekly / res.grossBiweekly) * 100,
        color: '#10b981',
      });
      links.push({
        source: 'preTax',
        target: 'coMatch',
        value: val(res.companyMatchBiweekly),
        formattedValue: formatCurrency(val(res.companyMatchBiweekly)),
        percentage: (res.companyMatchBiweekly / res.grossBiweekly) * 100,
        color: '#a7f3d0',
      });
    }

    if (res.hsaBiweekly > 0) {
      nodes.push({
        id: 'hsa',
        name: 'HSA (Health Savings)',
        category: 'preTax',
        valueBiweekly: res.hsaBiweekly,
        valueAnnual: res.hsaBiweekly * 26,
        percentageOfGross: (res.hsaBiweekly / res.grossBiweekly) * 100,
        color: '#0d9488',
      });
      links.push({
        source: 'preTax',
        target: 'hsa',
        value: val(res.hsaBiweekly),
        formattedValue: formatCurrency(val(res.hsaBiweekly)),
        percentage: (res.hsaBiweekly / res.grossBiweekly) * 100,
      });
    }

    if (res.fsaBiweekly > 0) {
      nodes.push({
        id: 'fsa',
        name: 'FSA Account',
        category: 'preTax',
        valueBiweekly: res.fsaBiweekly,
        valueAnnual: res.fsaBiweekly * 26,
        percentageOfGross: (res.fsaBiweekly / res.grossBiweekly) * 100,
        color: '#0284c7',
      });
      links.push({
        source: 'preTax',
        target: 'fsa',
        value: val(res.fsaBiweekly),
        formattedValue: formatCurrency(val(res.fsaBiweekly)),
        percentage: (res.fsaBiweekly / res.grossBiweekly) * 100,
      });
    }
  }

  // Taxes
  const taxesVal = val(res.totalTaxesBiweekly);
  if (taxesVal > 0) {
    const stateName = US_STATES[inputs.state]?.code || inputs.state;

    if (inputs.dissectTaxesInSankey) {
      // Dissected Taxes (Detailed nodes directly from gross)
      if (res.federalTaxBiweekly > 0) {
        nodes.push({
          id: 'tax_federal',
          name: 'Federal Income Tax',
          category: 'taxChild',
          valueBiweekly: res.federalTaxBiweekly,
          valueAnnual: res.federalTaxAnnual,
          percentageOfGross: res.percentages.federalTax,
          color: '#e11d48',
        });
        links.push({
          source: 'gross',
          target: 'tax_federal',
          value: val(res.federalTaxBiweekly),
          formattedValue: formatCurrency(val(res.federalTaxBiweekly)),
          percentage: res.percentages.federalTax,
          color: '#fda4af',
        });
      }

      if (res.stateTaxBiweekly > 0) {
        nodes.push({
          id: 'tax_state',
          name: `${stateName} State Income Tax`,
          category: 'taxChild',
          valueBiweekly: res.stateTaxBiweekly,
          valueAnnual: res.stateTaxAnnual,
          percentageOfGross: res.percentages.stateTax,
          color: '#f43f5e',
        });
        links.push({
          source: 'gross',
          target: 'tax_state',
          value: val(res.stateTaxBiweekly),
          formattedValue: formatCurrency(val(res.stateTaxBiweekly)),
          percentage: res.percentages.stateTax,
          color: '#fecdd3',
        });
      }

      if (res.socialSecurityBiweekly > 0) {
        nodes.push({
          id: 'tax_ss',
          name: 'Social Security (6.2%)',
          category: 'taxChild',
          valueBiweekly: res.socialSecurityBiweekly,
          valueAnnual: res.socialSecurityAnnual,
          percentageOfGross: res.percentages.socialSecurity,
          color: '#d97706',
        });
        links.push({
          source: 'gross',
          target: 'tax_ss',
          value: val(res.socialSecurityBiweekly),
          formattedValue: formatCurrency(val(res.socialSecurityBiweekly)),
          percentage: res.percentages.socialSecurity,
          color: '#fde68a',
        });
      }

      if (res.medicareBiweekly > 0) {
        nodes.push({
          id: 'tax_medicare',
          name: 'Medicare Tax (1.45%+)',
          category: 'taxChild',
          valueBiweekly: res.medicareBiweekly,
          valueAnnual: res.medicareAnnual,
          percentageOfGross: res.percentages.medicare,
          color: '#ea580c',
        });
        links.push({
          source: 'gross',
          target: 'tax_medicare',
          value: val(res.medicareBiweekly),
          formattedValue: formatCurrency(val(res.medicareBiweekly)),
          percentage: res.percentages.medicare,
          color: '#ffedd5',
        });
      }

      if (res.sdiBiweekly > 0) {
        nodes.push({
          id: 'tax_sdi',
          name: `${stateName} SDI / Paid Leave`,
          category: 'taxChild',
          valueBiweekly: res.sdiBiweekly,
          valueAnnual: res.sdiAnnual,
          percentageOfGross: res.percentages.sdi,
          color: '#ca8a04',
        });
        links.push({
          source: 'gross',
          target: 'tax_sdi',
          value: val(res.sdiBiweekly),
          formattedValue: formatCurrency(val(res.sdiBiweekly)),
          percentage: res.percentages.sdi,
          color: '#fef08a',
        });
      }
    } else {
      // Aggregated Taxes Node
      nodes.push({
        id: 'taxes',
        name: 'Total Taxes',
        category: 'taxes',
        valueBiweekly: res.totalTaxesBiweekly,
        valueAnnual: res.totalTaxesAnnual,
        percentageOfGross: res.percentages.taxes,
        color: '#f43f5e', // rose
      });

      links.push({
        source: 'gross',
        target: 'taxes',
        value: taxesVal,
        formattedValue: formatCurrency(taxesVal),
        percentage: res.percentages.taxes,
        color: '#fecdd3',
      });

      // Sub-tax nodes connected from 'taxes'
      if (res.federalTaxBiweekly > 0) {
        nodes.push({
          id: 'tax_federal',
          name: 'Federal Income Tax',
          category: 'taxChild',
          valueBiweekly: res.federalTaxBiweekly,
          valueAnnual: res.federalTaxAnnual,
          percentageOfGross: res.percentages.federalTax,
          color: '#e11d48',
        });
        links.push({
          source: 'taxes',
          target: 'tax_federal',
          value: val(res.federalTaxBiweekly),
          formattedValue: formatCurrency(val(res.federalTaxBiweekly)),
          percentage: res.percentages.federalTax,
        });
      }

      if (res.stateTaxBiweekly > 0) {
        nodes.push({
          id: 'tax_state',
          name: `${stateName} State Tax`,
          category: 'taxChild',
          valueBiweekly: res.stateTaxBiweekly,
          valueAnnual: res.stateTaxAnnual,
          percentageOfGross: res.percentages.stateTax,
          color: '#f43f5e',
        });
        links.push({
          source: 'taxes',
          target: 'tax_state',
          value: val(res.stateTaxBiweekly),
          formattedValue: formatCurrency(val(res.stateTaxBiweekly)),
          percentage: res.percentages.stateTax,
        });
      }

      if (res.socialSecurityBiweekly > 0) {
        nodes.push({
          id: 'tax_ss',
          name: 'Social Security',
          category: 'taxChild',
          valueBiweekly: res.socialSecurityBiweekly,
          valueAnnual: res.socialSecurityAnnual,
          percentageOfGross: res.percentages.socialSecurity,
          color: '#d97706',
        });
        links.push({
          source: 'taxes',
          target: 'tax_ss',
          value: val(res.socialSecurityBiweekly),
          formattedValue: formatCurrency(val(res.socialSecurityBiweekly)),
          percentage: res.percentages.socialSecurity,
        });
      }

      if (res.medicareBiweekly > 0) {
        nodes.push({
          id: 'tax_medicare',
          name: 'Medicare',
          category: 'taxChild',
          valueBiweekly: res.medicareBiweekly,
          valueAnnual: res.medicareAnnual,
          percentageOfGross: res.percentages.medicare,
          color: '#ea580c',
        });
        links.push({
          source: 'taxes',
          target: 'tax_medicare',
          value: val(res.medicareBiweekly),
          formattedValue: formatCurrency(val(res.medicareBiweekly)),
          percentage: res.percentages.medicare,
        });
      }

      if (res.sdiBiweekly > 0) {
        nodes.push({
          id: 'tax_sdi',
          name: 'SDI / Disability Tax',
          category: 'taxChild',
          valueBiweekly: res.sdiBiweekly,
          valueAnnual: res.sdiAnnual,
          percentageOfGross: res.percentages.sdi,
          color: '#ca8a04',
        });
        links.push({
          source: 'taxes',
          target: 'tax_sdi',
          value: val(res.sdiBiweekly),
          formattedValue: formatCurrency(val(res.sdiBiweekly)),
          percentage: res.percentages.sdi,
        });
      }
    }
  }

  // Optional Post-Tax Investments & Deductions (Roth 401k/IRA, 529, Child Accounts, ESPP)
  const postTaxVal = val(res.postTaxContributionsBiweekly);
  if (inputs.includePostTaxInSankey && postTaxVal > 0) {
    nodes.push({
      id: 'postTax',
      name: 'Post-Tax Accounts (Roth, 529, Child, ESPP)',
      category: 'postTax',
      valueBiweekly: res.postTaxContributionsBiweekly,
      valueAnnual: res.postTaxContributionsAnnual,
      percentageOfGross: res.percentages.postTax,
      color: '#059669', // emerald
    });

    links.push({
      source: 'gross',
      target: 'postTax',
      value: postTaxVal,
      formattedValue: formatCurrency(postTaxVal),
      percentage: res.percentages.postTax,
      color: '#a7f3d0',
    });

    // Sub-postTax nodes
    if (res.roth401kBiweekly > 0) {
      nodes.push({
        id: 'roth401k',
        name: 'Roth 401(k)',
        category: 'postTax',
        valueBiweekly: res.roth401kBiweekly,
        valueAnnual: res.roth401kBiweekly * 26,
        percentageOfGross: (res.roth401kBiweekly / res.grossBiweekly) * 100,
        color: '#10b981',
      });
      links.push({
        source: 'postTax',
        target: 'roth401k',
        value: val(res.roth401kBiweekly),
        formattedValue: formatCurrency(val(res.roth401kBiweekly)),
        percentage: (res.roth401kBiweekly / res.grossBiweekly) * 100,
      });
    }

    if (res.rothIraBiweekly > 0 || res.iraBiweekly > 0) {
      const iraTot = res.rothIraBiweekly + res.iraBiweekly;
      nodes.push({
        id: 'ira',
        name: res.rothIraBiweekly > 0 ? 'Roth IRA / IRA' : 'Traditional IRA (Post-Tax)',
        category: 'postTax',
        valueBiweekly: iraTot,
        valueAnnual: iraTot * 26,
        percentageOfGross: (iraTot / res.grossBiweekly) * 100,
        color: '#047857',
      });
      links.push({
        source: 'postTax',
        target: 'ira',
        value: val(iraTot),
        formattedValue: formatCurrency(val(iraTot)),
        percentage: (iraTot / res.grossBiweekly) * 100,
      });
    }

    if (res.plan529Biweekly > 0) {
      nodes.push({
        id: 'plan529',
        name: '529 College Savings',
        category: 'postTax',
        valueBiweekly: res.plan529Biweekly,
        valueAnnual: res.plan529Biweekly * 26,
        percentageOfGross: (res.plan529Biweekly / res.grossBiweekly) * 100,
        color: '#0284c7',
      });
      links.push({
        source: 'postTax',
        target: 'plan529',
        value: val(res.plan529Biweekly),
        formattedValue: formatCurrency(val(res.plan529Biweekly)),
        percentage: (res.plan529Biweekly / res.grossBiweekly) * 100,
      });
    }

    if (res.custodialAccountBiweekly > 0 || res.trumpAccountBiweekly > 0 || res.custodialIraBiweekly > 0) {
      const childTotal = res.custodialAccountBiweekly + res.trumpAccountBiweekly + res.custodialIraBiweekly;
      nodes.push({
        id: 'childAccounts',
        name: 'Child Accounts (Trump / UTMA / IRA)',
        category: 'postTax',
        valueBiweekly: childTotal,
        valueAnnual: childTotal * 26,
        percentageOfGross: (childTotal / res.grossBiweekly) * 100,
        color: '#14b8a6',
      });
      links.push({
        source: 'postTax',
        target: 'childAccounts',
        value: val(childTotal),
        formattedValue: formatCurrency(val(childTotal)),
        percentage: (childTotal / res.grossBiweekly) * 100,
      });
    }

    if (res.esppContributionBiweekly > 0) {
      nodes.push({
        id: 'espp',
        name: `ESPP (${inputs.esppPercent}%)`,
        category: 'postTax',
        valueBiweekly: res.esppContributionBiweekly,
        valueAnnual: res.esppContributionBiweekly * 26,
        percentageOfGross: (res.esppContributionBiweekly / res.grossBiweekly) * 100,
        color: '#0d9488',
      });
      links.push({
        source: 'postTax',
        target: 'espp',
        value: val(res.esppContributionBiweekly),
        formattedValue: formatCurrency(val(res.esppContributionBiweekly)),
        percentage: (res.esppContributionBiweekly / res.grossBiweekly) * 100,
      });
    }
  }

  // Net Take-Home Pay Node
  const takeHomeVal = val(res.netTakeHomePayBiweekly);
  if (takeHomeVal > 0) {
    nodes.push({
      id: 'takeHome',
      name: 'Net Paycheck (Bank Account)',
      category: 'takeHome',
      valueBiweekly: res.netTakeHomePayBiweekly,
      valueAnnual: res.netTakeHomePayAnnual,
      percentageOfGross: res.percentages.takeHome,
      color: '#16a34a', // bright green
    });

    links.push({
      source: 'gross',
      target: 'takeHome',
      value: takeHomeVal,
      formattedValue: formatCurrency(takeHomeVal),
      percentage: res.percentages.takeHome,
      color: '#86efac',
    });
  }

  return { nodes, links };
}

export function getMaximizedInputs(current: UserFinancialInputs): UserFinancialInputs {
  const taxLimits = getTaxLimitsForYear(current.taxYear || 2026);
  const grossBiweekly = current.grossSalary;
  const max401kAnnual = current.age >= 50
    ? taxLimits.TRADITIONAL_401K_MAX + taxLimits.TRADITIONAL_401K_CATCHUP
    : taxLimits.TRADITIONAL_401K_MAX;
  const biweekly401k = Math.round((max401kAnnual / 26) * 100) / 100;

  const employerHsaAnnual = current.employerHsaAnnual || 0;
  const maxHsaStatutoryAnnual = current.hsaCoverage === 'family'
    ? taxLimits.HSA_FAMILY_MAX + (current.age >= 55 ? taxLimits.HSA_CATCHUP : 0)
    : taxLimits.HSA_SINGLE_MAX + (current.age >= 55 ? taxLimits.HSA_CATCHUP : 0);
  const maxEmployeeHsaAnnual = Math.max(0, maxHsaStatutoryAnnual - employerHsaAnnual);
  const biweeklyHsa = Math.round((maxEmployeeHsaAnnual / 26) * 100) / 100;

  const maxIraAnnual = current.age >= 50
    ? taxLimits.IRA_MAX + taxLimits.IRA_CATCHUP
    : taxLimits.IRA_MAX;
  const biweeklyIra = Math.round((maxIraAnnual / 26) * 100) / 100;

  return {
    ...current,
    traditional401k: biweekly401k,
    traditional401kIsPercent: false,
    hsa: biweeklyHsa,
    rothIra: biweeklyIra,
    esppPercent: 15,
    plan529: 250, // $250/biweekly per child default for education
    custodialAccount: 150, // $150/biweekly for child growth
  };
}
