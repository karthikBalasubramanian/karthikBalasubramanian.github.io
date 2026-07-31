import { UserFinancialInputs, TaxBreakdownResult, StateTaxInfo, SankeyNodeData, SankeyLinkData } from '../types';
import { TAX_LIMITS_2026, US_STATES } from '../data/taxRates';

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
    ? TAX_LIMITS_2026.TRADITIONAL_401K_MAX + TAX_LIMITS_2026.TRADITIONAL_401K_CATCHUP
    : TAX_LIMITS_2026.TRADITIONAL_401K_MAX;
  
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

  // 401(k) Employer Match Calculation (e.g. 100% match on up to 6% of salary)
  const companyMatchPercent = inputs.companyMatchPercent ?? 100;
  const companyMatchUpToPercent = inputs.companyMatchUpToPercent ?? 6;

  const employee401kBiweekly = preTax401kBiweekly + roth401kBiweekly;
  const employee401kPercent = grossBiweekly > 0 ? (employee401kBiweekly / grossBiweekly) * 100 : 0;

  const eligibleMatchPercent = Math.min(employee401kPercent, companyMatchUpToPercent);
  const companyMatchBiweekly = grossBiweekly * (eligibleMatchPercent / 100) * (companyMatchPercent / 100);
  const companyMatchAnnual = companyMatchBiweekly * 26;

  const total401kAccumulationBiweekly = employee401kBiweekly + companyMatchBiweekly;
  const total401kAccumulationAnnual = total401kAccumulationBiweekly * 26;

  const hsaBiweekly = inputs.hsa || 0;
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
  const stdDeduction = TAX_LIMITS_2026.FEDERAL_STANDARD_DEDUCTION[inputs.filingStatus] || 15000;
  const federalTaxableIncome = Math.max(0, taxableGrossAnnual - stdDeduction);

  const federalBrackets = TAX_LIMITS_2026.FEDERAL_BRACKETS[inputs.filingStatus] || TAX_LIMITS_2026.FEDERAL_BRACKETS.single;
  let federalTaxAnnual = 0;

  for (const bracket of federalBrackets) {
    if (federalTaxableIncome > bracket.min) {
      const taxableInBracket = Math.min(federalTaxableIncome, bracket.max) - bracket.min;
      federalTaxAnnual += taxableInBracket * bracket.rate;
    }
  }
  const federalTaxBiweekly = federalTaxAnnual / 26;

  // 3. State Income Tax Calculation
  const stateInfo: StateTaxInfo = US_STATES[inputs.state] || US_STATES.OTHER;
  let stateTaxAnnual = 0;

  if (stateInfo.hasStateTax) {
    if (stateInfo.type === 'flat' && stateInfo.flatRate) {
      stateTaxAnnual = taxableGrossAnnual * stateInfo.flatRate;
    } else if (stateInfo.type === 'progressive' && stateInfo.brackets) {
      const bracketsList = stateInfo.brackets[inputs.filingStatus === 'married' ? 'married' : 'single'];
      for (const bracket of bracketsList) {
        if (taxableGrossAnnual > bracket.min) {
          const taxableInBracket = Math.min(taxableGrossAnnual, bracket.max) - bracket.min;
          stateTaxAnnual += taxableInBracket * bracket.rate;
        }
      }
    }
  }
  const stateTaxBiweekly = stateTaxAnnual / 26;

  // 4. Social Security Tax (6.2% up to $176,100 cap)
  const ssSubjectAnnual = Math.min(ficaGrossAnnual, TAX_LIMITS_2026.SOCIAL_SECURITY_WAGE_CAP);
  const socialSecurityAnnual = ssSubjectAnnual * TAX_LIMITS_2026.SOCIAL_SECURITY_RATE;
  const socialSecurityBiweekly = socialSecurityAnnual / 26;

  // 5. Medicare Tax (1.45% baseline + 0.9% additional over threshold)
  let medicareAnnual = ficaGrossAnnual * TAX_LIMITS_2026.MEDICARE_RATE;
  const addMedicareThreshold = inputs.filingStatus === 'married'
    ? TAX_LIMITS_2026.ADDITIONAL_MEDICARE_THRESHOLD_MARRIED
    : TAX_LIMITS_2026.ADDITIONAL_MEDICARE_THRESHOLD_SINGLE;

  if (ficaGrossAnnual > addMedicareThreshold) {
    medicareAnnual += (ficaGrossAnnual - addMedicareThreshold) * TAX_LIMITS_2026.ADDITIONAL_MEDICARE_RATE;
  }
  const medicareBiweekly = medicareAnnual / 26;

  // 6. State Disability Insurance (SDI / Paid Leave)
  let sdiAnnual = 0;
  if (stateInfo.hasSDI && stateInfo.sdiRate > 0) {
    const sdiWage = stateInfo.sdiMaxWage ? Math.min(grossAnnual, stateInfo.sdiMaxWage) : grossAnnual;
    sdiAnnual = sdiWage * stateInfo.sdiRate;
  }
  const sdiBiweekly = sdiAnnual / 26;

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

  // Net Take-Home Pay (Checking Account deposit)
  const netTakeHomePayBiweekly = grossBiweekly - preTaxDeductionsBiweekly - totalTaxesBiweekly - postTaxContributionsBiweekly;
  const netTakeHomePayAnnual = netTakeHomePayBiweekly * 26;

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
    bonusCompanyMatch = grossAnnualBonus * (eligibleMatchPercent / 100) * (companyMatchPercent / 100);
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
  const ssUsedSoFar = Math.min(ficaGrossAnnual, TAX_LIMITS_2026.SOCIAL_SECURITY_WAGE_CAP);
  const ssRemainingCap = Math.max(0, TAX_LIMITS_2026.SOCIAL_SECURITY_WAGE_CAP - ssUsedSoFar);
  const bonusSubjectSS = Math.min(bonusTaxableGross, ssRemainingCap);
  const bonusSS = bonusSubjectSS * TAX_LIMITS_2026.SOCIAL_SECURITY_RATE;

  let bonusMedicare = bonusTaxableGross * TAX_LIMITS_2026.MEDICARE_RATE;
  if (ficaGrossAnnual + bonusTaxableGross > addMedicareThreshold) {
    const amountInAddMedicare = Math.max(
      0,
      (ficaGrossAnnual + bonusTaxableGross) - Math.max(ficaGrossAnnual, addMedicareThreshold)
    );
    bonusMedicare += amountInAddMedicare * TAX_LIMITS_2026.ADDITIONAL_MEDICARE_RATE;
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

  // Post-Tax Investments & Deductions
  const postTaxVal = val(res.postTaxContributionsBiweekly);
  if (postTaxVal > 0) {
    nodes.push({
      id: 'postTax',
      name: 'Post-Tax & Child Allocations',
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
  const grossBiweekly = current.grossSalary;
  const max401kAnnual = current.age >= 50
    ? TAX_LIMITS_2026.TRADITIONAL_401K_MAX + TAX_LIMITS_2026.TRADITIONAL_401K_CATCHUP
    : TAX_LIMITS_2026.TRADITIONAL_401K_MAX;
  const biweekly401k = Math.round((max401kAnnual / 26) * 100) / 100;

  const maxHsaAnnual = current.hsaCoverage === 'family'
    ? TAX_LIMITS_2026.HSA_FAMILY_MAX + (current.age >= 55 ? TAX_LIMITS_2026.HSA_CATCHUP : 0)
    : TAX_LIMITS_2026.HSA_SINGLE_MAX + (current.age >= 55 ? TAX_LIMITS_2026.HSA_CATCHUP : 0);
  const biweeklyHsa = Math.round((maxHsaAnnual / 26) * 100) / 100;

  const maxIraAnnual = current.age >= 50
    ? TAX_LIMITS_2026.IRA_MAX + TAX_LIMITS_2026.IRA_CATCHUP
    : TAX_LIMITS_2026.IRA_MAX;
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
