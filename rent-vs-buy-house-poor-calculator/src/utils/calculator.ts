import type {
  UserHousingInputs,
  MortgageBreakdown,
  HousePoorAnalysis,
  NetWorthProjectionPoint,
  HedonicSpecMapping,
  ExpenseOptimizationRecommendation,
  StressTestMetrics,
  InstitutionalAnalysis,
} from '../types';
import { lookupZipCode } from '../data/zipDatabase';

export function calculateMortgagePiti(inputs: UserHousingInputs): MortgageBreakdown {
  const homePrice = inputs.targetHomePrice || 850000;
  const downPaymentPercent = inputs.downPaymentPercent ?? 20;
  const downPaymentAmount = homePrice * (downPaymentPercent / 100);
  const loanAmount = Math.max(0, homePrice - downPaymentAmount);

  const annualInterestRate = inputs.interestRate || 6.5;
  const monthlyRate = annualInterestRate / 100 / 12;
  const totalPayments = (inputs.loanTermYears || 30) * 12;

  let monthlyPrincipalAndInterest = 0;
  if (monthlyRate > 0 && totalPayments > 0 && loanAmount > 0) {
    monthlyPrincipalAndInterest =
      (loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, totalPayments))) /
      (Math.pow(1 + monthlyRate, totalPayments) - 1);
  }

  // Property Tax (ZIP specific or user override)
  const zipData = lookupZipCode(inputs.zipCode, inputs.state);
  const effectiveTaxRate = inputs.propertyTaxRate || zipData.propertyTaxRate;
  const monthlyPropertyTax = (homePrice * (effectiveTaxRate / 100)) / 12;

  // Homeowners Insurance (est 0.45% of home value annually or user override)
  const annualInsurance = inputs.homeInsuranceAnnual || Math.round(homePrice * 0.0045);
  const monthlyInsurance = annualInsurance / 12;

  // PMI (Private Mortgage Insurance if down payment < 20% or custom PMI override)
  let monthlyPmi = 0;
  if (inputs.customPmiPercent !== undefined) {
    monthlyPmi = (loanAmount * (inputs.customPmiPercent / 100)) / 12;
  } else if (downPaymentPercent < 20) {
    monthlyPmi = (loanAmount * 0.0075) / 12; // ~0.75% annual PMI rate
  }

  // HOA (Toggleable on/off)
  const monthlyHoa = (inputs.hasHoa ?? true) ? (inputs.hoaMonthly || 0) : 0;

  // Maintenance Reserve (Toggleable out-of-pocket reserve fund)
  const annualMaintenanceRate = inputs.maintenancePercentAnnual ?? 1.0;
  const monthlyMaintenance = (inputs.includeMaintenanceInPiti ?? true)
    ? (homePrice * (annualMaintenanceRate / 100)) / 12
    : 0;

  const totalMonthlyPiti =
    monthlyPrincipalAndInterest +
    monthlyPropertyTax +
    monthlyInsurance +
    monthlyPmi +
    monthlyHoa +
    monthlyMaintenance;

  return {
    loanAmount,
    downPaymentAmount,
    monthlyPrincipalAndInterest,
    monthlyPropertyTax,
    monthlyInsurance,
    monthlyPmi,
    monthlyHoa,
    monthlyMaintenance,
    totalMonthlyPiti,
  };
}

export function generateRedfinMlsUrl(zipCode: string, beds: number, baths: number, maxPrice: number): string {
  const cleanZip = zipCode.trim() || '95113';
  return `https://www.redfin.com/zipcode/${cleanZip}/filter/bedrooms=${beds},baths-min=${baths},max-price=${Math.round(maxPrice)}`;
}

export function analyzeHousePoorStatus(inputs: UserHousingInputs): HousePoorAnalysis {
  const monthlyNetTakeHome = inputs.monthlyTakeHome || 9048;
  const rainyDayBufferTarget = inputs.rainyDayBufferTarget ?? 500;
  const salaryRaisePct = inputs.annualSalaryRaisePercent ?? 3.0;

  // 1. Total Non-Housing Lifestyle Expenses
  const l = inputs.lifestyle;
  const totalLifestyleExpenses =
    (l.groceries || 0) +
    (l.utilities || 0) +
    (l.carPaymentInsurance || 0) +
    (l.subscriptionsStreaming || 0) +
    (l.diningOutEntertainment || 0) +
    (l.healthMedical || 0) +
    (l.otherMisc || 0);

  const surplusCashBeforeHousing = Math.max(0, monthlyNetTakeHome - totalLifestyleExpenses);

  // 2. Buy Costs & Buffer Today (Year 0)
  const mortgage = calculateMortgagePiti(inputs);
  const monthlyBuyHousingCost = mortgage.totalMonthlyPiti;
  const leftoverCashBufferBuy = surplusCashBeforeHousing - monthlyBuyHousingCost;
  const housingNetPercentBuy = monthlyNetTakeHome > 0 ? (monthlyBuyHousingCost / monthlyNetTakeHome) * 100 : 0;
  const isReadyToBuyToday = leftoverCashBufferBuy >= rainyDayBufferTarget;
  const isHousePoorBuy = leftoverCashBufferBuy < rainyDayBufferTarget || housingNetPercentBuy > 45;

  // 3. Rent Costs & Buffer
  const monthlyRentHousingCost = inputs.currentRent || 3000;
  const leftoverCashBufferRent = surplusCashBeforeHousing - monthlyRentHousingCost;
  const housingNetPercentRent = monthlyNetTakeHome > 0 ? (monthlyRentHousingCost / monthlyNetTakeHome) * 100 : 0;

  // Monthly Cash Flow Difference (Buying Cost vs Renting Cost)
  const monthlyRentSavings = monthlyBuyHousingCost - monthlyRentHousingCost;

  // 4. Multi-Year Homeownership Readiness Simulation (Years 0 to 7)
  const currentYear = new Date().getFullYear();
  const homePrice = inputs.targetHomePrice || 850000;
  const baseDownPayment = mortgage.downPaymentAmount;
  const monthlySavingsWhileRenting = inputs.monthlyDownPaymentSavings ?? Math.max(500, monthlyRentSavings > 0 ? monthlyRentSavings : 1000);

  const readinessTimeline = [];
  let readinessYear = 0;
  let foundReadiness = false;

  for (let yr = 0; yr <= 7; yr++) {
    const simTakeHome = Math.round(monthlyNetTakeHome * Math.pow(1 + salaryRaisePct / 100, yr));
    const extraDownPayment = yr * 12 * monthlySavingsWhileRenting;
    const simTotalDownPayment = baseDownPayment + extraDownPayment;
    const simLoanAmount = Math.max(0, homePrice - simTotalDownPayment);

    // Recalculate PITI with reduced loan amount
    const annualInterestRate = inputs.interestRate || 6.5;
    const monthlyRate = annualInterestRate / 100 / 12;
    const totalPayments = (inputs.loanTermYears || 30) * 12;
    let simPi = 0;
    if (monthlyRate > 0 && totalPayments > 0 && simLoanAmount > 0) {
      simPi = (simLoanAmount * (monthlyRate * Math.pow(1 + monthlyRate, totalPayments))) / (Math.pow(1 + monthlyRate, totalPayments) - 1);
    }
    const zipData = lookupZipCode(inputs.zipCode, inputs.state);
    const simTax = (homePrice * ((inputs.propertyTaxRate || zipData.propertyTaxRate) / 100)) / 12;
    const simIns = (inputs.homeInsuranceAnnual || Math.round(homePrice * 0.0045)) / 12;
    const simPmi = (simTotalDownPayment / homePrice) < 0.2 ? (simLoanAmount * 0.0075) / 12 : 0;
    const simHoa = (inputs.hasHoa ?? true) ? (inputs.hoaMonthly || 0) : 0;
    const simMaint = (inputs.includeMaintenanceInPiti ?? true) ? (homePrice * ((inputs.maintenancePercentAnnual ?? 1.0) / 100)) / 12 : 0;

    const simPiti = Math.round(simPi + simTax + simIns + simPmi + simHoa + simMaint);
    const simSurplus = Math.round(simTakeHome - totalLifestyleExpenses - simPiti);
    const isReady = simSurplus >= rainyDayBufferTarget;

    if (isReady && !foundReadiness) {
      readinessYear = yr;
      foundReadiness = true;
    }

    readinessTimeline.push({
      year: yr,
      calendarYear: currentYear + yr,
      monthlyTakeHome: simTakeHome,
      totalDownPaymentSaved: Math.round(simTotalDownPayment),
      loanAmount: Math.round(simLoanAmount),
      monthlyPiti: simPiti,
      lifestyleExpenses: totalLifestyleExpenses,
      monthlyCashflowSurplus: simSurplus,
      isReadyToBuy: isReady,
    });
  }

  if (!foundReadiness) {
    readinessYear = 7;
  }

  // 5. Decision Verdict Logic
  let verdictStatus: 'buy' | 'caution' | 'rent_recommended' = 'buy';
  let verdictTitle = '';
  let verdictMessage = '';

  if (isReadyToBuyToday) {
    verdictStatus = 'buy';
    verdictTitle = '🟢 YOU ARE READY TO BUY TODAY!';
    verdictMessage = `Buying this home leaves you with a healthy +$${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo cashflow surplus after all PITI and lifestyle expenses (exceeding your $${rainyDayBufferTarget}/mo rainy day buffer target)!`;
  } else if (foundReadiness) {
    verdictStatus = 'caution';
    verdictTitle = `🎯 HUMBLE ROADMAP: READY TO BUY IN YEAR ${readinessYear} (${currentYear + readinessYear})`;
    verdictMessage = `Buying today leaves only $${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo in cash buffer. But with 3% annual salary raises and saving while renting, you will be 100% ready to buy comfortably in Year ${readinessYear} (${currentYear + readinessYear}) with a +$${Math.round(readinessTimeline[readinessYear]?.monthlyCashflowSurplus || 0).toLocaleString()}/mo surplus!`;
  } else {
    verdictStatus = 'rent_recommended';
    verdictTitle = '🏠 RENT TODAY & BUILD DOWN PAYMENT';
    verdictMessage = `Buying today leaves your monthly cashflow tight ($${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo buffer). We recommend renting today while accumulating a larger down payment to reach comfortable ownership!`;
  }

  // 6. Max Safe Purchase Price calculation (where PITI = Surplus Cash - rainyDayBufferTarget)
  const targetMaxMonthlyPiti = Math.max(1000, surplusCashBeforeHousing - rainyDayBufferTarget);
  const maxSafeHomePrice = Math.round((inputs.targetHomePrice || 850000) * (targetMaxMonthlyPiti / Math.max(1, monthlyBuyHousingCost)));

  // 7. Stage 1 & 2: Hedonic Physical Spec Mapping
  const zipRegionData = lookupZipCode(inputs.zipCode, inputs.state);
  const ppsqft = zipRegionData.avgPricePerSqFt || 450;
  const affordableSqFt = Math.round(maxSafeHomePrice / ppsqft);
  let estimatedBeds = 3;
  let estimatedBaths = 2;

  if (affordableSqFt < 1000) {
    estimatedBeds = 1;
    estimatedBaths = 1;
  } else if (affordableSqFt < 1400) {
    estimatedBeds = 2;
    estimatedBaths = 2;
  } else if (affordableSqFt < 2000) {
    estimatedBeds = 3;
    estimatedBaths = 2;
  } else if (affordableSqFt < 2800) {
    estimatedBeds = 4;
    estimatedBaths = 2.5;
  } else {
    estimatedBeds = 4;
    estimatedBaths = 3.5;
  }

  const hedonicSpecMapping: HedonicSpecMapping = {
    affordableSqFt,
    estimatedBeds,
    estimatedBaths,
    pricePerSqFt: ppsqft,
    zipCode: zipRegionData.zip,
    cityName: zipRegionData.city,
  };

  // 8. Stage 3: Lifestyle Expense Optimization Matrix
  const monthlyPaymentGap = Math.max(0, monthlyBuyHousingCost + rainyDayBufferTarget - surplusCashBeforeHousing);
  const discretionaryCategories = [
    { category: 'diningOutEntertainment', label: 'Dining Out & Entertainment', currentAmount: l.diningOutEntertainment || 0 },
    { category: 'subscriptionsStreaming', label: 'Subscriptions & Streaming', currentAmount: l.subscriptionsStreaming || 0 },
    { category: 'otherMisc', label: 'Other Miscellaneous Spend', currentAmount: l.otherMisc || 0 },
    { category: 'groceries', label: 'Groceries (Optimization)', currentAmount: l.groceries || 0 },
  ];

  let remainingGapToBridge = monthlyPaymentGap;
  const categoryTrims = [];
  let totalTrimmed = 0;

  for (const cat of discretionaryCategories) {
    if (remainingGapToBridge <= 0 || cat.currentAmount <= 0) {
      categoryTrims.push({
        ...cat,
        recommendedTrim: 0,
        newAmount: cat.currentAmount,
      });
      continue;
    }

    const maxTrimPossible = Math.round(cat.currentAmount * 0.6);
    const trimAmount = Math.min(remainingGapToBridge, maxTrimPossible);
    remainingGapToBridge -= trimAmount;
    totalTrimmed += trimAmount;

    categoryTrims.push({
      ...cat,
      recommendedTrim: trimAmount,
      newAmount: cat.currentAmount - trimAmount,
    });
  }

  const expenseOptimization: ExpenseOptimizationRecommendation = {
    monthlyPaymentGap: Math.round(monthlyPaymentGap),
    categoryTrims,
    totalTrimmed: Math.round(totalTrimmed),
    canBridgeGap100Percent: remainingGapToBridge <= 0,
  };

  // 9. Stage 4: Stress Test & Reserve Buffer Metrics
  const grossMonthlyIncomeEst = monthlyNetTakeHome / 0.70;
  const housingExpenseRatio = (monthlyBuyHousingCost / grossMonthlyIncomeEst) * 100;
  const totalMonthlyLivingCost = monthlyBuyHousingCost + totalLifestyleExpenses;
  const reserveBufferMonths = leftoverCashBufferBuy > 0 ? (leftoverCashBufferBuy * 12) / totalMonthlyLivingCost : 0;

  let riskLevel: 'low' | 'moderate' | 'house_poor' = 'low';
  let riskLabel = 'Low Risk — Safe Cash Cushion';

  if (leftoverCashBufferBuy < 0 || housingExpenseRatio > 45) {
    riskLevel = 'house_poor';
    riskLabel = 'High Risk — Cashflow Deficit / House Poor';
  } else if (leftoverCashBufferBuy < rainyDayBufferTarget || housingExpenseRatio > 35) {
    riskLevel = 'moderate';
    riskLabel = 'Moderate Risk — Tight Daily Buffer';
  }

  const stressTestMetrics: StressTestMetrics = {
    reserveBufferMonths: Math.round(reserveBufferMonths * 10) / 10,
    housingExpenseRatio: Math.round(housingExpenseRatio * 10) / 10,
    riskLevel,
    riskLabel,
  };

  // 10. 5 Institutional Financial Engineering Layers
  // Layer 1: Unrecoverable Costs Equation
  const interestYear1Monthly = (mortgage.loanAmount * ((inputs.interestRate || 6.5) / 100)) / 12;
  const principalYear1Monthly = Math.max(0, mortgage.monthlyPrincipalAndInterest - interestYear1Monthly);
  const capitalOpportunityCostMonthly = Math.round((mortgage.downPaymentAmount + (homePrice * 0.03)) * (0.05 / 12)); // 5% net forgone yield / 12
  const unrecoverableBuyMonthly = Math.round(
    interestYear1Monthly +
    mortgage.monthlyPropertyTax +
    mortgage.monthlyInsurance +
    mortgage.monthlyHoa +
    mortgage.monthlyMaintenance +
    mortgage.monthlyPmi +
    capitalOpportunityCostMonthly
  );
  const unrecoverableRentMonthly = Math.round(monthlyRentHousingCost);
  const unrecoverableDeltaMonthly = unrecoverableBuyMonthly - unrecoverableRentMonthly;

  // Layer 2: Dynamic Tax Shield Engine
  const annualInterest = interestYear1Monthly * 12;
  const annualPropertyTax = mortgage.monthlyPropertyTax * 12;
  const itemizedDeductionsAnnual = annualInterest + Math.min(annualPropertyTax, 10000); // SALT $10k cap
  const standardDeduction = 14600; // Federal Standard Deduction
  const marginalTaxRate = 0.32; // Combined Federal + State marginal rate
  const taxShieldAnnualRefund = Math.max(0, itemizedDeductionsAnnual - standardDeduction) * marginalTaxRate;
  const taxShieldMonthlyRefund = Math.round(taxShieldAnnualRefund / 12);
  const afterTaxMonthlyPiti = Math.max(0, monthlyBuyHousingCost - taxShieldMonthlyRefund);

  // Layer 3: Crossover Horizon (T*) Break-Even Solver
  let crossoverBreakEvenYear = 4;
  let cumUnrecBuy = 0;
  let cumRent = 0;
  for (let yr = 1; yr <= 15; yr++) {
    const yrRent = monthlyRentHousingCost * 12 * Math.pow(1.035, yr - 1);
    const yrUnrecBuy = unrecoverableBuyMonthly * 12 * Math.pow(1.02, yr - 1);
    cumRent += yrRent;
    cumUnrecBuy += yrUnrecBuy;
    if (cumUnrecBuy <= cumRent && crossoverBreakEvenYear === 4) {
      crossoverBreakEvenYear = yr;
    }
  }

  // Layer 4: Terminal Net Worth NPV Differential (10-Yr)
  const rMkt = 0.07;
  const rHome = 0.04;
  const upfrontCapital = mortgage.downPaymentAmount + Math.round(homePrice * 0.03);

  // Rent Portfolio calculation (Upfront capital + monthly savings invested in stocks)
  const rentMonthlySavings = Math.max(0, monthlyRentSavings);
  const rentAnnualSavings = rentMonthlySavings * 12;
  const rentSavingsFv = rMkt > 0 
    ? rentAnnualSavings * ((Math.pow(1 + rMkt, 10) - 1) / rMkt)
    : rentAnnualSavings * 10;
  const upfrontCap10Yr = upfrontCapital * Math.pow(1 + rMkt, 10);
  const totalRentVal = upfrontCap10Yr + rentSavingsFv;
  const totalRentCapitalInvested = upfrontCapital + (rentAnnualSavings * 10);
  const rentCapGains = Math.max(0, totalRentVal - totalRentCapitalInvested);
  const terminalNetWorthRent10Yr = Math.round(totalRentCapitalInvested + (rentCapGains * 0.85));

  // Buy Portfolio calculation (Home equity post-selling fee + buy monthly savings if rent > buy)
  const homeVal10Yr = homePrice * Math.pow(1 + rHome, 10);
  const buyEquity10Yr = (homeVal10Yr * 0.94) - (mortgage.loanAmount * 0.78);
  const buyMonthlySavings = Math.max(0, -monthlyRentSavings);
  const buyAnnualSavings = buyMonthlySavings * 12;
  const buySavingsFv = rMkt > 0 
    ? buyAnnualSavings * ((Math.pow(1 + rMkt, 10) - 1) / rMkt)
    : buyAnnualSavings * 10;
  const buyCapGains = Math.max(0, buySavingsFv - (buyAnnualSavings * 10));
  const buySavingsPostTax = (buyAnnualSavings * 10) + (buyCapGains * 0.85);
  const terminalNetWorthBuy10Yr = Math.round(buyEquity10Yr + buySavingsPostTax);
  const terminalNetWorthDelta10Yr = terminalNetWorthBuy10Yr - terminalNetWorthRent10Yr;

  // Layer 5: Monte Carlo Simulation (1,000 Iterations)
  let monteCarloWins = 0;
  const iterations = 1000;
  for (let i = 0; i < iterations; i++) {
    // Random home appreciation: mu = 4.0%, std = 3.5%
    const randApprec = 0.04 + (Math.random() - 0.5) * 0.07;
    // Random market return: mu = 7.0%, std = 10.0%
    const randMkt = Math.max(-0.05, 0.07 + (Math.random() - 0.5) * 0.20);
    // Random repair shock: 15% probability of a $4,000 repair shock
    const repairShock = Math.random() < 0.15 ? 4000 : 0;

    // Rent Simulation with market return
    const simRentUpfront = upfrontCapital * Math.pow(1 + randMkt, 10);
    const simRentSavingsFv = randMkt !== 0
      ? rentAnnualSavings * ((Math.pow(1 + randMkt, 10) - 1) / randMkt)
      : rentAnnualSavings * 10;
    const simRentTotal = simRentUpfront + simRentSavingsFv;
    const simRentCapGains = Math.max(0, simRentTotal - totalRentCapitalInvested);
    const simRentNw = totalRentCapitalInvested + (simRentCapGains * 0.85);

    // Buy Simulation with home appreciation and stock return on surplus cash
    const simBuyHomeVal = homePrice * Math.pow(1 + randApprec, 10);
    const simBuyEquity = (simBuyHomeVal * 0.94) - (mortgage.loanAmount * 0.78) - repairShock;
    const simBuySavingsFv = randMkt !== 0
      ? buyAnnualSavings * ((Math.pow(1 + randMkt, 10) - 1) / randMkt)
      : buyAnnualSavings * 10;
    const simBuyCapGains = Math.max(0, simBuySavingsFv - (buyAnnualSavings * 10));
    const simBuySavingsPostTax = (buyAnnualSavings * 10) + (simBuyCapGains * 0.85);
    const simBuyNw = simBuyEquity + simBuySavingsPostTax;

    if (simBuyNw >= simRentNw) {
      monteCarloWins++;
    }
  }
  const monteCarloConfidenceScore = Math.round((monteCarloWins / iterations) * 100);

  const institutional: InstitutionalAnalysis = {
    unrecoverableBuyMonthly,
    unrecoverableRentMonthly,
    unrecoverableDeltaMonthly,
    capitalOpportunityCostMonthly: Math.round(capitalOpportunityCostMonthly),
    principalEquityPaydownMonthly: Math.round(principalYear1Monthly),
    taxShieldMonthlyRefund,
    afterTaxMonthlyPiti,
    crossoverBreakEvenYear,
    terminalNetWorthBuy10Yr,
    terminalNetWorthRent10Yr,
    terminalNetWorthDelta10Yr,
    monteCarloConfidenceScore,
    monteCarloIterations: iterations,
  };

  const mlsSearchUrl = generateRedfinMlsUrl(
    inputs.zipCode,
    inputs.targetBeds,
    inputs.targetBaths,
    inputs.targetHomePrice
  );

  return {
    monthlyNetTakeHome,
    totalLifestyleExpenses,
    surplusCashBeforeHousing,
    rainyDayBufferTarget,
    monthlyBuyHousingCost,
    leftoverCashBufferBuy,
    housingNetPercentBuy,
    isHousePoorBuy,
    monthlyRentHousingCost,
    leftoverCashBufferRent,
    housingNetPercentRent,
    monthlyRentSavings,
    verdictStatus,
    verdictTitle,
    verdictMessage,
    isReadyToBuyToday,
    readinessYear,
    readinessTimeline,
    hedonicSpecMapping,
    expenseOptimization,
    stressTestMetrics,
    institutional,
    maxSafeHomePrice,
    mlsSearchUrl,
  };
}

export function generateNetWorthComparison(inputs: UserHousingInputs): NetWorthProjectionPoint[] {
  const points: NetWorthProjectionPoint[] = [];
  const mortgage = calculateMortgagePiti(inputs);
  const homePrice = inputs.targetHomePrice || 850000;
  const downPayment = mortgage.downPaymentAmount;
  const monthlyRent = inputs.currentRent || 3000;
  const monthlyBuyPiti = mortgage.totalMonthlyPiti;

  const homeAppreciationRate = 0.04; // 4% annual home value growth
  const investmentReturnRate = 0.07; // 7% S&P 500 annual return
  const monthlyInvestmentReturn = investmentReturnRate / 12;

  let currentHomeValue = homePrice;
  let remainingLoanBalance = mortgage.loanAmount;
  let rentInvestedPortfolio = downPayment; // Renter starts with down payment in stock market

  const annualInterestRate = inputs.interestRate || 6.5;
  const monthlyInterestRate = annualInterestRate / 100 / 12;

  for (let year = 0; year <= 10; year++) {
    if (year === 0) {
      points.push({
        year: 0,
        buyNetWorth: downPayment,
        rentAndInvestNetWorth: downPayment,
      });
      continue;
    }

    // Simulate 12 months for year
    for (let m = 1; m <= 12; m++) {
      // Home appreciation
      currentHomeValue *= 1 + homeAppreciationRate / 12;

      // Mortgage amortization
      if (remainingLoanBalance > 0) {
        const interestForMonth = remainingLoanBalance * monthlyInterestRate;
        const principalForMonth = Math.min(
          remainingLoanBalance,
          mortgage.monthlyPrincipalAndInterest - interestForMonth
        );
        remainingLoanBalance -= principalForMonth;
      }

      // Rent & Invest Portfolio Growth
      rentInvestedPortfolio *= 1 + monthlyInvestmentReturn;
      // If buying costs more than renting, renter invests the monthly savings
      const monthlyDifference = monthlyBuyPiti - monthlyRent;
      if (monthlyDifference > 0) {
        rentInvestedPortfolio += monthlyDifference;
      }
    }

    const buyNetWorth = Math.round(currentHomeValue - remainingLoanBalance);
    const rentAndInvestNetWorth = Math.round(rentInvestedPortfolio);

    points.push({
      year,
      buyNetWorth,
      rentAndInvestNetWorth,
    });
  }

  return points;
}
