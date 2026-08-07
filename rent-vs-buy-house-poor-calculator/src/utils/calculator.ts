import type { UserHousingInputs, MortgageBreakdown, HousePoorAnalysis, NetWorthProjectionPoint } from '../types';
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

  // 2. Buy Costs & Buffer
  const mortgage = calculateMortgagePiti(inputs);
  const monthlyBuyHousingCost = mortgage.totalMonthlyPiti;
  const leftoverCashBufferBuy = surplusCashBeforeHousing - monthlyBuyHousingCost;
  const housingNetPercentBuy = monthlyNetTakeHome > 0 ? (monthlyBuyHousingCost / monthlyNetTakeHome) * 100 : 0;
  const isHousePoorBuy = leftoverCashBufferBuy < 500 || housingNetPercentBuy > 45;

  // 3. Rent Costs & Buffer
  const monthlyRentHousingCost = inputs.currentRent || 3000;
  const leftoverCashBufferRent = surplusCashBeforeHousing - monthlyRentHousingCost;
  const housingNetPercentRent = monthlyNetTakeHome > 0 ? (monthlyRentHousingCost / monthlyNetTakeHome) * 100 : 0;

  // Monthly Cash Flow Difference (Buying Cost vs Renting Cost)
  const monthlyRentSavings = monthlyBuyHousingCost - monthlyRentHousingCost;

  // 4. Decision Verdict Logic
  let verdictStatus: 'buy' | 'caution' | 'rent_recommended' = 'buy';
  let verdictTitle = '';
  let verdictMessage = '';

  if (isHousePoorBuy) {
    verdictStatus = 'rent_recommended';
    verdictTitle = '🚫 RECOMMENDATION: KEEP RENTING!';
    verdictMessage = `Buying this home at $${Math.round(monthlyBuyHousingCost).toLocaleString()}/mo will leave you HOUSE POOR with only $${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo in cash buffer. Renting at $${Math.round(monthlyRentHousingCost).toLocaleString()}/mo saves you $${Math.round(monthlyRentSavings).toLocaleString()}/mo in cash flow, which you can safely invest!`;
  } else if (leftoverCashBufferBuy < 1500 || housingNetPercentBuy > 35) {
    verdictStatus = 'caution';
    verdictTitle = '🟡 PROCEED WITH CAUTION';
    verdictMessage = `Buying this home takes ${housingNetPercentBuy.toFixed(1)}% of your net paycheck, leaving $${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo in cash buffer. You can afford it, but unexpected home repairs may feel tight.`;
  } else {
    verdictStatus = 'buy';
    verdictTitle = '🟢 GREAT FIT: YOU CAN SAFELY BUY THIS HOME!';
    verdictMessage = `Housing consumes only ${housingNetPercentBuy.toFixed(1)}% of your net income, leaving a healthy $${Math.round(leftoverCashBufferBuy).toLocaleString()}/mo leftover cash buffer for savings and lifestyle!`;
  }

  // 5. Max Safe Purchase Price calculation (where PITI = Surplus Cash - $1,500 safety buffer)
  const targetMaxMonthlyPiti = Math.max(1000, surplusCashBeforeHousing - 1500);
  // Estimate max price using simple inverse ratio
  const maxSafeHomePrice = Math.round((inputs.targetHomePrice || 850000) * (targetMaxMonthlyPiti / Math.max(1, monthlyBuyHousingCost)));

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
