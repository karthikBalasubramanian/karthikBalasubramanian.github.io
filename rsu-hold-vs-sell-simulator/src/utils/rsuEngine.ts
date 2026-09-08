import {
  RSUGrantConfig,
  StockPriceData,
  VestEvent,
  TrajectoryPoint,
  StrategyResult,
  StrategyKey,
  SimulationOutput
} from '../types';

/**
 * Helper to find stock price closest to a target year & quarter label
 */
export function getPriceAtQuarter(
  priceData: StockPriceData,
  year: number,
  quarter: number
): number {
  const targetLabel = `${year} Q${quarter}`;
  const exact = priceData.prices.find(p => p.quarterLabel === targetLabel);
  if (exact) return exact.price;

  // Fallback to nearest date
  if (priceData.prices.length === 0) return 100;
  const targetDate = `${year}-${String((quarter - 1) * 3 + 1).padStart(2, '0')}-01`;
  let closest = priceData.prices[0];
  let minDiff = Math.abs(new Date(closest.date).getTime() - new Date(targetDate).getTime());

  for (const p of priceData.prices) {
    const diff = Math.abs(new Date(p.date).getTime() - new Date(targetDate).getTime());
    if (diff < minDiff) {
      minDiff = diff;
      closest = p;
    }
  }

  return closest.price;
}

export function runRsuSimulation(
  config: RSUGrantConfig,
  stockData: StockPriceData,
  sp500Data: StockPriceData,
  nasdaqData: StockPriceData,
  sectorData: StockPriceData
): SimulationOutput {
  const events: VestEvent[] = [];
  const taxRate = config.taxWithholdingPct / 100;
  const totalQuarters = config.vestingYears * 4; // 16 quarters

  const quarterlyShareFraction = 1 / totalQuarters;

  let currentYear = config.startYear;
  let currentQtr = config.startQuarter;

  for (let q = 1; q <= totalQuarters; q++) {
    // 1. Initial Grant Vesting
    let initialGrossShares = 0;
    if (config.hasCliff) {
      if (q < 4) {
        initialGrossShares = 0;
      } else if (q === 4) {
        initialGrossShares = config.totalShares * 0.25; // 25% cliff at Month 12
      } else {
        initialGrossShares = config.totalShares * quarterlyShareFraction;
      }
    } else {
      initialGrossShares = config.totalShares * quarterlyShareFraction;
    }

    // 2. Annual Refresher Grants Stacking (if enabled)
    // Refresher 1: Granted at Q5 (Year 2 Q1) -> Vests quarterly 1/16th per qtr starting Q5
    // Refresher 2: Granted at Q9 (Year 3 Q1) -> Vests quarterly 1/16th per qtr starting Q9
    // Refresher 3: Granted at Q13 (Year 4 Q1) -> Vests quarterly 1/16th per qtr starting Q13
    let refresherGrossShares = 0;
    if (config.enableRefresherGrants && config.annualRefresherShares > 0) {
      const qtrRefresherFrac = config.annualRefresherShares / 16;
      if (q >= 5) refresherGrossShares += qtrRefresherFrac; // Refresher Y2
      if (q >= 9) refresherGrossShares += qtrRefresherFrac; // Refresher Y3
      if (q >= 13) refresherGrossShares += qtrRefresherFrac; // Refresher Y4
    }

    const totalGrossShares = initialGrossShares + refresherGrossShares;
    const taxShares = Math.round(totalGrossShares * taxRate * 100) / 100;
    const netShares = totalGrossShares - taxShares;

    const stockPrice = getPriceAtQuarter(stockData, currentYear, currentQtr);
    const sp500Price = getPriceAtQuarter(sp500Data, currentYear, currentQtr);
    const nasdaqPrice = getPriceAtQuarter(nasdaqData, currentYear, currentQtr);
    const sectorPrice = getPriceAtQuarter(sectorData, currentYear, currentQtr);

    const grossValue = totalGrossShares * stockPrice;
    const netProceeds = netShares * stockPrice;

    events.push({
      quarterIndex: q,
      quarterLabel: `${currentYear} Q${currentQtr}`,
      date: `${currentYear}-${String((currentQtr - 1) * 3 + 1).padStart(2, '0')}-01`,
      year: currentYear,
      initialSharesVested: Math.round(initialGrossShares * 100) / 100,
      refresherSharesVested: Math.round(refresherGrossShares * 100) / 100,
      grossSharesVested: Math.round(totalGrossShares * 100) / 100,
      taxSharesWithheld: taxShares,
      netSharesVested: Math.round(netShares * 100) / 100,
      stockPriceAtVest: stockPrice,
      grossValueAtVest: Math.round(grossValue),
      netProceedsAtVest: Math.round(netProceeds),
      sp500PriceAtVest: sp500Price,
      nasdaqPriceAtVest: nasdaqPrice,
      sectorEtfPriceAtVest: sectorPrice,
    });

    // Increment quarter
    currentQtr++;
    if (currentQtr > 4) {
      currentQtr = 1;
      currentYear++;
    }
  }

  // Generate Wealth Trajectories across time
  const trajectory: TrajectoryPoint[] = [];

  let cumHoldShares = 0;
  let cumSp500Shares = 0;
  let cumNasdaqShares = 0;
  let cumSectorShares = 0;
  let cumCashValue = 0;
  let cumHybridShares = 0;
  let cumHybridSp500Shares = 0;

  const quarterlyCashYield = Math.pow(1 + config.cashApyPct / 100, 0.25) - 1;

  for (let i = 0; i < events.length; i++) {
    const e = events[i];

    // Compound existing cash balance
    cumCashValue = cumCashValue * (1 + quarterlyCashYield);

    if (e.netSharesVested > 0) {
      // 1. Pure Hold
      cumHoldShares += e.netSharesVested;

      // 2. Pure Sell & S&P 500
      const sp500SharesBought = e.netProceedsAtVest / e.sp500PriceAtVest;
      cumSp500Shares += sp500SharesBought;

      // 3. Pure Sell & Nasdaq 100
      const nasdaqSharesBought = e.netProceedsAtVest / e.nasdaqPriceAtVest;
      cumNasdaqShares += nasdaqSharesBought;

      // 4. Pure Sell & Sector ETF
      const sectorSharesBought = e.netProceedsAtVest / e.sectorEtfPriceAtVest;
      cumSectorShares += sectorSharesBought;

      // 5. Pure Sell & Cash
      cumCashValue += e.netProceedsAtVest;

      // 6. Hybrid 50/50
      cumHybridShares += e.netSharesVested * 0.5;
      cumHybridSp500Shares += (e.netProceedsAtVest * 0.5) / e.sp500PriceAtVest;
    }

    const holdVal = Math.round(cumHoldShares * e.stockPriceAtVest);
    const sp500Val = Math.round(cumSp500Shares * e.sp500PriceAtVest);
    const nasdaqVal = Math.round(cumNasdaqShares * e.nasdaqPriceAtVest);
    const sectorVal = Math.round(cumSectorShares * e.sectorEtfPriceAtVest);
    const cashVal = Math.round(cumCashValue);
    const hybridVal = Math.round(
      cumHybridShares * e.stockPriceAtVest + cumHybridSp500Shares * e.sp500PriceAtVest
    );

    trajectory.push({
      quarterLabel: e.quarterLabel,
      date: e.date,
      stockPrice: e.stockPriceAtVest,
      sp500Price: e.sp500PriceAtVest,
      holdValue: holdVal,
      sp500Value: sp500Val,
      nasdaqValue: nasdaqVal,
      sectorValue: sectorVal,
      cashValue: cashVal,
      hybrid5050Value: hybridVal,
    });
  }

  // Get current market prices (latest available in series)
  const currentStockPrice = stockData.prices[stockData.prices.length - 1]?.price || events[events.length - 1].stockPriceAtVest;
  const currentSp500Price = sp500Data.prices[sp500Data.prices.length - 1]?.price || events[events.length - 1].sp500PriceAtVest;
  const currentNasdaqPrice = nasdaqData.prices[nasdaqData.prices.length - 1]?.price || events[events.length - 1].nasdaqPriceAtVest;
  const currentSectorPrice = sectorData.prices[sectorData.prices.length - 1]?.price || events[events.length - 1].sectorEtfPriceAtVest;

  // Final values today
  const totalProceedsInvested = events.reduce((sum, e) => sum + e.netProceedsAtVest, 0);

  const finalHoldValue = Math.round(cumHoldShares * currentStockPrice);
  const finalSp500Value = Math.round(cumSp500Shares * currentSp500Price);
  const finalNasdaqValue = Math.round(cumNasdaqShares * currentNasdaqPrice);
  const finalSectorValue = Math.round(cumSectorShares * currentSectorPrice);
  const finalCashValue = Math.round(cumCashValue);
  const finalHybridValue = Math.round(
    cumHybridShares * currentStockPrice + cumHybridSp500Shares * currentSp500Price
  );
  // LTCG 1yr estimate
  const finalLtcgValue = Math.round(finalSp500Value * 0.96);

  // Maximum Drawdown Helper
  function calculateMaxDrawdown(values: number[]): number {
    let maxDrawdown = 0;
    let peak = values[0] || 0;
    for (const val of values) {
      if (val > peak) peak = val;
      if (peak > 0) {
        const dd = (peak - val) / peak;
        if (dd > maxDrawdown) maxDrawdown = dd;
      }
    }
    return Math.round(maxDrawdown * 100);
  }

  const holdDrawdown = calculateMaxDrawdown(trajectory.map(t => t.holdValue));
  const sp500Drawdown = calculateMaxDrawdown(trajectory.map(t => t.sp500Value));
  const nasdaqDrawdown = calculateMaxDrawdown(trajectory.map(t => t.nasdaqValue));
  const sectorDrawdown = calculateMaxDrawdown(trajectory.map(t => t.sectorValue));

  // Concentration Risk Score Assessment
  function getConcentrationRisk(drawdown: number, companySymbol: string): { label: 'Low' | 'Moderate' | 'High' | 'Severe'; pct: number } {
    if (drawdown > 50) return { label: 'Severe', pct: 90 };
    if (drawdown > 35) return { label: 'High', pct: 75 };
    if (drawdown > 20) return { label: 'Moderate', pct: 50 };
    return { label: 'Low', pct: 25 };
  }

  const strategies: Record<StrategyKey, StrategyResult> = {
    hold: {
      key: 'hold',
      name: `Pure Hold (${config.symbol})`,
      description: `Hold 100% of vested net shares in ${config.symbol} until today.`,
      finalValueToday: finalHoldValue,
      totalProceedsInvested,
      netGainToday: finalHoldValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalHoldValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: holdDrawdown,
      concentrationRiskScore: getConcentrationRisk(holdDrawdown, config.symbol).label,
      concentrationRiskPct: getConcentrationRisk(holdDrawdown, config.symbol).pct,
      color: '#38bdf8', // Sky blue
    },
    sp500: {
      key: 'sp500',
      name: 'Sell & S&P 500 (SPY)',
      description: 'Sell 100% on vest date and reinvest proceeds immediately into S&P 500 ETF.',
      finalValueToday: finalSp500Value,
      totalProceedsInvested,
      netGainToday: finalSp500Value - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalSp500Value - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: sp500Drawdown,
      concentrationRiskScore: 'Low',
      concentrationRiskPct: 15,
      color: '#10b981', // Emerald green
    },
    nasdaq: {
      key: 'nasdaq',
      name: 'Sell & Nasdaq 100 (QQQ)',
      description: 'Sell 100% on vest date and reinvest proceeds into tech-focused QQQ index.',
      finalValueToday: finalNasdaqValue,
      totalProceedsInvested,
      netGainToday: finalNasdaqValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalNasdaqValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: nasdaqDrawdown,
      concentrationRiskScore: 'Moderate',
      concentrationRiskPct: 35,
      color: '#a855f7', // Purple
    },
    sector: {
      key: 'sector',
      name: `Sell & Sector ETF (${config.sectorEtf})`,
      description: `Sell 100% on vest date and reinvest into industry sector ETF (${config.sectorEtf}).`,
      finalValueToday: finalSectorValue,
      totalProceedsInvested,
      netGainToday: finalSectorValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalSectorValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: sectorDrawdown,
      concentrationRiskScore: 'Moderate',
      concentrationRiskPct: 40,
      color: '#f59e0b', // Amber
    },
    cash: {
      key: 'cash',
      name: 'Sell & Cash (HYSA)',
      description: `Sell 100% on vest date and keep in cash earning ${config.cashApyPct}% APY.`,
      finalValueToday: finalCashValue,
      totalProceedsInvested,
      netGainToday: finalCashValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalCashValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: 0,
      concentrationRiskScore: 'Low',
      concentrationRiskPct: 0,
      color: '#64748b', // Slate gray
    },
    hybrid5050: {
      key: 'hybrid5050',
      name: 'Hybrid 50/50 Strategy',
      description: 'Hold 50% company stock, Sell 50% to reinvest into S&P 500 ETF.',
      finalValueToday: finalHybridValue,
      totalProceedsInvested,
      netGainToday: finalHybridValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalHybridValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: Math.round((holdDrawdown + sp500Drawdown) / 2),
      concentrationRiskScore: 'Moderate',
      concentrationRiskPct: 45,
      color: '#ec4899', // Pink
    },
    ltcg1yr: {
      key: 'ltcg1yr',
      name: '1-Year LTCG Hold Strategy',
      description: 'Hold stock for 1 year to qualify for long-term capital gains tax rate, then sell to S&P 500.',
      finalValueToday: finalLtcgValue,
      totalProceedsInvested,
      netGainToday: finalLtcgValue - totalProceedsInvested,
      returnPct: totalProceedsInvested > 0 ? Math.round(((finalLtcgValue - totalProceedsInvested) / totalProceedsInvested) * 100) : 0,
      maxDrawdownPct: Math.round(holdDrawdown * 0.8),
      concentrationRiskScore: 'Moderate',
      concentrationRiskPct: 50,
      color: '#06b6d4', // Cyan
    },
  };

  const oppCost = finalSp500Value - finalHoldValue;
  const oppCostPct = finalHoldValue > 0 ? Math.round((oppCost / finalHoldValue) * 100) : 0;

  // Determine best performing strategy
  let bestKey: StrategyKey = 'sp500';
  let maxVal = -Infinity;
  (Object.keys(strategies) as StrategyKey[]).forEach(k => {
    if (strategies[k].finalValueToday > maxVal) {
      maxVal = strategies[k].finalValueToday;
      bestKey = k;
    }
  });

  return {
    config,
    events,
    strategies,
    trajectory,
    opportunityCostOfHolding: oppCost,
    opportunityCostPct: oppCostPct,
    bestStrategyKey: bestKey,
    currentPrices: {
      stock: currentStockPrice,
      sp500: currentSp500Price,
      nasdaq: currentNasdaqPrice,
      sectorEtf: currentSectorPrice,
    },
  };
}
