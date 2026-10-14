import {
  RSUGrantConfig,
  StockPriceData,
  PricePoint,
  VestEvent,
  TrajectoryPoint,
  StrategyResult,
  StrategyKey,
  SimulationOutput
} from '../types/index';

/* -------------------------------------------------------------------------------------------------
 * Conventions
 * - Time is tracked as a month index (year * 12 + month - 1). Every transaction happens on the 1st of
 *   a month and is priced at the previous month's closing bar (i.e. the last close before that date).
 * - `close` is the split-adjusted price used for vest value / tax withholding.
 * - `adjClose` is dividend-adjusted. All positions are held as "total-return units" = dollars / adjClose,
 *   so dividends are reinvested for every asset (company stock and ETFs alike).
 * ------------------------------------------------------------------------------------------------- */

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const toMonthIdx = (year: number, month: number) => year * 12 + (month - 1);
const fromMonthIdx = (idx: number) => ({ year: Math.floor(idx / 12), month: (idx % 12) + 1 });
const pad2 = (n: number) => String(n).padStart(2, '0');
const monthDate = (idx: number) => {
  const { year, month } = fromMonthIdx(idx);
  return `${year}-${pad2(month)}-01`;
};
const monthLabel = (idx: number) => {
  const { year, month } = fromMonthIdx(idx);
  return `${MONTH_LABELS[month - 1]} ${year}`;
};
const round2 = (v: number) => Math.round(v * 100) / 100;

interface PriceSeries {
  byIdx: Map<number, PricePoint>;
  firstIdx: number;
  lastIdx: number;
  latest: PricePoint | null;
}

function indexSeries(data: StockPriceData): PriceSeries {
  const byIdx = new Map<number, PricePoint>();
  for (const p of data.prices) byIdx.set(toMonthIdx(p.year, p.month), p);
  const first = data.prices[0];
  const latest = data.prices[data.prices.length - 1] ?? null;
  return {
    byIdx,
    firstIdx: first ? toMonthIdx(first.year, first.month) : Infinity,
    lastIdx: latest ? toMonthIdx(latest.year, latest.month) : -Infinity,
    latest,
  };
}

/**
 * Market price on the 1st day of month `idx` = the previous month's closing bar.
 * - Small internal data gaps (≤ 3 months) fall back to the most recent earlier bar.
 * - If the asset had not started trading yet (pre-IPO / ETF launch) the first available bar is used,
 *   i.e. the purchase is deferred to the first tradable price.
 * - Dates beyond the available data return null (never extrapolated).
 */
function priceOnFirstOf(series: PriceSeries, idx: number): PricePoint | null {
  if (idx - 1 > series.lastIdx) return null;
  if (idx - 1 < series.firstIdx) return series.byIdx.get(series.firstIdx) ?? null;
  for (let back = 1; back <= 3; back++) {
    const p = series.byIdx.get(idx - back);
    if (p) return p;
  }
  return null;
}

type Asset = 'stock' | 'spy' | 'qqq' | 'sector';
const ASSETS: Asset[] = ['stock', 'spy', 'qqq', 'sector'];
type AssetPrices = Record<Asset, PricePoint>;
type AdjPrices = Record<Asset, number>;

interface Lot {
  asset: Asset;
  units: number; // total-return units (dollars / adjClose at purchase)
  basis: number; // cost basis in dollars
  acqIdx: number; // month index acquired
}

interface Book {
  lots: Lot[];
  cash: number;
  cashPrincipal: number;
  // Time-weighted (contribution-neutral) performance index used for drawdown
  twrIndex: number;
  twrPeak: number;
  maxDrawdown: number;
  prevPostValue: number;
}

type Allocation = Partial<Record<Asset | 'cash', number>>;

const STRATEGY_ALLOCATION: Record<StrategyKey, Allocation> = {
  hold: { stock: 1 },
  sp500: { spy: 1 },
  nasdaq: { qqq: 1 },
  sector: { sector: 1 },
  cash: { cash: 1 },
  hybrid5050: { stock: 0.5, spy: 0.5 },
  ltcg1yr: { stock: 1 }, // each tranche migrates to SPY after 12 months (see step())
};
const STRATEGY_KEYS = Object.keys(STRATEGY_ALLOCATION) as StrategyKey[];

const LTCG_HOLDING_MONTHS = 12;

export function runRsuSimulation(
  config: RSUGrantConfig,
  stockData: StockPriceData,
  sp500Data: StockPriceData,
  nasdaqData: StockPriceData,
  sectorData: StockPriceData,
  now: Date = new Date()
): SimulationOutput {
  const series: Record<Asset, PriceSeries> = {
    stock: indexSeries(stockData),
    spy: indexSeries(sp500Data),
    qqq: indexSeries(nasdaqData),
    sector: indexSeries(sectorData),
  };

  const pricesAt = (idx: number): AssetPrices | null => {
    const out: Partial<AssetPrices> = {};
    for (const a of ASSETS) {
      const p = priceOnFirstOf(series[a], idx);
      if (!p) return null;
      out[a] = p;
    }
    return out as AssetPrices;
  };
  const toAdj = (p: AssetPrices): AdjPrices => ({
    stock: p.stock.adjClose,
    spy: p.spy.adjClose,
    qqq: p.qqq.adjClose,
    sector: p.sector.adjClose,
  });

  const latestPrices: AssetPrices | null = ASSETS.every(a => series[a].latest)
    ? {
        stock: series.stock.latest!,
        spy: series.spy.latest!,
        qqq: series.qqq.latest!,
        sector: series.sector.latest!,
      }
    : null;
  const latestAdj = latestPrices ? toAdj(latestPrices) : null;

  const withholdingRate = Math.min(1, Math.max(0, config.taxWithholdingPct / 100));
  const cgRate = Math.min(1, Math.max(0, config.capitalGainsTaxPct / 100));
  // Short-term gains and cash interest are ordinary income (no FICA): approximate with the withholding
  // rate minus FICA (7.65%), never below the long-term capital gains rate.
  const ordinaryRate = Math.max(cgRate, withholdingRate - 0.0765);

  const nowIdx = toMonthIdx(now.getFullYear(), now.getMonth() + 1);
  const nowTime = now.getTime();

  /* ---------------------------------------------------------------------------------------------
   * 1. Vesting schedule
   *    Grant date = 1st day of startYear/startQuarter. Vest k happens k quarters after the grant, so
   *    the cliff (vest 4) lands at month 12 and the final initial-grant vest at month 48.
   *    Refreshers are granted at the start of years 2..N, each vesting quarterly over vestingYears
   *    (no cliff) on its own schedule — they are not truncated when the initial grant finishes.
   * --------------------------------------------------------------------------------------------- */
  const quartersPerGrant = Math.max(1, Math.round(config.vestingYears * 4));
  const refresherCount =
    config.enableRefresherGrants && config.annualRefresherShares > 0
      ? Math.max(0, Math.round(config.vestingYears) - 1)
      : 0;
  const totalQuarters = quartersPerGrant + 4 * refresherCount;
  const grantIdx = toMonthIdx(config.startYear, (config.startQuarter - 1) * 3 + 1);

  const events: VestEvent[] = [];
  const contributionsByIdx = new Map<number, number>();
  let firstVestedIdx: number | null = null;
  let totalProceedsInvested = 0;

  for (let k = 1; k <= totalQuarters; k++) {
    let initialGrossShares = 0;
    if (k <= quartersPerGrant) {
      if (config.hasCliff && quartersPerGrant >= 4) {
        if (k < 4) initialGrossShares = 0;
        else if (k === 4) initialGrossShares = (config.totalShares * 4) / quartersPerGrant; // 1-year cliff (25% of a 4-yr grant)
        else initialGrossShares = config.totalShares / quartersPerGrant;
      } else {
        initialGrossShares = config.totalShares / quartersPerGrant;
      }
    }

    let refresherGrossShares = 0;
    for (let r = 1; r <= refresherCount; r++) {
      const grantQuarterOffset = 4 * r; // granted at the start of year r+1
      if (k > grantQuarterOffset && k <= grantQuarterOffset + quartersPerGrant) {
        refresherGrossShares += config.annualRefresherShares / quartersPerGrant;
      }
    }

    const totalGrossShares = initialGrossShares + refresherGrossShares;
    const taxShares = totalGrossShares * withholdingRate;
    const netShares = totalGrossShares - taxShares;

    const vestIdx = grantIdx + 3 * k;
    const { year, month } = fromMonthIdx(vestIdx);
    const quarter = Math.floor((month - 1) / 3) + 1;
    const vestTime = new Date(year, month - 1, 1).getTime();

    // Only vests that have actually happened (and have market data) count toward any strategy.
    const prices = vestTime <= nowTime ? pricesAt(vestIdx) : null;
    const isVested = prices !== null;

    const stockPrice = prices?.stock.close ?? 0;
    const grossValue = totalGrossShares * stockPrice;
    const netProceeds = netShares * stockPrice;

    if (isVested && netProceeds > 0) {
      contributionsByIdx.set(vestIdx, (contributionsByIdx.get(vestIdx) ?? 0) + netProceeds);
      totalProceedsInvested += netProceeds;
      if (firstVestedIdx === null) firstVestedIdx = vestIdx;
    }

    events.push({
      quarterIndex: k,
      quarterLabel: `${year} Q${quarter}`,
      date: monthDate(vestIdx),
      year,
      status: isVested ? 'vested' : 'future',
      initialSharesVested: round2(initialGrossShares),
      refresherSharesVested: round2(refresherGrossShares),
      grossSharesVested: round2(totalGrossShares),
      taxSharesWithheld: round2(taxShares),
      netSharesVested: round2(netShares),
      stockPriceAtVest: round2(stockPrice),
      grossValueAtVest: Math.round(grossValue),
      netProceedsAtVest: Math.round(netProceeds),
      sp500PriceAtVest: round2(prices?.spy.close ?? 0),
      nasdaqPriceAtVest: round2(prices?.qqq.close ?? 0),
      sectorEtfPriceAtVest: round2(prices?.sector.close ?? 0),
      stockGrowthToToday: prices && latestAdj ? latestAdj.stock / prices.stock.adjClose : 0,
      sp500GrowthToToday: prices && latestAdj ? latestAdj.spy / prices.spy.adjClose : 0,
    });
  }

  /* ---------------------------------------------------------------------------------------------
   * 2. Month-by-month portfolio simulation, from the first vest through today
   * --------------------------------------------------------------------------------------------- */
  const books = Object.fromEntries(
    STRATEGY_KEYS.map(k => [
      k,
      { lots: [], cash: 0, cashPrincipal: 0, twrIndex: 1, twrPeak: 1, maxDrawdown: 0, prevPostValue: 0 } as Book,
    ])
  ) as Record<StrategyKey, Book>;

  const bookValue = (b: Book, adj: AdjPrices) =>
    b.lots.reduce((sum, lot) => sum + lot.units * adj[lot.asset], 0) + b.cash;
  const stockValue = (b: Book, adj: AdjPrices) =>
    b.lots.reduce((sum, lot) => sum + (lot.asset === 'stock' ? lot.units * adj.stock : 0), 0);

  const step = (adj: AdjPrices, stepIdx: number, cashGrowth: number, contribution: number) => {
    for (const key of STRATEGY_KEYS) {
      const b = books[key];
      b.cash *= cashGrowth;

      if (key === 'ltcg1yr') {
        // Sell company-stock tranches once they qualify for long-term treatment, pay LTCG on the
        // appreciation since vest (cost basis = FMV at vest), and reinvest the remainder in SPY.
        const kept: Lot[] = [];
        const added: Lot[] = [];
        for (const lot of b.lots) {
          if (lot.asset === 'stock' && stepIdx - lot.acqIdx >= LTCG_HOLDING_MONTHS) {
            const proceeds = lot.units * adj.stock;
            const tax = Math.max(0, proceeds - lot.basis) * cgRate; // losses: no credit modeled
            const reinvest = proceeds - tax;
            added.push({ asset: 'spy', units: reinvest / adj.spy, basis: reinvest, acqIdx: stepIdx });
          } else {
            kept.push(lot);
          }
        }
        b.lots = kept.concat(added);
      }

      // Time-weighted return: measure performance before this step's new money is added.
      const preValue = bookValue(b, adj);
      if (b.prevPostValue > 0) {
        b.twrIndex *= preValue / b.prevPostValue;
        if (b.twrIndex > b.twrPeak) b.twrPeak = b.twrIndex;
        const dd = 1 - b.twrIndex / b.twrPeak;
        if (dd > b.maxDrawdown) b.maxDrawdown = dd;
      }

      if (contribution > 0) {
        for (const [target, frac] of Object.entries(STRATEGY_ALLOCATION[key]) as [Asset | 'cash', number][]) {
          const dollars = contribution * frac;
          if (target === 'cash') {
            b.cash += dollars;
            b.cashPrincipal += dollars;
          } else {
            b.lots.push({ asset: target, units: dollars / adj[target], basis: dollars, acqIdx: stepIdx });
          }
        }
      }

      b.prevPostValue = bookValue(b, adj);
    }
  };

  const trajectory: TrajectoryPoint[] = [];
  const pushPoint = (label: string, date: string, p: AssetPrices, adj: AdjPrices) => {
    trajectory.push({
      label,
      date,
      stockPrice: round2(p.stock.close),
      sp500Price: round2(p.spy.close),
      holdValue: Math.round(bookValue(books.hold, adj)),
      sp500Value: Math.round(bookValue(books.sp500, adj)),
      nasdaqValue: Math.round(bookValue(books.nasdaq, adj)),
      sectorValue: Math.round(bookValue(books.sector, adj)),
      cashValue: Math.round(bookValue(books.cash, adj)),
      hybrid5050Value: Math.round(bookValue(books.hybrid5050, adj)),
      ltcg1yrValue: Math.round(bookValue(books.ltcg1yr, adj)),
    });
  };

  const monthlyCashGrowth = Math.pow(1 + config.cashApyPct / 100, 1 / 12);
  let lastStepTime: number | null = null;
  let pendingCashGrowth = 1;

  if (firstVestedIdx !== null) {
    for (let idx = firstVestedIdx; idx <= nowIdx; idx++) {
      if (idx > firstVestedIdx) pendingCashGrowth *= monthlyCashGrowth;
      const p = pricesAt(idx);
      if (!p) continue; // data gap: cash interest keeps accruing, valuation resumes next priced month
      const adj = toAdj(p);
      step(adj, idx, pendingCashGrowth, contributionsByIdx.get(idx) ?? 0);
      pendingCashGrowth = 1;
      const { year, month } = fromMonthIdx(idx);
      lastStepTime = new Date(year, month - 1, 1).getTime();
      pushPoint(monthLabel(idx), monthDate(idx), p, adj);
    }

    // Final "Today" mark-to-market using the latest available prices; cash accrues to today.
    if (latestPrices && latestAdj && lastStepTime !== null) {
      const days = Math.max(0, (nowTime - lastStepTime) / 86_400_000);
      const cashGrowth = Math.pow(1 + config.cashApyPct / 100, days / 365.25);
      step(latestAdj, nowIdx, cashGrowth, 0);
      pushPoint('Today', `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`, latestPrices, latestAdj);
    }
  }

  /* ---------------------------------------------------------------------------------------------
   * 3. Results
   * --------------------------------------------------------------------------------------------- */
  const hasResults = firstVestedIdx !== null && latestAdj !== null;
  const finalValue = (key: StrategyKey) => (hasResults ? Math.round(bookValue(books[key], latestAdj!)) : 0);

  // Tax owed if every position were sold today. Short-term vs long-term per lot; gains and losses
  // net against each other within the strategy (no carry-forward credit). Cash interest taxed as ordinary.
  const liquidationTax = (b: Book): number => {
    if (!hasResults) return 0;
    let st = 0;
    let lt = 0;
    for (const lot of b.lots) {
      const gain = lot.units * latestAdj![lot.asset] - lot.basis;
      if (nowIdx - lot.acqIdx >= LTCG_HOLDING_MONTHS) lt += gain;
      else st += gain;
    }
    if (st < 0) {
      lt += st;
      st = 0;
    }
    if (lt < 0) {
      st += lt;
      lt = 0;
    }
    const interest = Math.max(0, b.cash - b.cashPrincipal);
    return Math.max(0, st) * ordinaryRate + Math.max(0, lt) * cgRate + interest * ordinaryRate;
  };

  const exposurePct = (key: StrategyKey) => {
    if (!hasResults) return 0;
    const total = bookValue(books[key], latestAdj!);
    return total > 0 ? Math.round((stockValue(books[key], latestAdj!) / total) * 100) : 0;
  };

  const riskLabel = (key: StrategyKey, drawdownPct: number, exposure: number): StrategyResult['concentrationRiskScore'] => {
    if (key === 'hold') return drawdownPct > 50 ? 'Severe' : 'High'; // 100% single stock is never low risk
    if (exposure >= 50) return drawdownPct > 50 ? 'Severe' : 'High';
    if (exposure >= 20) return 'Moderate';
    if (key === 'nasdaq' || key === 'sector') return 'Moderate'; // sector / mega-cap tech concentration
    return 'Low';
  };

  const meta: Record<StrategyKey, { name: string; description: string; color: string }> = {
    hold: {
      name: `Pure Hold (${config.symbol})`,
      description: `Hold 100% of vested net shares in ${config.symbol} until today (dividends reinvested).`,
      color: '#38bdf8', // Sky blue
    },
    sp500: {
      name: 'Sell & S&P 500 (SPY)',
      description: 'Sell 100% on vest date and reinvest proceeds immediately into S&P 500 ETF (dividends reinvested).',
      color: '#10b981', // Emerald green
    },
    nasdaq: {
      name: 'Sell & Nasdaq 100 (QQQ)',
      description: 'Sell 100% on vest date and reinvest proceeds into tech-focused QQQ index.',
      color: '#a855f7', // Purple
    },
    sector: {
      name: `Sell & Sector ETF (${config.sectorEtf})`,
      description: `Sell 100% on vest date and reinvest into industry sector ETF (${config.sectorEtf}).`,
      color: '#f59e0b', // Amber
    },
    cash: {
      name: 'Sell & Cash (HYSA)',
      description: `Sell 100% on vest date and keep in cash earning ${config.cashApyPct}% APY, compounded to today.`,
      color: '#64748b', // Slate gray
    },
    hybrid5050: {
      name: 'Hybrid 50/50 Strategy',
      description: 'Hold 50% company stock, Sell 50% to reinvest into S&P 500 ETF.',
      color: '#ec4899', // Pink
    },
    ltcg1yr: {
      name: '1-Year LTCG Hold Strategy',
      description: `Hold each vest 12 months, then sell (paying ${config.capitalGainsTaxPct}% LTCG on gains since vest) and move into S&P 500.`,
      color: '#06b6d4', // Cyan
    },
  };

  const strategies = Object.fromEntries(
    STRATEGY_KEYS.map(key => {
      const b = books[key];
      const finalValueToday = finalValue(key);
      const afterTaxValueToday = Math.round(finalValueToday - liquidationTax(b));
      const proceeds = Math.round(totalProceedsInvested);
      const maxDrawdownPct = Math.round(b.maxDrawdown * 100);
      const exposure = exposurePct(key);
      const result: StrategyResult = {
        key,
        ...meta[key],
        finalValueToday,
        afterTaxValueToday,
        totalProceedsInvested: proceeds,
        netGainToday: finalValueToday - proceeds,
        returnPct: proceeds > 0 ? Math.round(((finalValueToday - proceeds) / proceeds) * 100) : 0,
        maxDrawdownPct,
        concentrationRiskScore: riskLabel(key, maxDrawdownPct, exposure),
        concentrationRiskPct: exposure,
      };
      return [key, result];
    })
  ) as Record<StrategyKey, StrategyResult>;

  const oppCost = strategies.sp500.finalValueToday - strategies.hold.finalValueToday;
  const oppCostPct = strategies.hold.finalValueToday > 0 ? Math.round((oppCost / strategies.hold.finalValueToday) * 100) : 0;

  // Rank on after-tax liquidation value so the LTCG strategy (which has already paid tax) is compared fairly.
  let bestKey: StrategyKey = 'sp500';
  let maxVal = -Infinity;
  for (const k of STRATEGY_KEYS) {
    if (strategies[k].afterTaxValueToday > maxVal) {
      maxVal = strategies[k].afterTaxValueToday;
      bestKey = k;
    }
  }

  const vestedCount = events.filter(e => e.status === 'vested').length;

  return {
    config,
    events,
    vestedCount,
    futureCount: events.length - vestedCount,
    strategies,
    trajectory,
    opportunityCostOfHolding: oppCost,
    opportunityCostPct: oppCostPct,
    bestStrategyKey: bestKey,
    asOfDate: `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`,
    dataSources: {
      stock: stockData.source,
      sp500: sp500Data.source,
      nasdaq: nasdaqData.source,
      sector: sectorData.source,
    },
    currentPrices: {
      stock: latestPrices?.stock.close ?? 0,
      sp500: latestPrices?.spy.close ?? 0,
      nasdaq: latestPrices?.qqq.close ?? 0,
      sectorEtf: latestPrices?.sector.close ?? 0,
    },
  };
}
