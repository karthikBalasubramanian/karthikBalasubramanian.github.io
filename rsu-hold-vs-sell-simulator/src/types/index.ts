export type StrategyKey = 
  | 'hold' 
  | 'sp500' 
  | 'nasdaq' 
  | 'sector' 
  | 'cash' 
  | 'hybrid5050' 
  | 'ltcg1yr';

export interface StockTicker {
  symbol: string;
  name: string;
  sector: string;
  sectorEtf: string;
  sectorEtfName: string;
}

export interface RSUGrantConfig {
  symbol: string;
  companyName: string;
  totalShares: number; // split-adjusted share count (today's share basis)
  startYear: number;
  startQuarter: number; // 1, 2, 3, 4 — grant date is the first day of this quarter
  hasCliff: boolean; // true = 25% at month 12, then quarterly; false = 1/16th every quarter from month 3
  vestingYears: number; // default 4
  taxWithholdingPct: number; // default 35%
  statePreset: string; // US_STATE_RSU_TAXES key or 'Custom'
  capitalGainsTaxPct: number; // default 20% — applied to realized / liquidation gains
  cashApyPct: number; // default 4.5%
  sectorEtf: string;
  // Refresher Grants Stacking
  enableRefresherGrants: boolean;
  annualRefresherShares: number; // shares granted at the start of years 2, 3, 4; each vests quarterly over vestingYears
}

export interface PricePoint {
  date: string; // YYYY-MM-01 (month of the bar)
  year: number;
  month: number; // 1-12
  close: number; // split-adjusted month-end close (latest price for the current month)
  adjClose: number; // split- and dividend-adjusted close (total return)
}

export type DataSource = 'live' | 'synthetic';

export interface StockPriceData {
  symbol: string;
  prices: PricePoint[]; // monthly, ascending
  source: DataSource;
}

export type VestStatus = 'vested' | 'future';

export interface VestEvent {
  quarterIndex: number; // 1..N quarters after grant
  date: string; // YYYY-MM-DD
  quarterLabel: string;
  year: number;
  status: VestStatus; // 'future' = after today or no market data yet; excluded from all strategies
  initialSharesVested: number;
  refresherSharesVested: number;
  grossSharesVested: number;
  taxSharesWithheld: number;
  netSharesVested: number;
  stockPriceAtVest: number;
  grossValueAtVest: number;
  netProceedsAtVest: number;
  // Benchmark prices at this vest date
  sp500PriceAtVest: number;
  nasdaqPriceAtVest: number;
  sectorEtfPriceAtVest: number;
  // Total-return growth multiple (dividends reinvested) from vest date to today
  stockGrowthToToday: number;
  sp500GrowthToToday: number;
}

export interface TrajectoryPoint {
  label: string; // e.g. "Apr 2019" or "Today"
  date: string;
  stockPrice: number;
  sp500Price: number;
  holdValue: number;
  sp500Value: number;
  nasdaqValue: number;
  sectorValue: number;
  cashValue: number;
  hybrid5050Value: number;
  ltcg1yrValue: number;
}

export interface StrategyResult {
  key: StrategyKey;
  name: string;
  description: string;
  finalValueToday: number; // market value today
  afterTaxValueToday: number; // value if fully liquidated today, after capital-gains / interest tax
  totalProceedsInvested: number;
  netGainToday: number;
  returnPct: number;
  maxDrawdownPct: number; // time-weighted (not masked by new vest contributions)
  concentrationRiskScore: 'Low' | 'Moderate' | 'High' | 'Severe';
  concentrationRiskPct: number; // single-stock exposure of the portfolio today (0-100)
  color: string;
}

export interface SimulationOutput {
  config: RSUGrantConfig;
  events: VestEvent[]; // all scheduled vests (vested + future)
  vestedCount: number;
  futureCount: number;
  strategies: Record<StrategyKey, StrategyResult>;
  trajectory: TrajectoryPoint[]; // monthly, first vest -> today
  opportunityCostOfHolding: number; // (sp500Value - holdValue)
  opportunityCostPct: number;
  bestStrategyKey: StrategyKey; // ranked on after-tax liquidation value
  asOfDate: string; // date of latest market data used for "today"
  dataSources: Record<'stock' | 'sp500' | 'nasdaq' | 'sector', DataSource>;
  currentPrices: {
    stock: number;
    sp500: number;
    nasdaq: number;
    sectorEtf: number;
  };
}

export type GameChoice = 'hold' | 'sell_sp500' | 'split5050';

export interface GameQuarterState {
  quarterIndex: number;
  quarterLabel: string;
  date: string;
  stockPriceAtVest: number;
  sp500PriceAtVest: number;
  netSharesVested: number;
  netProceedsAtVest: number;
  userChoice?: GameChoice;
  userPortfolioValueAtNextQuarter?: number;
}

export interface Badge {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlocked: boolean;
}
