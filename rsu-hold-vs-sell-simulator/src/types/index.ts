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
  totalShares: number;
  startYear: number;
  startQuarter: number; // 1, 2, 3, 4
  hasCliff: boolean; // true = 25% at month 12, then quarterly; false = continuous 1/16th quarterly
  vestingYears: number; // default 4
  taxWithholdingPct: number; // default 35%
  statePreset: 'CA' | 'NY' | 'WA' | 'TX' | 'Custom';
  capitalGainsTaxPct: number; // default 20%
  cashApyPct: number; // default 4.5%
  sectorEtf: string;
  // Refresher Grants Stacking
  enableRefresherGrants: boolean;
  annualRefresherShares: number; // e.g. 100 shares granted each year in Q1 of Y2, Y3, Y4
}

export interface PricePoint {
  date: string; // YYYY-MM-DD
  quarterLabel: string; // e.g. "2019 Q1"
  year: number;
  quarter: number;
  price: number;
}

export interface StockPriceData {
  symbol: string;
  prices: PricePoint[];
}

export interface VestEvent {
  quarterIndex: number; // 1 to 16
  date: string; // YYYY-MM-DD
  quarterLabel: string;
  year: number;
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
}

export interface TrajectoryPoint {
  quarterLabel: string;
  date: string;
  stockPrice: number;
  sp500Price: number;
  holdValue: number;
  sp500Value: number;
  nasdaqValue: number;
  sectorValue: number;
  cashValue: number;
  hybrid5050Value: number;
}

export interface StrategyResult {
  key: StrategyKey;
  name: string;
  description: string;
  finalValueToday: number;
  totalProceedsInvested: number;
  netGainToday: number;
  returnPct: number;
  maxDrawdownPct: number;
  concentrationRiskScore: 'Low' | 'Moderate' | 'High' | 'Severe';
  concentrationRiskPct: number;
  color: string;
}

export interface SimulationOutput {
  config: RSUGrantConfig;
  events: VestEvent[];
  strategies: Record<StrategyKey, StrategyResult>;
  trajectory: TrajectoryPoint[];
  opportunityCostOfHolding: number; // (sp500Value - holdValue)
  opportunityCostPct: number;
  bestStrategyKey: StrategyKey;
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
