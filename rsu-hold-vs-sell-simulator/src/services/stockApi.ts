import { PricePoint, StockPriceData, StockTicker } from '../types';

export const POPULAR_TICKERS: StockTicker[] = [
  { symbol: 'GOOGL', name: 'Alphabet Inc. (Google)', sector: 'Communication Services', sectorEtf: 'XLC', sectorEtfName: 'Communication Services Select Sector SPDR' },
  { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'MSFT', name: 'Microsoft Corp.', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'META', name: 'Meta Platforms Inc.', sector: 'Communication Services', sectorEtf: 'XLC', sectorEtfName: 'Communication Services Select Sector SPDR' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer Cyclical', sectorEtf: 'XLY', sectorEtfName: 'Consumer Discretionary Select Sector SPDR' },
  { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Consumer Cyclical', sectorEtf: 'XLY', sectorEtfName: 'Consumer Discretionary Select Sector SPDR' },
  { symbol: 'NFLX', name: 'Netflix Inc.', sector: 'Communication Services', sectorEtf: 'XLC', sectorEtfName: 'Communication Services Select Sector SPDR' },
  { symbol: 'ADBE', name: 'Adobe Inc.', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'CRM', name: 'Salesforce Inc.', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'PLTR', name: 'Palantir Technologies', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
  { symbol: 'UBER', name: 'Uber Technologies', sector: 'Technology', sectorEtf: 'XLK', sectorEtfName: 'Technology Select Sector SPDR Fund' },
];

export const BENCHMARKS: Record<string, { symbol: string; name: string }> = {
  SPY: { symbol: 'SPY', name: 'S&P 500 ETF (SPY)' },
  QQQ: { symbol: 'QQQ', name: 'Nasdaq 100 ETF (QQQ)' },
  XLK: { symbol: 'XLK', name: 'Technology ETF (XLK)' },
  XLC: { symbol: 'XLC', name: 'Communication ETF (XLC)' },
  XLY: { symbol: 'XLY', name: 'Consumer Discretionary ETF (XLY)' },
};

// v2: monthly bars with dividend-adjusted closes. Bumping the prefix invalidates old quarterly caches.
const CACHE_PREFIX = 'rsu_stock_cache_v2_';
const CACHE_TTL_MS = 12 * 60 * 60 * 1000; // 12 hours

interface CacheItem {
  timestamp: number;
  data: StockPriceData;
}

// Memory fallback cache (live data only)
const memoryCache = new Map<string, StockPriceData>();

function isValidCachedData(data: StockPriceData | undefined): data is StockPriceData {
  return (
    !!data &&
    data.source === 'live' &&
    Array.isArray(data.prices) &&
    data.prices.length > 0 &&
    typeof data.prices[0].adjClose === 'number'
  );
}

function getFromCache(symbol: string): StockPriceData | null {
  const upper = symbol.toUpperCase();
  if (memoryCache.has(upper)) {
    return memoryCache.get(upper)!;
  }

  try {
    const raw = localStorage.getItem(CACHE_PREFIX + upper);
    if (raw) {
      const item: CacheItem = JSON.parse(raw);
      if (Date.now() - item.timestamp < CACHE_TTL_MS && isValidCachedData(item.data)) {
        memoryCache.set(upper, item.data);
        return item.data;
      }
    }
  } catch (e) {
    console.warn('LocalStorage read error:', e);
  }
  return null;
}

function saveToCache(symbol: string, data: StockPriceData): void {
  // Never persist synthetic data: a transient network/CORS failure must not masquerade as real prices.
  if (data.source !== 'live') return;
  const upper = symbol.toUpperCase();
  memoryCache.set(upper, data);
  try {
    const item: CacheItem = {
      timestamp: Date.now(),
      data,
    };
    localStorage.setItem(CACHE_PREFIX + upper, JSON.stringify(item));
  } catch (e) {
    console.warn('LocalStorage write error:', e);
  }
}

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * Parse a Yahoo Finance chart payload (monthly bars) into ascending monthly price points.
 * Exported for testing.
 */
export function parseYahooChart(json: any): PricePoint[] {
  const result = json?.chart?.result?.[0];
  const timestamps: number[] | undefined = result?.timestamp;
  const closes: (number | null)[] | undefined = result?.indicators?.quote?.[0]?.close;
  const adjCloses: (number | null)[] | undefined = result?.indicators?.adjclose?.[0]?.adjclose;
  if (!timestamps || !closes) return [];

  // Keyed by YYYY-MM; later entries win so a trailing live quote overrides a stale partial bar.
  const byMonth = new Map<string, PricePoint>();

  for (let i = 0; i < timestamps.length; i++) {
    const close = closes[i];
    if (!isFiniteNumber(close) || close <= 0) continue;
    const adj = adjCloses?.[i];

    // Bars are stamped at the start of the month (exchange time). Shift by 12h and read in UTC so the
    // viewer's local timezone can never roll a bar back into the previous month.
    const d = new Date((timestamps[i] + 12 * 3600) * 1000);
    const year = d.getUTCFullYear();
    const month = d.getUTCMonth() + 1;
    const key = `${year}-${String(month).padStart(2, '0')}`;

    byMonth.set(key, {
      date: `${key}-01`,
      year,
      month,
      close,
      adjClose: isFiniteNumber(adj) && adj > 0 ? adj : close,
    });
  }

  return Array.from(byMonth.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * Live Stock Price Fetcher
 * Queries Yahoo Finance API (monthly bars, full history) with CORS proxy failovers.
 */
export async function fetchStockHistory(symbol: string): Promise<StockPriceData> {
  const upperSymbol = symbol.toUpperCase();
  const cached = getFromCache(upperSymbol);
  if (cached) {
    return cached;
  }

  const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(upperSymbol)}?interval=1mo&range=max`;

  // Attempt live API endpoints
  const urls = [
    yahooUrl,
    `https://corsproxy.io/?${encodeURIComponent(yahooUrl)}`,
    `https://api.allorigins.win/raw?url=${encodeURIComponent(yahooUrl)}`,
  ];

  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const json = await res.json();
      const pricePoints = parseYahooChart(json);
      if (pricePoints.length > 0) {
        const data: StockPriceData = { symbol: upperSymbol, prices: pricePoints, source: 'live' };
        saveToCache(upperSymbol, data);
        return data;
      }
    } catch (err) {
      console.warn(`Failed fetching from ${url}:`, err);
    }
  }

  // Illustrative fallback if API is blocked by CORS/offline. Flagged as synthetic and never cached.
  console.info(`Using client-side synthetic price curve for ${upperSymbol}`);
  return generateFallbackPrices(upperSymbol);
}

const monthIndex = (year: number, month: number) => year * 12 + (month - 1);

/**
 * Synthetic fallback generator: a smooth, illustrative monthly curve (2017 -> current month) shaped like
 * the 2018-2025 market cycle. NOT real market data and does not include dividends.
 */
export function generateFallbackPrices(symbol: string, now: Date = new Date()): StockPriceData {
  const upper = symbol.toUpperCase();
  const prices: PricePoint[] = [];

  // Anchor prices per symbol (split-adjusted)
  const baseAnchors: Record<string, { start: number; peak2021: number; current: number }> = {
    GOOGL: { start: 55, peak2021: 145, current: 175 },
    GOOG: { start: 55, peak2021: 145, current: 175 },
    AAPL: { start: 42, peak2021: 175, current: 230 },
    MSFT: { start: 90, peak2021: 335, current: 415 },
    NVDA: { start: 6, peak2021: 30, current: 125 },
    META: { start: 180, peak2021: 380, current: 510 },
    AMZN: { start: 70, peak2021: 185, current: 195 },
    TSLA: { start: 20, peak2021: 380, current: 220 },
    NFLX: { start: 250, peak2021: 680, current: 650 },
    ADBE: { start: 200, peak2021: 650, current: 520 },
    CRM: { start: 120, peak2021: 300, current: 290 },
    PLTR: { start: 10, peak2021: 35, current: 32 },
    UBER: { start: 40, peak2021: 60, current: 75 },
    SPY: { start: 270, peak2021: 470, current: 580 },
    QQQ: { start: 160, peak2021: 400, current: 490 },
    XLK: { start: 65, peak2021: 170, current: 225 },
    XLC: { start: 45, peak2021: 85, current: 90 },
    XLY: { start: 100, peak2021: 210, current: 200 },
  };

  const anchor = baseAnchors[upper] || { start: 50, peak2021: 150, current: 180 };
  const dipPct = upper === 'NVDA' ? 0.5 : upper === 'META' ? 0.35 : 0.75;
  const preCovid = anchor.start + (anchor.peak2021 - anchor.start) * 0.4;

  const endYear = now.getFullYear();
  const endMonth = now.getMonth() + 1;

  // Knots: (monthIndex, price). Log-linear interpolation between knots keeps the curve continuous.
  const knots: Array<[number, number]> = [
    [monthIndex(2017, 1), anchor.start * 0.85],
    [monthIndex(2018, 1), anchor.start],
    [monthIndex(2020, 2), preCovid],
    [monthIndex(2020, 3), preCovid * 0.8], // Covid drop
    [monthIndex(2021, 12), anchor.peak2021], // 2021 peak
    [monthIndex(2022, 10), anchor.peak2021 * dipPct], // 2022 trough
    [Math.max(monthIndex(endYear, endMonth), monthIndex(2022, 11)), anchor.current],
  ];

  const startIdx = knots[0][0];
  const endIdx = knots[knots.length - 1][0];

  for (let idx = startIdx; idx <= endIdx; idx++) {
    let k = 0;
    while (k < knots.length - 2 && idx > knots[k + 1][0]) k++;
    const [i0, p0] = knots[k];
    const [i1, p1] = knots[k + 1];
    const t = i1 === i0 ? 0 : (idx - i0) / (i1 - i0);
    const price = Math.exp(Math.log(p0) + (Math.log(p1) - Math.log(p0)) * t);

    const year = Math.floor(idx / 12);
    const month = (idx % 12) + 1;
    const rounded = Math.round(price * 100) / 100;
    prices.push({
      date: `${year}-${String(month).padStart(2, '0')}-01`,
      year,
      month,
      close: rounded,
      adjClose: rounded,
    });
  }

  return { symbol: upper, prices, source: 'synthetic' };
}

/**
 * Fetch all required datasets concurrently for a company simulation
 */
export async function fetchSimulationData(
  companySymbol: string,
  sectorEtf: string
): Promise<{
  stock: StockPriceData;
  sp500: StockPriceData;
  nasdaq: StockPriceData;
  sector: StockPriceData;
}> {
  const [stock, sp500, nasdaq, sector] = await Promise.all([
    fetchStockHistory(companySymbol),
    fetchStockHistory('SPY'),
    fetchStockHistory('QQQ'),
    fetchStockHistory(sectorEtf || 'XLK'),
  ]);

  return { stock, sp500, nasdaq, sector };
}
