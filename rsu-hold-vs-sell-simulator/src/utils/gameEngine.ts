import { Badge, GameChoice, GameQuarterState, SimulationOutput } from '../types';

export function initializeGameState(simulation: SimulationOutput): GameQuarterState[] {
  return simulation.events.map(e => ({
    quarterIndex: e.quarterIndex,
    quarterLabel: e.quarterLabel,
    date: e.date,
    stockPriceAtVest: e.stockPriceAtVest,
    sp500PriceAtVest: e.sp500PriceAtVest,
    netSharesVested: e.netSharesVested,
    netProceedsAtVest: e.netProceedsAtVest,
  }));
}

export function computeGameScore(
  choices: Record<number, GameChoice>,
  simulation: SimulationOutput
): {
  userFinalValue: number;
  optimalFinalValue: number;
  sp500FinalValue: number;
  holdFinalValue: number;
  regretIndex: number; // 0 to 100
  regretCategory: 'Zero Regret' | 'Low Regret' | 'Moderate Regret' | 'High Regret';
  badges: Badge[];
} {
  const events = simulation.events;
  const currentStockPrice = simulation.currentPrices.stock;
  const currentSp500Price = simulation.currentPrices.sp500;

  let totalUserStockShares = 0;
  let totalUserSp500Shares = 0;

  let totalHoldChoices = 0;
  let totalSellChoices = 0;

  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    const choice = choices[e.quarterIndex] || 'sell_sp500';

    if (choice === 'hold') {
      totalUserStockShares += e.netSharesVested;
      totalHoldChoices++;
    } else if (choice === 'sell_sp500') {
      totalUserSp500Shares += e.netProceedsAtVest / e.sp500PriceAtVest;
      totalSellChoices++;
    } else if (choice === 'split5050') {
      totalUserStockShares += e.netSharesVested * 0.5;
      totalUserSp500Shares += (e.netProceedsAtVest * 0.5) / e.sp500PriceAtVest;
      totalHoldChoices += 0.5;
      totalSellChoices += 0.5;
    }
  }

  const userFinalValue = Math.round(
    totalUserStockShares * currentStockPrice + totalUserSp500Shares * currentSp500Price
  );

  const holdFinalValue = simulation.strategies.hold.finalValueToday;
  const sp500FinalValue = simulation.strategies.sp500.finalValueToday;
  const optimalFinalValue = Math.max(holdFinalValue, sp500FinalValue);

  // Prospect Theory Regret Index (Loss Aversion multiplier = 2.25x)
  const shortfall = optimalFinalValue - userFinalValue;
  const maxPossibleShortfall = Math.abs(holdFinalValue - sp500FinalValue) || 1;
  let regretRatio = shortfall / maxPossibleShortfall;
  if (regretRatio < 0) regretRatio = 0;
  if (regretRatio > 1) regretRatio = 1;

  const regretIndex = Math.round(regretRatio * 100);

  let regretCategory: 'Zero Regret' | 'Low Regret' | 'Moderate Regret' | 'High Regret' = 'Zero Regret';
  if (regretIndex > 65) regretCategory = 'High Regret';
  else if (regretIndex > 35) regretCategory = 'Moderate Regret';
  else if (regretIndex > 10) regretCategory = 'Low Regret';

  // Badges
  const badges: Badge[] = [
    {
      id: 'wisdom_masses',
      title: 'Wisdom of the Masses',
      description: 'Divested >70% of vest proceeds into S&P 500 broad market index.',
      icon: 'ShieldCheck',
      unlocked: totalSellChoices / events.length >= 0.7,
    },
    {
      id: 'diamond_hands',
      title: 'Diamond Hands',
      description: 'Held company stock across >70% of all vest quarters.',
      icon: 'Gem',
      unlocked: totalHoldChoices / events.length >= 0.7,
    },
    {
      id: 'balanced_tactician',
      title: 'Balanced Tactician',
      description: 'Utilized 50/50 hybrid split strategy for smooth diversification.',
      icon: 'Scale',
      unlocked: Object.values(choices).filter(c => c === 'split5050').length >= 4,
    },
    {
      id: 'regret_minimizer',
      title: 'Regret Minimizer',
      description: 'Achieved a low regret index score (< 20%).',
      icon: 'Award',
      unlocked: regretIndex <= 20,
    },
  ];

  return {
    userFinalValue,
    optimalFinalValue,
    sp500FinalValue,
    holdFinalValue,
    regretIndex,
    regretCategory,
    badges,
  };
}
