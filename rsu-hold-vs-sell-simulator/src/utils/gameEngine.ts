import { Badge, GameChoice, GameQuarterState, SimulationOutput, VestEvent } from '../types/index';

/** Only quarters that have actually vested (with market data) are playable. */
export function getPlayableEvents(simulation: SimulationOutput): VestEvent[] {
  return simulation.events.filter(e => e.status === 'vested' && e.netSharesVested > 0);
}

export function initializeGameState(simulation: SimulationOutput): GameQuarterState[] {
  return getPlayableEvents(simulation).map(e => ({
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
  const events = getPlayableEvents(simulation);

  let userValue = 0;
  let optimalValue = 0;
  let worstValue = 0;

  let totalHoldChoices = 0;
  let totalSellChoices = 0;

  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    const choice = choices[e.quarterIndex] || 'sell_sp500';

    // Value today of this quarter's net proceeds under each choice (total return, dividends reinvested)
    const holdToday = e.netProceedsAtVest * e.stockGrowthToToday;
    const sellToday = e.netProceedsAtVest * e.sp500GrowthToToday;

    optimalValue += Math.max(holdToday, sellToday);
    worstValue += Math.min(holdToday, sellToday);

    if (choice === 'hold') {
      userValue += holdToday;
      totalHoldChoices++;
    } else if (choice === 'sell_sp500') {
      userValue += sellToday;
      totalSellChoices++;
    } else if (choice === 'split5050') {
      userValue += 0.5 * holdToday + 0.5 * sellToday;
      totalHoldChoices += 0.5;
      totalSellChoices += 0.5;
    }
  }

  const userFinalValue = Math.round(userValue);

  const holdFinalValue = simulation.strategies.hold.finalValueToday;
  const sp500FinalValue = simulation.strategies.sp500.finalValueToday;
  // Hindsight optimum: the better of hold vs sell picked independently for every quarter
  const optimalFinalValue = Math.round(optimalValue);

  // Regret = share of the avoidable gap (worst possible -> perfect hindsight) that was left on the table
  const shortfall = optimalValue - userValue;
  const maxPossibleShortfall = optimalValue - worstValue || 1;
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
      unlocked: events.length > 0 && totalSellChoices / events.length >= 0.7,
    },
    {
      id: 'diamond_hands',
      title: 'Diamond Hands',
      description: 'Held company stock across >70% of all vest quarters.',
      icon: 'Gem',
      unlocked: events.length > 0 && totalHoldChoices / events.length >= 0.7,
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
