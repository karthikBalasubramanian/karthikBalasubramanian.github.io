import React, { useState } from 'react';
import { SimulationOutput, GameChoice } from '../types';
import { computeGameScore, getPlayableEvents } from '../utils/gameEngine';
import { Gamepad2, ShieldCheck, Gem, Scale, Award, RefreshCw, Sparkles, Check, ArrowRight } from 'lucide-react';

interface GameModeProps {
  simulation: SimulationOutput;
}

export const GameMode: React.FC<GameModeProps> = ({ simulation }) => {
  const [choices, setChoices] = useState<Record<number, GameChoice>>({});
  const [currentStep, setCurrentStep] = useState(1);

  const playableEvents = getPlayableEvents(simulation);
  const totalQuarters = playableEvents.length;
  const safeStep = Math.min(Math.max(1, currentStep), Math.max(1, totalQuarters));
  const currentEvent = playableEvents[safeStep - 1];

  const handleChoice = (choice: GameChoice) => {
    if (!currentEvent) return;
    setChoices(prev => ({ ...prev, [currentEvent.quarterIndex]: choice }));
    if (safeStep < totalQuarters) {
      setCurrentStep(safeStep + 1);
    }
  };

  const applyAll = (choice: GameChoice) => {
    const newChoices: Record<number, GameChoice> = {};
    playableEvents.forEach(e => {
      newChoices[e.quarterIndex] = choice;
    });
    setChoices(newChoices);
    setCurrentStep(Math.max(1, totalQuarters));
  };

  const resetGame = () => {
    setChoices({});
    setCurrentStep(1);
  };

  const score = computeGameScore(choices, simulation);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  if (totalQuarters === 0) {
    return (
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-8 shadow-xl text-center">
        <Gamepad2 className="w-8 h-8 text-emerald-400 mx-auto mb-3" />
        <h2 className="text-lg font-extrabold text-white">No vests to replay yet</h2>
        <p className="text-xs text-slate-400 mt-1">
          None of the {simulation.events.length} scheduled vests for this grant have happened yet. Pick an earlier grant year to play.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Game Header */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-emerald-500 to-teal-700 rounded-xl text-white shadow-lg shadow-emerald-950/50">
              <Gamepad2 className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-extrabold text-white">The Vesting Time Machine Game</h2>
              <p className="text-xs text-slate-400">
                Step through your {totalQuarters} vested quarters and test your decision strategy against hindsight
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => applyAll('sell_sp500')}
              className="px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800 hover:bg-emerald-900 transition-all"
            >
              Auto-Divest S&P 500
            </button>
            <button
              onClick={() => applyAll('hold')}
              className="px-3 py-1.5 rounded-lg text-xs font-bold bg-sky-950 text-sky-300 border border-sky-800 hover:bg-sky-900 transition-all"
            >
              Auto-Pure Hold
            </button>
            <button
              onClick={resetGame}
              className="p-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all"
              title="Reset Game"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Step-by-Step Decision Console */}
        <div className="lg:col-span-2 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl flex flex-col justify-between">
          <div>
            {/* Step Progress Bar */}
            <div className="flex items-center justify-between text-xs font-bold mb-2">
              <span className="text-emerald-400">
                Quarter {safeStep} of {totalQuarters}: {currentEvent?.quarterLabel}
              </span>
              <span className="text-slate-400">
                {Object.keys(choices).length} Decisions Completed
              </span>
            </div>

            <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden mb-6 border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
                style={{ width: `${(safeStep / totalQuarters) * 100}%` }}
              ></div>
            </div>

            {/* Current Quarter Vest Event Details Card */}
            {currentEvent && (
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 sm:p-5 mb-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                  <div>
                    <span className="text-xs text-slate-400">Vesting Date</span>
                    <h4 className="text-base font-extrabold text-white">{currentEvent.date} ({currentEvent.quarterLabel})</h4>
                  </div>
                  <div className="text-right">
                    <span className="text-xs text-slate-400">Net Shares Vested</span>
                    <h4 className="text-base font-extrabold text-emerald-400">{currentEvent.netSharesVested} Shares</h4>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-slate-400">Company Stock Price ({simulation.config.symbol}):</span>
                    <div className="text-lg font-bold text-white">${currentEvent.stockPriceAtVest}</div>
                  </div>
                  <div>
                    <span className="text-slate-400">S&P 500 Price (SPY):</span>
                    <div className="text-lg font-bold text-emerald-400">${currentEvent.sp500PriceAtVest}</div>
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-slate-800/80 text-xs text-slate-400">
                  Net Cash Value at Vest: <span className="text-white font-bold">{formatCurrency(currentEvent.netProceedsAtVest)}</span>
                </div>
              </div>
            )}

            {/* Decision Action Buttons */}
            <div className="space-y-3">
              <p className="text-xs font-bold text-slate-300">Choose your action for this vest event:</p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <button
                  type="button"
                  onClick={() => handleChoice('hold')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    choices[currentEvent?.quarterIndex] === 'hold'
                      ? 'bg-sky-950/80 border-sky-500 text-white shadow-lg shadow-sky-950/50'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between font-bold text-sky-400 mb-1">
                    <span>Hold Stock</span>
                    {choices[currentEvent?.quarterIndex] === 'hold' && <Check className="w-4 h-4" />}
                  </div>
                  <p className="text-[11px] text-slate-400">Keep 100% of vested shares in {simulation.config.symbol}</p>
                </button>

                <button
                  type="button"
                  onClick={() => handleChoice('sell_sp500')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    choices[currentEvent?.quarterIndex] === 'sell_sp500'
                      ? 'bg-emerald-950/80 border-emerald-500 text-white shadow-lg shadow-emerald-950/50'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between font-bold text-emerald-400 mb-1">
                    <span>Sell & S&P 500</span>
                    {choices[currentEvent?.quarterIndex] === 'sell_sp500' && <Check className="w-4 h-4" />}
                  </div>
                  <p className="text-[11px] text-slate-400">Divest proceeds into S&P 500 index ETF (SPY)</p>
                </button>

                <button
                  type="button"
                  onClick={() => handleChoice('split5050')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    choices[currentEvent?.quarterIndex] === 'split5050'
                      ? 'bg-purple-950/80 border-purple-500 text-white shadow-lg shadow-purple-950/50'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between font-bold text-purple-400 mb-1">
                    <span>50/50 Hybrid</span>
                    {choices[currentEvent?.quarterIndex] === 'split5050' && <Check className="w-4 h-4" />}
                  </div>
                  <p className="text-[11px] text-slate-400">Hold 50% stock, reinvest 50% in S&P 500</p>
                </button>
              </div>
            </div>
          </div>

          {/* Stepper Footer Controls */}
          <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between">
            <button
              disabled={safeStep <= 1}
              onClick={() => setCurrentStep(safeStep - 1)}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 transition-all"
            >
              &larr; Previous Quarter
            </button>

            <span className="text-xs text-slate-400 font-medium">
              Click buttons to decide or jump quarters
            </span>

            <button
              disabled={safeStep >= totalQuarters}
              onClick={() => setCurrentStep(safeStep + 1)}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-emerald-500 transition-all flex items-center gap-1"
            >
              Next Quarter &rarr;
            </button>
          </div>
        </div>

        {/* Right: Scoreboard, Prospect Theory Regret Index, & Badges */}
        <div className="space-y-6">
          {/* Scoreboard Card */}
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 shadow-xl">
            <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" /> Game Scorecard Today
            </h3>

            <div className="space-y-3 text-xs">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400 font-medium">Your Strategy Portfolio:</span>
                <span className="text-lg font-extrabold text-white">{formatCurrency(score.userFinalValue)}</span>
              </div>

              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400 font-medium">Hindsight Machine (Optimal):</span>
                <span className="text-base font-bold text-emerald-400">{formatCurrency(score.optimalFinalValue)}</span>
              </div>

              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400 font-medium">Mass Wisdom (Pure S&P 500):</span>
                <span className="text-base font-bold text-slate-200">{formatCurrency(score.sp500FinalValue)}</span>
              </div>
            </div>

            {/* Regret Meter */}
            <div className="mt-5 pt-4 border-t border-slate-800">
              <div className="flex items-center justify-between text-xs font-bold mb-1.5">
                <span className="text-slate-300">Regret Minimization Score</span>
                <span className={`px-2 py-0.5 rounded ${
                  score.regretCategory === 'High Regret'
                    ? 'bg-rose-950 text-rose-400 border border-rose-800'
                    : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                }`}>
                  {score.regretCategory} ({score.regretIndex}%)
                </span>
              </div>

              <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                <div
                  className={`h-full transition-all ${
                    score.regretIndex > 50 ? 'bg-rose-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${100 - score.regretIndex}%` }}
                ></div>
              </div>
              <p className="text-[10px] text-slate-400 mt-1.5">
                Share of the gap between your worst and best possible hold/sell picks (in hindsight, per quarter) that you left on the table. Undecided quarters count as Sell &amp; S&amp;P 500.
              </p>
            </div>
          </div>

          {/* Unlocked Badges */}
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 shadow-xl">
            <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <Award className="w-4 h-4 text-amber-400" /> Strategy Achievement Badges
            </h3>

            <div className="grid grid-cols-1 gap-2.5">
              {score.badges.map(b => (
                <div
                  key={b.id}
                  className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${
                    b.unlocked
                      ? 'bg-slate-950 border-emerald-500/40 text-white'
                      : 'bg-slate-950/40 border-slate-850 opacity-40 text-slate-500'
                  }`}
                >
                  <div className={`p-2 rounded-lg ${b.unlocked ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}>
                    {b.id === 'wisdom_masses' ? <ShieldCheck className="w-4 h-4" /> : b.id === 'diamond_hands' ? <Gem className="w-4 h-4" /> : <Scale className="w-4 h-4" />}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold">{b.title}</h4>
                    <p className="text-[10px] text-slate-400">{b.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
