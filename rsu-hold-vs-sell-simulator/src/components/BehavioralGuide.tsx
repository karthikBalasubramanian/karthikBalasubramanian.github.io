import React from 'react';
import { BookOpen, Users, Lightbulb, AlertTriangle, ShieldCheck, CheckCircle, ArrowRight } from 'lucide-react';

export const BehavioralGuide: React.FC = () => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-emerald-950/80 via-slate-900 to-teal-950/80 border border-emerald-500/30 rounded-2xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-emerald-500/20 text-emerald-400 rounded-xl border border-emerald-500/30 shrink-0">
            <Users className="w-7 h-7" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
              Lifestyle & Wealth Philosophy
            </span>
            <h2 className="text-xl sm:text-2xl font-extrabold text-white mt-1">
              Empowering Employees through the Wisdom of the Masses
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 mt-2 leading-relaxed">
              Financial economic literature consistently recommends immediate divestment of vested equity into broad market index funds. Here is the empirical evidence and mental framework behind making informed RSU decisions.
            </p>
          </div>
        </div>
      </div>

      {/* Core Principles Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {/* Principle 1: Cash Bonus Equivalence */}
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-amber-400">
            <Lightbulb className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">1. The Cash Bonus Equivalence Principle</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Ask yourself: <strong className="text-white">"If my employer deposited a $50,000 cash bonus into my bank account today, would I spend $50,000 buying company stock?"</strong>
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-850 text-xs text-slate-400 leading-relaxed">
            If the answer is no, holding vested RSUs is logically flawed. Holding vested shares is identical to receiving cash and immediately choosing to buy company stock at market price.
          </div>
        </div>

        {/* Principle 2: Double-Dip Concentration Risk */}
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-rose-400">
            <AlertTriangle className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">2. Double-Dip Concentration Risk</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Your human capital (monthly salary, bonus, healthcare, career growth) is already 100% tied to your employer's financial health.
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-850 text-xs text-slate-400 leading-relaxed">
            Holding equity ties your net worth to the same company. If industry downturns strike, you risk losing both your job and your life savings simultaneously.
          </div>
        </div>

        {/* Principle 3: Tax Myth Debunked */}
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-emerald-400">
            <ShieldCheck className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">3. Tax Myth vs. Financial Reality</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            RSUs are taxed as ordinary income on the day they vest. Shares are withheld or sold to satisfy this income tax obligation regardless of whether you hold or sell.
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-850 text-xs text-slate-400 leading-relaxed">
            Because your tax basis is set to the market price on vest date, selling immediately incurs <strong className="text-emerald-400">$0 additional capital gains tax</strong>.
          </div>
        </div>

        {/* Principle 4: Wisdom of the Masses */}
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-sky-400">
            <Users className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">4. Empirical Market Wisdom (S&P 500)</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Academic research shows that <strong className="text-white">over 70% of individual S&P 500 stocks underperform the S&P 500 index itself</strong> over 10-year rolling windows.
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-850 text-xs text-slate-400 leading-relaxed">
            Broad market indexing harnesses the collective innovation of hundreds of top companies. Diversification guarantees you capture the total economic growth of the entire market.
          </div>
        </div>
      </div>

      {/* Actionable Strategy Recommendations */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-emerald-400" /> Evidence-Based Action Plan for Employees
        </h3>

        <div className="space-y-3 text-xs text-slate-300">
          <div className="flex items-start gap-3 p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="font-extrabold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">Strategy A</span>
            <div>
              <strong className="text-white block font-bold mb-0.5">Automated Sell-on-Vest (100% Index Divestment)</strong>
              <span>Set up automatic sell orders on your broker platform (E*TRADE, Fidelity, Schwab) to sell 100% of shares on vest date and transfer to S&P 500 index ETF (SPY/VOO).</span>
            </div>
          </div>

          <div className="flex items-start gap-3 p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="font-extrabold text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded">Strategy B</span>
            <div>
              <strong className="text-white block font-bold mb-0.5">Hybrid 50/50 Rule-Based Cap</strong>
              <span>If you strongly believe in your company's high growth potential, keep 50% in company stock but cap total company stock exposure to less than 15% of your total net worth.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
