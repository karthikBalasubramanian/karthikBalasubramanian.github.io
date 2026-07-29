import React, { useState, useEffect } from 'react';
import { ParentInputs } from '../types';
import { Sparkles, Send, Loader2, Bot, Lightbulb, MessageSquare, Key, Eye, EyeOff, ShieldCheck, ExternalLink, Check, Trash2 } from 'lucide-react';
import { GoogleGenAI } from '@google/genai';

interface AIAdvisorTabProps {
  inputs: ParentInputs;
}

export const AIAdvisorTab: React.FC<AIAdvisorTabProps> = ({ inputs }) => {
  const [userApiKey, setUserApiKey] = useState<string>('');
  const [showKey, setShowKey] = useState<boolean>(false);
  const [isKeySaved, setIsKeySaved] = useState<boolean>(false);
  const [showKeyConfig, setShowKeyConfig] = useState<boolean>(false);

  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<string | null>(null);
  const [responseSource, setResponseSource] = useState<'custom_key' | 'server_key' | 'fallback_engine' | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load saved API key on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('user_gemini_api_key');
    if (savedKey) {
      setUserApiKey(savedKey);
      setIsKeySaved(true);
    }
  }, []);

  const handleSaveKey = () => {
    if (userApiKey.trim()) {
      localStorage.setItem('user_gemini_api_key', userApiKey.trim());
      setIsKeySaved(true);
      setShowKeyConfig(false);
    }
  };

  const handleClearKey = () => {
    localStorage.removeItem('user_gemini_api_key');
    setUserApiKey('');
    setIsKeySaved(false);
  };

  const sampleQuestions = [
    'What if my child decides not to go to college? How does SECURE 2.0 protect my 529 investment?',
    'Can I open both a 529 Plan and a Trump Account ($5k limit) for the same child at the same time?',
    'How do I legally document earned income for a 10-year-old child to qualify for a Custodial Roth IRA?',
    'What are the state tax deduction limits for 529 contributions in my state?',
    'Which account is better if I want to help my child buy their first home at age 25?',
  ];

  const generateFallbackResponse = (query: string): string => {
    const qLower = query.toLowerCase();
    const age = inputs.childCurrentAge;
    const monthly = inputs.monthlyContribution;
    const annual = monthly * 12;

    if (qLower.includes('college') || qLower.includes('secure 2.0') || qLower.includes('roth') || qLower.includes('529')) {
      return `### 🎓 529 Plan & SECURE 2.0 Rollover Strategy for Age ${age}

**1. No-Penalty College Flexibility (SECURE 2.0 Act):**
Under Section 126 of the SECURE 2.0 Act, if your child does not attend college or leaves funds unused, you can roll over up to **$35,000 lifetime** from the 529 plan directly into a **Roth IRA** in the child's name with **ZERO penalty and ZERO tax**.

**2. Key Eligibility Requirements for the Rollover:**
* The 529 account must have been open for at least **15 years**.
* Rolled-over funds (and earnings) must have been in the 529 account for at least **5 years**.
* Rollovers are subject to annual Roth IRA contribution limits (e.g., $7,000/year in 2024/2025).
* The child must have compensation/earned income equal to or exceeding the rollover amount in that year.

**3. Action Plan for $${monthly}/month ($${annual}/year):**
* Keep contributing $${monthly}/mo to a 529 plan now while child is age ${age}.
* By age 18, you will have accumulated significant compound growth.
* If college is not needed, initiate annual $7,000 rollovers to their Roth IRA over a 5-year span ($35k max). This seeds $35,000 into a tax-free Roth IRA that can grow to over **$500,000+** by their retirement!`;
    }

    if (qLower.includes('trump') || qLower.includes('custodial') || qLower.includes('$5k') || qLower.includes('both')) {
      return `### 🇺🇸 Dual Strategy: 529 Plan + Trump Account / Custodial IRA

**1. Can you open both?**
**Yes!** A 529 Plan and a Trump Account ($5,000 annual limit child savings account / Custodial IRA) serve complementary goals:

* **529 Plan:** Best for tax-free growth targeted at education or eventual SECURE 2.0 $35k Roth IRA rollover.
* **Trump Account ($5,000 limit):** Dedicated child savings account growing tax-deferred up to age 18, which can then be rolled over into a Traditional or Roth IRA (growing tax-free until retirement at age 60/65).

**2. Optimal Contribution Split for $${monthly}/month ($${annual}/year):**
* **Option A (Education Focus):** $${Math.round(monthly * 0.7)}/mo into 529 Plan + $${Math.round(monthly * 0.3)}/mo into Trump Account.
* **Option B (Retirement & Home Focus):** Maximize Trump Account up to $5,000/yr ($416/mo) if your primary goal is wealth transfer at age 18, and put remaining in 529 or UTMA.`;
    }

    if (qLower.includes('earned income') || qLower.includes('document') || qLower.includes('roth') || qLower.includes('work')) {
      return `### 💼 Documenting Earned Income for Child Custodial Roth IRA

**1. What Counts as Eligible Child Earned Income?**
To contribute to a Custodial Roth IRA, the child must have legitimate earned income from work performed.
* **Legitimate Jobs:** Modeling, family business tasks (e.g. website management, social media, cleaning offices), babysitting, lawn care, tutoring.
* **Unearned Income (Does NOT count):** Allowances for normal household chores, dividends, interest, or monetary gifts.

**2. Documentation Checklist for IRS Audits:**
1. **Detailed Logbook:** Record date, task description, hours worked, and fair market hourly wage rate.
2. **Fair Market Value:** Pay reasonable market rates (e.g., $12–$15/hr for office filing/cleaning; do NOT pay $500/hr for taking out trash).
3. **Form W-2 or 1099:** If working for family business, issue a formal W-2 or 1099-NEC.
4. **Separate Bank Account:** Deposit wages into child's checking/savings account before transferring to the Custodial Roth IRA.`;
    }

    return `### 💡 Personalized Child Wealth Recommendation (Age ${age}, $${monthly}/mo)

**1. Recommended Core Allocation:**
* **70% Core ($${Math.round(monthly * 0.7)}/mo):** 529 College Savings Plan — State tax benefits and tax-free growth.
* **20% Trump/Custodial IRA ($${Math.round(monthly * 0.2)}/mo):** Long-term compounding with rollover options at age 18.
* **10% UTMA / Brokerage ($${Math.round(monthly * 0.1)}/mo):** Flexible cash for non-education expenses (sports, car, first apartment).

**2. State Tax Tip (${inputs.state || 'General US'}):**
Check if your state offers a state income tax deduction or credit for contributing to your home state's 529 plan!

*(Tip: Enter your Google Gemini API key above for live real-time AI reasoning on any custom question!)*`;
  };

  const handleAskAI = async (queryText?: string) => {
    const textToSubmit = queryText || prompt;
    if (!textToSubmit.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setResponseSource(null);

    const systemInstruction = `You are a certified financial planner specializing in child investment accounts, education funding, tax strategy, and early wealth building in the United States.
You provide clear, objective, accurate, actionable advice covering:
1. 529 College Savings Plans (tax-free growth for education, SECURE 2.0 Act rollover up to $35k to Roth IRA after 15 years, state tax deductions).
2. Trump Accounts / Child Savings Accounts / Custodial IRA accounts ($5,000 yearly contribution limit, growth up to age 18, rollover strategies to Traditional or Roth IRA up to age 60/65).
3. UTMA / UGMA Custodial Accounts (flexibility, Kiddie tax thresholds, control at age 18/21).
4. Custodial Roth IRAs (requires child earned income, 100% tax-free growth, withdrawal rules).
5. Coverdell ESAs ($2,000/yr limit, education focus).
6. Taxable Brokerage / Trust accounts.

Keep your tone warm, professional, encouraging, and structured. Use formatting like bullet points and concise bold titles. Do not give binding legal tax advice, but explain financial rules clearly.`;

    const promptText = `Parent Scenario Details:
- Child Current Age: ${inputs.childCurrentAge} years old
- Monthly Contribution Budget: $${inputs.monthlyContribution}/month ($${inputs.monthlyContribution * 12}/year)
- Primary Goal: ${inputs.primaryGoal}
- State of Residence: ${inputs.state || 'Not specified'}

User Question: ${textToSubmit}`;

    // 1. First priority: User's custom Gemini API key entered directly in UI
    const activeUserKey = userApiKey.trim() || localStorage.getItem('user_gemini_api_key');

    if (activeUserKey) {
      try {
        const ai = new GoogleGenAI({ apiKey: activeUserKey });
        const res = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: promptText,
          config: {
            systemInstruction,
            temperature: 0.7,
          },
        });

        if (res.text) {
          setResponse(res.text);
          setResponseSource('custom_key');
          setLoading(false);
          return;
        }
      } catch (err: any) {
        console.warn('Client-side Gemini API call failed with custom key:', err);
        // Fall through to try server or fallback
      }
    }

    // 2. Second priority: Express backend server endpoint (if running fullstack)
    try {
      const res = await fetch('/api/ai-advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: textToSubmit,
          childAge: inputs.childCurrentAge,
          monthlyContribution: inputs.monthlyContribution,
          investmentGoal: inputs.primaryGoal,
          state: inputs.state,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.text) {
          setResponse(data.text);
          setResponseSource('server_key');
          setLoading(false);
          return;
        }
      }
    } catch (err) {
      // Backend unavailable or 404 (e.g. GitHub Pages static host)
    }

    // 3. Third priority: Smart Instant Built-in Financial Advisor Engine
    setResponse(generateFallbackResponse(textToSubmit));
    setResponseSource('fallback_engine');
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> AI Child Wealth Strategist (Gemini AI)
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
              Ask the AI Financial Advisor
            </h2>
            <p className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm mt-1.5 leading-relaxed">
              Get instant, tailored advice on tax strategies, state-specific 529 deductions, SECURE 2.0 Roth rollovers, and child earned income compliance.
            </p>
          </div>

          <button
            id="toggle-key-settings-button"
            onClick={() => setShowKeyConfig(!showKeyConfig)}
            className={`px-4 py-2.5 rounded-xl border text-xs font-bold flex items-center gap-2 shrink-0 transition-all ${
              isKeySaved
                ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300'
                : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
            }`}
          >
            <Key className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            {isKeySaved ? 'API Key Active (Private)' : 'Configure Gemini API Key'}
            {isKeySaved && <Check className="w-3.5 h-3.5 text-emerald-600" />}
          </button>
        </div>

        {/* API Key Configuration Drawer */}
        {(showKeyConfig || (!isKeySaved && false)) && (
          <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800 space-y-4">
            <div className="p-4 rounded-xl bg-indigo-50/70 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/50 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-indigo-900 dark:text-indigo-200 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-indigo-600" /> Private &amp; Secure Client-Side API Key
                </span>
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] font-bold text-indigo-600 dark:text-indigo-400 hover:underline inline-flex items-center gap-1"
                >
                  Get Free Gemini Key <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <p className="text-[11px] text-indigo-800 dark:text-indigo-300 leading-relaxed">
                Enter your Google Gemini API key below to interact directly with Gemini from your browser. Your key is stored strictly in your browser's local storage and is never saved to any external server.
              </p>

              <div className="flex flex-col sm:flex-row items-center gap-2">
                <div className="relative w-full">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={userApiKey}
                    onChange={(e) => setUserApiKey(e.target.value)}
                    placeholder="AIzaSy..."
                    className="w-full text-xs p-3 pr-10 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-white font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey(!showKey)}
                    className="absolute right-3 top-3 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                  >
                    {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto shrink-0">
                  <button
                    id="save-user-api-key-button"
                    onClick={handleSaveKey}
                    disabled={!userApiKey.trim()}
                    className="w-full sm:w-auto px-4 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs transition-colors disabled:opacity-50"
                  >
                    Save Key
                  </button>
                  {isKeySaved && (
                    <button
                      id="clear-user-api-key-button"
                      onClick={handleClearKey}
                      className="p-3 rounded-lg bg-red-100 dark:bg-red-950/60 text-red-600 dark:text-red-400 hover:bg-red-200 text-xs transition-colors"
                      title="Clear saved key"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-4">
        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center justify-between">
          <span className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-indigo-600" /> What would you like to ask?
          </span>
          <span className="text-[10px] text-slate-400">
            {isKeySaved ? '⚡ Connected to Gemini API (Custom Key)' : '💡 Smart Mode (No API key required)'}
          </span>
        </label>

        <div className="relative">
          <textarea
            id="ai-prompt-textarea"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. Can I contribute to both a 529 and a Trump account? What happens at age 18?"
            className="w-full text-xs sm:text-sm p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
          />
          <button
            id="submit-ai-prompt-button"
            onClick={() => handleAskAI()}
            disabled={loading || !prompt.trim()}
            className="absolute bottom-3 right-3 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            Ask Advisor
          </button>
        </div>

        {/* Sample Question Chips */}
        <div className="space-y-2 pt-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <Lightbulb className="w-3.5 h-3.5 text-amber-500" /> Common Questions:
          </span>
          <div className="flex flex-wrap gap-2">
            {sampleQuestions.map((q, idx) => (
              <button
                key={idx}
                id={`sample-question-${idx}`}
                onClick={() => {
                  setPrompt(q);
                  handleAskAI(q);
                }}
                className="text-xs text-left px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors"
              >
                "{q}"
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
          <div className="text-sm font-bold text-slate-900 dark:text-white">
            Analyzing your scenario...
          </div>
          <p className="text-xs text-slate-500">
            Evaluating tax rules, 529 state limits, and rollover math for age {inputs.childCurrentAge}...
          </p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs">
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-emerald-600" />
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                AI Financial Advisor Recommendation
              </h3>
            </div>
            <span className="text-[10px] font-mono px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
              {responseSource === 'custom_key' && '⚡ Live Gemini AI (Your API Key)'}
              {responseSource === 'server_key' && '⚡ Live Gemini AI (Server)'}
              {responseSource === 'fallback_engine' && '💡 Smart Financial Advisor Knowledge Base'}
            </span>
          </div>

          <div className="prose dark:prose-invert max-w-none text-xs sm:text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap leading-relaxed">
            {response}
          </div>
        </div>
      )}
    </div>
  );
};

